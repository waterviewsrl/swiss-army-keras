'''PP-MobileSeg (Baidu PaddleSeg, 2023, "PP-MobileSeg: Explore the Fast and
Accurate Semantic Segmentation Model on Mobile Devices",
arXiv:2304.05152).

Architecture verified directly against the official PaddleSeg source
(`paddleseg/models/pp_mobileseg.py`), same rigor as every other decoder in
this plan. **AAM lives here, not in `_backbone_strideformer.py`** — a
deliberate departure from the reference's own module boundary (the
reference's `StrideFormer.forward()` runs AAM internally, producing one
already-fused feature its own trivial head consumes); see
`_backbone_strideformer.py`'s module docstring for the full reasoning.
This file's `pp_mobileseg(...)` pulls a 3-level pyramid via `backbone_zoo`
(any registered backbone with at least 3 taps, `feature_levels`-overridable
same as `pp_liteseg`) and applies AAM itself, so this decoder still
reproduces the paper's real pipeline end to end when paired with
`StrideFormer-Tiny`/`-Base` — while both pieces stay independently reusable
elsewhere in this repo.

- **AAM has two real variants, not one, and they are not interchangeable**
  (confirmed from the reference's own per-variant training configs — Tiny
  pairs with the full variant, Base with the simplified one):
  - **`'full'`** (`InjectionMultiSumallmultiallsum`): every one of the 3
    input levels gets both an "embed" 1x1 conv+BN and a separate "activate"
    1x1 conv+BN+sigmoid; `out = sigmoid(act(t8))·sigmoid(act(t16))·
    sigmoid(act(t32)) · (embed(t8)+embed(t16)) + embed(t32)` — the two
    finer levels are gated by the product of *all three* levels' sigmoid
    activations (deep, semantic features act as a learned veto suppressing
    noisy shallow detail) before being added to the coarsest/most-semantic
    level, itself added back unweighted as a residual.
  - **`'simplified'`** (`InjectionMultiSumallmultiallsumSimpx8`): drops
    `t8`'s own "activate" branch and `t16`'s own "embed" branch entirely
    (not just zeroes them — the sub-layers don't exist in this variant):
    `out = sigmoid(act(t16))·sigmoid(act(t32)) · embed(t8) + embed(t32)`.
  Both are plain elementwise `Multiply`/`Add` chains between already-
  channel-matched (`out_channels`-wide) tensors — no channel-reduction
  step, so no custom `Layer` subclass is needed here (unlike PIDNet's
  PagFM, §3.4, which needs one for its channel-sum attention map).
- **The decode head is intentionally trivial** (`PPMobileSegHead`): one
  1x1 conv+BN+ReLU (`linear_fuse`, optionally depthwise via `head_use_dw`)
  → dropout → a bare 1x1 classifier conv (`conv_seg`, bias, no BN/act) — no
  further multi-scale fusion happens here, all of it already happened in
  AAM.
- **VIM (the reference's inference-only latency optimization) is not
  ported** — already recommended against in plan.md §3.5: dynamic-shape
  `unique`/`gather` ops and a changed output contract (a class-index map,
  not logits) fight this repo's TFLite-exportability goal. Only the
  reference's `upsample='intepolate'` path is implemented (a plain bilinear
  upsample to the input's spatial resolution, the same "decoder heads
  upsample internally" convention every other decoder in this repo already
  follows, filling the same gap the reference itself leaves to its own
  training/export code).

Follows the `pp_liteseg`/`lr_aspp` precedent for encoder wiring via
`backbone_zoo` and `tfmot.quantization.keras.quantize_annotate_layer`
wrapping for QAT parity.
'''

from __future__ import absolute_import

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dropout,
    Layer,
    Multiply,
    UpSampling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.utils import register_keras_serializable

import tensorflow_model_optimization as tfmot

from swiss_army_keras.layer_utils import qat_annotate_layer
from swiss_army_keras._backbone_zoo import backbone_zoo


@register_keras_serializable(package='swiss_army_keras')
class BilinearResize(Layer):
    '''Resize to an explicit `(height, width)` target via `tf.image.resize`
    - unlike `UpSampling2D` (used everywhere else in this repo, where every
    resize target is always an integer multiple *larger* than its input),
    AAM's `'full'` variant needs to resize `t8` *down* by half
    (`F.interpolate(inputs[0], scale_factor=0.5, ...)` in the reference) -
    `UpSampling2D`'s `size` argument only accepts a positive integer
    upsample multiplier, it cannot express a downsample. A proper `Layer`
    subclass (not a raw functional call), the usual QAT-model-cloning-
    safety reason.
    '''

    def __init__(self, target_height, target_width, **kwargs):
        super().__init__(**kwargs)
        self.target_height = target_height
        self.target_width = target_width

    def call(self, x):
        return tf.image.resize(x, size=(self.target_height, self.target_width), method='bilinear')

    def get_config(self):
        config = super().get_config()
        config.update({'target_height': self.target_height, 'target_width': self.target_width})
        return config


def _embed(x, out_channels, name=''):
    x = qat_annotate_layer(Conv2D(
        out_channels, 1, padding='same', use_bias=False, kernel_initializer=HeNormal(), name=f'{name}_conv',
    ))(x)
    return qat_annotate_layer(BatchNormalization(name=f'{name}_bn'))(x)


def _activate(x, out_channels, name=''):
    x = _embed(x, out_channels, name=name)
    return qat_annotate_layer(Activation('sigmoid', name=f'{name}_sigmoid'))(x)


def _resize_to(x, target_h, target_w, name=''):
    if x.shape[1] == target_h and x.shape[2] == target_w:
        return x
    # Not qat_annotate_layer-wrapped: tfmot's default QAT scheme has no
    # QuantizeConfig for a custom Layer it doesn't recognize (confirmed -
    # quantize_apply raises `RuntimeError: ...BilinearResize is not
    # supported`), the same class of "leave the unsupported layer type
    # unannotated" finding already made for Add/Multiply/Concatenate and
    # standalone pre-activation BN elsewhere in this codebase.
    return BilinearResize(target_h, target_w, name=name)(x)


def aam_full(t8, t16, t32, out_channels, name='aam'):
    '''`InjectionMultiSumallmultiallsum` - see module docstring.'''
    target_h, target_w = t8.shape[1] // 2, t8.shape[2] // 2  # confirmed from source: t8 is halved first, landing on t16's resolution

    low1 = _resize_to(t8, target_h, target_w, name=f'{name}_low1_down')
    low1_act = _activate(low1, out_channels, name=f'{name}_low1_act')
    low1_embed = _embed(low1, out_channels, name=f'{name}_low1_embed')

    low2 = _resize_to(t16, target_h, target_w, name=f'{name}_low2_resize')
    low2_act = _activate(low2, out_channels, name=f'{name}_low2_act')
    low2_embed = _embed(low2, out_channels, name=f'{name}_low2_embed')

    high_act = _activate(t32, out_channels, name=f'{name}_high_act')
    high_act = _resize_to(high_act, target_h, target_w, name=f'{name}_high_act_resize')
    high_embed = _embed(t32, out_channels, name=f'{name}_high_embed')
    high_embed = _resize_to(high_embed, target_h, target_w, name=f'{name}_high_embed_resize')

    gate = Multiply(name=f'{name}_gate_mul1')([low1_act, low2_act])
    gate = Multiply(name=f'{name}_gate_mul2')([gate, high_act])

    low_sum = Add(name=f'{name}_low_sum')([low1_embed, low2_embed])
    gated = Multiply(name=f'{name}_gated')([gate, low_sum])
    return Add(name=f'{name}_out')([gated, high_embed])


def aam_simplified(t8, t16, t32, out_channels, name='aam'):
    '''`InjectionMultiSumallmultiallsumSimpx8` - see module docstring. Drops
    `t8`'s own activate branch and `t16`'s own embed branch entirely
    relative to `aam_full`, not just zeroes them out.

    **Output resolution is `t8`'s own (the finest), not `t16`'s** - the
    opposite of `aam_full`, which halves `t8` down to `t16`'s resolution.
    Confirmed from source: `low_feat1 = embed(inputs[0])` (no downsampling
    of `t8` at all), `low_feat2 = interpolate(inputs[1],
    size=low_feat1.shape)` (`t16` *upsampled* to `t8`'s resolution instead)
    - easy to get backwards by assuming symmetry with `aam_full`, which
    this implementation's first draft did (a real bug found while
    verifying against the raw source rather than trusting that
    assumption).
    '''
    target_h, target_w = t8.shape[1], t8.shape[2]

    low1_embed = _embed(t8, out_channels, name=f'{name}_low1_embed')

    low2 = _resize_to(t16, target_h, target_w, name=f'{name}_low2_resize')
    low2_act = _activate(low2, out_channels, name=f'{name}_low2_act')

    high_act = _activate(t32, out_channels, name=f'{name}_high_act')
    high_act = _resize_to(high_act, target_h, target_w, name=f'{name}_high_act_resize')
    high_embed = _embed(t32, out_channels, name=f'{name}_high_embed')
    high_embed = _resize_to(high_embed, target_h, target_w, name=f'{name}_high_embed_resize')

    gate = Multiply(name=f'{name}_gate_mul')([low2_act, high_act])
    gated = Multiply(name=f'{name}_gated')([gate, low1_embed])
    return Add(name=f'{name}_out')([gated, high_embed])


_AAM_VARIANTS = {'full': aam_full, 'simplified': aam_simplified}


def pp_mobileseg_head(x, n_labels, use_dw=False, dropout_ratio=0.1, name='head'):
    channels = x.shape[-1]
    x = qat_annotate_layer(Conv2D(
        channels, 1, padding='same', use_bias=False, groups=channels if use_dw else 1,
        kernel_initializer=HeNormal(), name=f'{name}_linear_fuse_conv',
    ))(x)
    x = qat_annotate_layer(BatchNormalization(name=f'{name}_linear_fuse_bn'))(x)
    x = qat_annotate_layer(Activation('relu', name=f'{name}_linear_fuse_act'))(x)
    x = Dropout(dropout_ratio, name=f'{name}_dropout')(x)
    return qat_annotate_layer(Conv2D(n_labels, 1, padding='same', use_bias=True, name=f'{name}_conv_seg'))(x)


def pp_mobileseg(input_tensor, n_labels, feature_levels=(1, 2, 3), aam_variant='full', aam_out_channels=256,
                 head_use_dw=True, backbone=None, weights='imagenet', freeze_backbone=True,
                 freeze_batch_norm=True, name='pp_mobileseg'):
    '''PP-MobileSeg (Tang et al., 2023) segmentation decoder.

    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Input((None, None, 3))`.
        n_labels: number of segmentation classes.
        feature_levels: a 3-tuple of 1-indexed encoder feature levels to
                        use as the (x8, x16, x32) inputs to AAM - default
                        `(1, 2, 3)` matches `StrideFormer-Tiny`/`-Base`'s
                        own genuine 3-tap pyramid (stride 8/16/32, no
                        earlier taps exist to skip).
        aam_variant: `'full'` (`InjectionMultiSumallmultiallsum`, pairs
                     with `StrideFormer-Tiny` in the reference) or
                     `'simplified'` (`InjectionMultiSumallmultiallsumSimpx8`,
                     pairs with `StrideFormer-Base`) - these are not
                     interchangeable defaults, see module docstring.
        aam_out_channels: AAM's fused output channel count (256 in the
                          reference, for both variants).
        head_use_dw: whether `pp_mobileseg_head`'s `linear_fuse` conv is
                     depthwise (`True` in the reference).
        backbone: the name of a backbone registered in
                  `swiss_army_keras._backbone_zoo.BACKBONE_REGISTRY` with at
                  least 3 taps - unlike `pp_liteseg`/`lr_aspp`, this decoder
                  has no `backbone=None` plain-conv fallback (the reference
                  itself has none either; this decoder's whole design is
                  built around a swappable pyramid backbone feeding AAM).
        weights: one of `None`, `'imagenet'`, or a path to a weights file.
        freeze_backbone: `True` to freeze the backbone's weights.
        freeze_batch_norm: `True` to also freeze the backbone's BN layers.
        name: prefix of the created Keras model and its layers.

    Output
    ----------
        A `tf.keras.Model` with a `.preprocessing` attribute, producing an
        `(H, W, n_labels)` logit map at the input's spatial resolution.
    '''
    if backbone is None:
        raise ValueError(
            '`pp_mobileseg` has no backbone=None fallback (see docstring) - pass a backbone name '
            'registered in `BACKBONE_REGISTRY`, e.g. backbone="StrideFormer-Tiny".'
        )
    if aam_variant not in _AAM_VARIANTS:
        raise ValueError(f"aam_variant must be one of {sorted(_AAM_VARIANTS)}, got {aam_variant!r}")

    backbone_ = backbone_zoo(backbone, weights, input_tensor, max(feature_levels), freeze_backbone, freeze_batch_norm)
    x_encoder = backbone_([input_tensor, ])
    t8, t16, t32 = (x_encoder[i - 1] for i in feature_levels)

    aam_fn = _AAM_VARIANTS[aam_variant]
    fused = aam_fn(t8, t16, t32, aam_out_channels, name=f'{name}_aam')

    x = pp_mobileseg_head(fused, n_labels, use_dw=head_use_dw, name=f'{name}_head')

    x = qat_annotate_layer(UpSampling2D(
        size=(input_tensor.shape[1] // x.shape[1], input_tensor.shape[2] // x.shape[2]),
        interpolation='bilinear', name=f'{name}_upsample_to_input',
    ))(x)

    m = Model([input_tensor, ], [x, ])

    with tfmot.quantization.keras.quantize_scope():
        m = tfmot.quantization.keras.quantize_apply(m)

    m.preprocessing = backbone_.preprocessing

    return m
