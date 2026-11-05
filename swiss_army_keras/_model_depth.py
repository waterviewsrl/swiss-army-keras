
'''Monocular (single-image) depth estimation (plan.md §13.5, Phase 11).

Two genuinely different head architectures, not one reuse pattern wearing
two names:

- **`lr_aspp_depth`**: the cheap, generic-reuse route. `lr_aspp`'s (and
  `pp_liteseg`'s/`deeplab_v3_plus`'s/`ddrnet`'s/`pp_mobileseg`'s) final
  layer is already a plain, activation-less `Conv2D(n_labels, ...)` — a
  valid depth-regression output with `n_labels=1` and zero architecture
  changes. This wraps the existing, unmodified `lr_aspp` (this repo's
  cheapest decoder) with `n_labels=1` and adds the depth-specific
  output-convention choice (`output_mode`, below) as a post-hoc step,
  entirely outside `lr_aspp`'s own `quantize_apply` call.
- **`fastdepth_head`**: a genuinely different, purpose-built decoder
  topology — FastDepth's own NNConv5-style cascade (Wofk, Ma, Karaman,
  Sze, "FastDepth: Fast Monocular Depth Estimation on Embedded Systems",
  ICRA 2019, arXiv:1903.03273 — demonstrated real-time on a Jetson TX2,
  directly relevant to this repo's own edge-deployment framing): a
  depthwise-separable conv followed by a nearest-neighbor 2x upsample,
  cascaded, with **additive** (not concatenative — a real, verified detail,
  not assumed to match `lr_aspp`/`pp_liteseg`'s own concat-based skips)
  skip connections from 3 middle encoder taps.

**`output_mode='sigmoid'|'linear'`, exposed on both** — the depth-vs-
inverse-depth fork this repo's own research flagged, not silently resolved:
`'sigmoid'` (the default) bounds the output to `[0, 1]`, following the same
precedent `pose_regression_head` already established (`_model_classifier.py`,
plan.md §13.3): a bounded target is easier to train and to quantize than
raw, scene-dependent, unbounded metric depth, whose dynamic range a single
int8 scale must span entirely. The deployer is responsible for the
depth-range convention this implies (e.g. treating the `[0,1]` output as
normalized inverse depth against a known `d_min`) — not fixed by this
file, since it's genuinely deployment-dependent. `'linear'` returns raw,
unbounded regression output for callers who'd rather handle scale
themselves.

**Sigmoid and the Axis ARTPEC-8 DLPU.** The sigmoid operator (`LOGISTIC`)
is not on the ARTPEC-8 operator list as published, so a model exported with
its sigmoid fails the export check for that profile. The sigmoid has no
weights, so the fix is at export time and needs no retraining: pass the
trained model through `deploy_utils.drop_output_activations` (or build the
`Quantizer` with `drop_output_activations=True`). The exported model then
returns the value before the sigmoid, and the application applies the
sigmoid itself, for instance with a 256-entry lookup table on the uint8
output.

**Feature levels.** Both heads adapt to the number of feature levels of
the backbone: five for most, four for RepViT and FastViT, three for
StrideFormer.
'''

from __future__ import absolute_import

from swiss_army_keras.layer_utils import *
from swiss_army_keras._backbone_zoo import backbone_zoo
from swiss_army_keras._model_lr_aspp import lr_aspp

from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    DepthwiseConv2D,
    UpSampling2D,
)
from tensorflow.keras.models import Model

import tensorflow as tf
import tensorflow_model_optimization as tfmot


def _valid_output_mode(output_mode):
    if output_mode not in ('sigmoid', 'linear'):
        raise ValueError(f"Unknown output_mode '{output_mode}' - expected 'sigmoid' or 'linear'.")


def lr_aspp_depth(input_tensor, output_mode='sigmoid', backbone=None, weights='imagenet',
                  freeze_backbone=True, freeze_batch_norm=True, name='lr_aspp_depth', **lr_aspp_kwargs):
    '''Monocular depth via `lr_aspp(n_labels=1)`, unmodified, plus a
    post-hoc output-convention step.

    **Post-hoc, not baked into `lr_aspp` itself**: `lr_aspp` already ends
    with its own single `quantize_apply` call; adding a sigmoid *outside*
    that (a plain, un-annotated `tf.sigmoid` on the already-built model's
    output, wrapped in a fresh `Model(...)`, no second `quantize_apply`) is
    the same established pattern `layer_utils.add_energy_score_output`/
    `uint8_input_wrapper` already use for small float32 math after an
    already-quantized model with no trainable weights and no need to
    re-quantize — and avoids touching `lr_aspp` itself at all.

    Input
    ----------
        input_tensor, output_mode: see module docstring.
        backbone, weights, freeze_backbone, freeze_batch_norm, name,
        **lr_aspp_kwargs: passed straight through to `lr_aspp` (e.g.
            `deep_layer`/`shallow_layer`/`num_filters` overrides, or
            `gate=False` for the gate-free module). The feature levels
            default to ones that exist for the chosen backbone.

    Output
    ----------
        A `tf.keras.Model` with a `.preprocessing` attribute, producing an
        `(H, W, 1)` depth map at the input's spatial resolution.
    '''
    _valid_output_mode(output_mode)

    model = lr_aspp(
        input_tensor, n_labels=1, backbone=backbone, weights=weights,
        freeze_backbone=freeze_backbone, freeze_batch_norm=freeze_batch_norm,
        name=name, **lr_aspp_kwargs,
    )

    if output_mode == 'linear':
        return model

    depth = tf.sigmoid(model.output)
    wrapped = Model(model.input, depth, name=f'{name}_sigmoid')
    wrapped.preprocessing = model.preprocessing
    return wrapped


def _depthwise_sep_upsample_block(x, out_channels, name):
    '''FastDepth's own "NNConv5" decoder block: a depthwise-separable 5x5
    conv (halving/reducing channels toward `out_channels`) followed by a
    nearest-neighbor 2x upsample — deliberately convolving *before*
    upsampling, keeping the conv cost on the smaller, pre-upsample spatial
    resolution.
    '''
    x = qat_annotate_layer(DepthwiseConv2D(
        5, padding='same', use_bias=False, name=f'{name}_dw'))(x)
    x = qat_annotate_layer(BatchNormalization(name=f'{name}_dw_bn'))(x)
    x = qat_annotate_layer(Activation('relu', name=f'{name}_dw_act'))(x)
    x = qat_annotate_layer(Conv2D(
        out_channels, 1, padding='same', use_bias=False, name=f'{name}_pw'))(x)
    x = qat_annotate_layer(BatchNormalization(name=f'{name}_pw_bn'))(x)
    x = qat_annotate_layer(Activation('relu', name=f'{name}_pw_act'))(x)
    x = qat_annotate_layer(UpSampling2D(
        size=2, interpolation='nearest', name=f'{name}_upsample'))(x)
    return x


def _conv_bn(x, out_channels, name):
    x = qat_annotate_layer(Conv2D(
        out_channels, 1, padding='same', use_bias=False, name=f'{name}_conv'))(x)
    x = qat_annotate_layer(BatchNormalization(name=f'{name}_bn'))(x)
    return x


def fastdepth_decoder(features, input_tensor, output_mode='sigmoid',
                      decoder_channels=(256, 128, 64, 32, 16), name='fastdepth'):
    '''The FastDepth decoder on an existing list of encoder features.

    This is the part of `fastdepth_head` that follows the encoder. It is
    exposed so that several heads can share one backbone (see `multi_task`).
    The layers are annotated for quantization-aware training but
    `quantize_apply` is left to the caller.

    The cascade starts from the deepest feature and doubles the resolution
    once per entry of `decoder_channels`, stopping early if the input
    resolution is reached. After each step, one of the three next-deepest
    features is added if its resolution matches. A final bilinear resize
    covers any remaining gap to the input resolution.

    Input
    ----------
        features: list of encoder feature tensors, shallowest first (at
                  least one).
        input_tensor: the model input, used for the output resolution.
        output_mode, decoder_channels, name: see `fastdepth_head`.

    Output
    ----------
        The `(H, W, 1)` depth tensor at the input resolution.
    '''
    _valid_output_mode(output_mode)

    features = list(features)
    x = features[-1]
    # FastDepth's own "3 middle encoder layers", additive fusion; fewer on
    # backbones that expose fewer levels.
    skips = {int(t.shape[1]): t for t in features[-2::-1][:3]}

    height = int(input_tensor.shape[1])
    for i, out_channels in enumerate(decoder_channels):
        if int(x.shape[1]) * 2 > height:
            break
        x = _depthwise_sep_upsample_block(x, out_channels, name=f'{name}_up{i}')
        skip = skips.pop(int(x.shape[1]), None)
        if skip is not None:
            skip_proj = _conv_bn(skip, out_channels, name=f'{name}_skip{i}')
            x = Add(name=f'{name}_skip{i}_add')([x, skip_proj])

    raw = qat_annotate_layer(Conv2D(
        1, 1, padding='same', use_bias=True, name=f'{name}_depth_out'))(x)

    # Safety net, not an assumption: verify/fix the final resolution from
    # real shapes, not an assumed 2**5 stride product - backbones differ in
    # the stride of their deepest level.
    if int(raw.shape[1]) != height:
        raw = qat_annotate_layer(UpSampling2D(
            size=(input_tensor.shape[1] // raw.shape[1], input_tensor.shape[2] // raw.shape[2]),
            interpolation='bilinear', name=f'{name}_final_upsample',
        ))(raw)

    # The sigmoid is the last layer, so that it can be dropped at export
    # for accelerators that do not run it.
    if output_mode == 'sigmoid':
        raw = qat_annotate_layer(Activation('sigmoid', name=f'{name}_sigmoid'))(raw)

    return raw


def fastdepth_head(input_tensor, output_mode='sigmoid', decoder_channels=(256, 128, 64, 32, 16),
                   backbone='MobileNetV2', weights='imagenet', freeze_backbone=True,
                   freeze_batch_norm=True, name='fastdepth'):
    '''FastDepth-style monocular depth decoder (Wofk et al. 2019) — a
    genuinely different topology from `lr_aspp_depth`, not another
    reuse of the same decoder: an NNConv5 cascade (depthwise-separable
    conv + nearest-neighbor 2x upsample, repeated) with **additive**
    skip connections from up to 3 middle encoder taps, verified from the
    paper directly (not assumed to match this repo's other, concat-based
    decoders' own skip-fusion convention).

    No `backbone=None` no-backbone fallback (unlike `lr_aspp`/`pp_liteseg`)
    — FastDepth's whole design is an encoder-decoder pair, not something
    with a documented plain-conv-encoder equivalent, so one isn't invented
    here.

    Input
    ----------
        input_tensor, output_mode: see module docstring.
        decoder_channels: per-stage output channel count for each of the
                          cascaded upsample blocks, deepest-to-shallowest
                          (`(256, 128, 64, 32, 16)` default). Blocks beyond
                          the input resolution are not built.
        backbone: name of a backbone registered in
                  `swiss_army_keras._backbone_zoo.BACKBONE_REGISTRY`
                  (`'MobileNetV2'` default — FastDepth's own reference
                  encoder family). Backbones with fewer than five feature
                  levels are supported: the cascade uses the levels that
                  exist.
        weights, freeze_backbone, freeze_batch_norm, name: as every other
            `backbone_zoo`-based model in this repo.

    Output
    ----------
        A `tf.keras.Model` with a `.preprocessing` attribute, producing an
        `(H, W, 1)` depth map at the input's spatial resolution.
    '''
    _valid_output_mode(output_mode)

    backbone_ = backbone_zoo(backbone, weights, input_tensor, 5, freeze_backbone, freeze_batch_norm)
    features = backbone_([input_tensor, ])

    raw = fastdepth_decoder(
        features, input_tensor, output_mode=output_mode, decoder_channels=decoder_channels, name=name)

    m = Model([input_tensor, ], [raw, ])

    with tfmot.quantization.keras.quantize_scope():
        m = tfmot.quantization.keras.quantize_apply(m)

    m.preprocessing = backbone_.preprocessing

    return m
