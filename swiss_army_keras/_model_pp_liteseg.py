
'''PP-LiteSeg with UAFM (Liu et al., 2022, "PP-LiteSeg: A Superior
Real-Time Semantic Segmentation Model", arXiv:2204.02681).

Architecture verified directly against the official PaddleSeg source
(`paddleseg/models/pp_liteseg.py`, `paddleseg/models/layers/tensor_fusion.py`,
`tensor_fusion_helper.py` — PaddlePaddle/PaddleSeg, release/2.8), not the
paper text alone, per plan.md §7's "verify against the real source, don't
guess" principle (see plan.md §3.2 for the full research findings this was
built from):

- **SPPM** (`PPContextModule`): for each of `bin_sizes=(1, 2, 4)`, an
  adaptive-average-pool down to a `bin_size x bin_size` grid, a shared-shape
  1x1 Conv+BN+ReLU (every scale projects straight to `out_channels=128`, no
  per-scale channel reduction), upsampled back to the input's spatial size.
  Combined by **element-wise sum across scales** (not concatenation, unlike
  classic ASPP/PSPM), then one more 3x3 Conv+BN+ReLU.
- **UAFM** (`UAFM_SpAtten`, PP-LiteSeg's actual default arm type — confirmed
  from the executable `PPLiteSeg.__init__` default, not the docstring, which
  names a stale variant): the low-level ("detail") skip feature is
  channel-projected (3x3 Conv+BN+ReLU) to match the already-fused,
  upsampled high-level feature's channel count; a spatial attention map is
  computed from `[mean_channel(x), max_channel(x), mean_channel(y),
  max_channel(y)]` (4 single-channel maps concatenated) through two small
  convs + sigmoid; the blend is `out = x*atten + y*(1-atten)` (the
  reference's learnable `scale` parameter is initialized to 1.0 and has
  `stop_gradient=True`, so it never moves from 1.0 in practice — implemented
  here as the literal constant `1.0`, not a dead trainable parameter); one
  more 3x3 Conv+BN+ReLU projects to the stage's output channel count.
- **3 UAFM stages** at x8/x16/x32 (PaddleSeg's default `backbone_indices`),
  chained: `high_feat = SPPM(x32)`, then reverse-fold `high_feat =
  UAFM(low=x16, high=high_feat)`, then `high_feat = UAFM(low=x8,
  high=high_feat)` — each stage takes the *previous* stage's fused output as
  its high-level input, not the raw backbone feature.
- **SegHead**: 3x3 Conv+BN+ReLU (mid_channels, default 64) -> 1x1 Conv (bias,
  no BN/activation) -> `n_labels` logits, upsampled to the input's spatial
  size (this final upsample is done by the top-level model in the reference
  too, not inside the head itself). Only the finest (x8-derived) head is
  used at inference in the reference (the x16/x32 heads are training-only
  deep-supervision outputs) — this implementation only builds the one
  inference-path head, matching plan.md §3.2's diagram (no aux-head
  requirement stated there, unlike DDRNet §3.3).

Follows the `deeplab_v3_plus`/`lr_aspp` precedent for encoder wiring via
`backbone_zoo` (incl. the `backbone=None` no-backbone fallback) and
`tfmot.quantization.keras.quantize_annotate_layer` wrapping for QAT parity.
'''

from __future__ import absolute_import

from swiss_army_keras.layer_utils import *
from swiss_army_keras._backbone_zoo import backbone_zoo
from swiss_army_keras._model_unet_2d import UNET_left

from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Layer,
    UpSampling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.utils import register_keras_serializable

from tensorflow.nn import relu

import tensorflow as tf
import tensorflow_model_optimization as tfmot


@register_keras_serializable(package='swiss_army_keras')
class ChannelMeanMaxConcat(Layer):
    '''For each tensor in the input list, computes `mean` and `max` over the
    channel axis (keepdims), then concatenates every resulting
    single-channel map together — PP-LiteSeg's UAFM `avg_max_reduce_channel`
    helper (`tensor_fusion_helper.py`). A proper `Layer` subclass rather than
    raw functional `tf.reduce_mean`/`tf.reduce_max`/`tf.concat` calls, for
    the same QAT-model-cloning-safety reason documented on MobileNetV4's
    `MultiQueryAttention` (`_backbone_mobilenetv4.py`, plan.md §2.2): an op
    taking a *list* of tensors, auto-wrapped by Keras as a `TFOpLambda`, does
    not reliably survive `tfmot`'s `quantize_apply` model-cloning step.
    '''

    def call(self, inputs):
        maps = []
        for t in inputs:
            maps.append(tf.reduce_mean(t, axis=-1, keepdims=True))
            maps.append(tf.reduce_max(t, axis=-1, keepdims=True))
        return tf.concat(maps, axis=-1)


@register_keras_serializable(package='swiss_army_keras')
class AttentionBlend(Layer):
    '''`x * atten + y * (1 - atten)` — UAFM's final spatial-attention-weighted
    blend of its two (already channel-matched, already spatially-matched)
    inputs, `atten` a single-channel map broadcasting over both. The
    reference's learnable `scale` parameter multiplying `atten` is
    initialized to 1.0 with `stop_gradient=True` (PaddleSeg's
    `tensor_fusion.py`), so it's a permanent constant in practice, not a
    trainable weight — folded into the `1.0` literal here, not reproduced as
    a dead `Layer` weight. Same Layer-subclass rationale as
    `ChannelMeanMaxConcat` above.
    '''

    def call(self, inputs):
        x, y, atten = inputs
        return x * atten + y * (1.0 - atten)


def _conv_bn_act(x, num_filters, kernel_size=1, use_bias=False, activation=relu, name=''):
    x = qat_annotate_layer(Conv2D(
        num_filters, kernel_size, padding='same', use_bias=use_bias,
        kernel_initializer=HeNormal(), name=f'{name}_conv',
    ))(x)
    x = qat_annotate_layer(BatchNormalization(name=f'{name}_bn'))(x)
    if activation is not None:
        x = qat_annotate_layer(Activation(activation, name=f'{name}_act'))(x)
    return x


def sppm_module(x, out_channels=128, bin_sizes=(1, 2, 4), name='sppm'):
    '''Simplified Pyramid Pooling Module: per-scale adaptive-avg-pool (down
    to a `bin_size x bin_size` grid, implemented as `AveragePooling2D` with
    an exact `H // bin_size` window/stride — requires `H`, `W` evenly
    divisible by every entry of `bin_sizes`, same "assume clean integer
    ratios" convention `deeplab_v3_plus`'s own upsample-size arithmetic
    already relies on) -> shared-shape 1x1 Conv+BN+ReLU -> upsample back ->
    **sum** across scales (not concat) -> one more 3x3 Conv+BN+ReLU.
    '''
    height, width = int(x.shape[1]), int(x.shape[2])
    max_bin = max(bin_sizes)
    if height % max_bin or width % max_bin or height < max_bin or width < max_bin:
        raise ValueError(
            "sppm_module: the deepest (stride-32) feature map is {}x{} but "
            "`sppm_bin_sizes`={} needs both sides to be a positive multiple of "
            "{} - with the default bins this means an input of at least "
            "{}x{} (and a multiple of {}). Use a larger input or smaller "
            "`sppm_bin_sizes`.".format(
                height, width, tuple(bin_sizes), max_bin,
                32 * max_bin, 32 * max_bin, 32 * max_bin))
    branches = []
    for i, bin_size in enumerate(bin_sizes):
        pool_h, pool_w = height // bin_size, width // bin_size
        pooled = qat_annotate_layer(AveragePooling2D(
            pool_size=(pool_h, pool_w), strides=(pool_h, pool_w), name=f'{name}_pool{i}',
        ))(x)
        branch = _conv_bn_act(pooled, out_channels, kernel_size=1, name=f'{name}_branch{i}')
        branch = qat_annotate_layer(UpSampling2D(
            size=(pool_h, pool_w), interpolation='bilinear', name=f'{name}_up{i}',
        ))(branch)
        branches.append(branch)

    # `Add` (like `Multiply`/`Concatenate` elsewhere in this codebase's
    # decoders — see `lr_aspp`'s finding, plan.md §3.1) is left unannotated:
    # a merge layer `tfmot`'s default QAT scheme doesn't support annotating.
    summed = Add(name=f'{name}_sum')(branches)
    return _conv_bn_act(summed, out_channels, kernel_size=3, name=f'{name}_out')


def uafm_block(low_feature, high_feature, out_channels, name='uafm', attention=True):
    '''One UAFM (`UAFM_SpAtten`) stage: fuses a high-resolution "detail"
    skip feature (`low_feature`) with an already-fused, lower-resolution
    "semantic" feature (`high_feature`, upsampled here to match) via a
    spatial attention map computed from both inputs' per-pixel channel
    mean/max statistics.
    '''
    y_channels = int(high_feature.shape[-1])

    x_proj = _conv_bn_act(low_feature, y_channels, kernel_size=3, name=f'{name}_prepare_x')
    y_up = qat_annotate_layer(UpSampling2D(
        size=(low_feature.shape[1] // high_feature.shape[1],
              low_feature.shape[2] // high_feature.shape[2]),
        interpolation='bilinear', name=f'{name}_prepare_y',
    ))(high_feature)

    if not attention:
        # Plain additive fusion: no channel statistics, sigmoid or blending,
        # so only convolutions, a resize and an addition are exported.
        fused = Add(name=f'{name}_add')([x_proj, y_up])
        return _conv_bn_act(fused, out_channels, kernel_size=3, name=f'{name}_conv_out')

    atten_maps = ChannelMeanMaxConcat(name=f'{name}_atten_maps')([x_proj, y_up])
    atten = _conv_bn_act(atten_maps, 2, kernel_size=3, name=f'{name}_atten_conv1')
    atten = _conv_bn_act(atten, 1, kernel_size=3, activation=None, name=f'{name}_atten_conv2')
    atten = qat_annotate_layer(Activation('sigmoid', name=f'{name}_atten_sigmoid'))(atten)

    fused = AttentionBlend(name=f'{name}_blend')([x_proj, y_up, atten])
    return _conv_bn_act(fused, out_channels, kernel_size=3, name=f'{name}_conv_out')


def seg_head(x, n_labels, mid_channels=64, name='seghead'):
    x = _conv_bn_act(x, mid_channels, kernel_size=3, name=f'{name}_conv')
    return qat_annotate_layer(
        Conv2D(n_labels, kernel_size=1, padding='same', use_bias=True, name=f'{name}_classifier'))(x)


def pp_liteseg(input_tensor, n_labels, filter_num_down=[64, 128, 256, 512, 1024],
               feature_levels=(3, 4, 5), sppm_out_channels=128, sppm_bin_sizes=(1, 2, 4),
               arm_out_channels=(64, 96), seg_head_mid_channels=64,
               stack_num_down=2, activation='ReLU', batch_norm=False, pool=True,
               backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True,
               name='pp_liteseg', fusion='uafm'):
    '''PP-LiteSeg with UAFM (Liu et al., 2022) segmentation decoder.

    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Input((None, None, 3))`.
        n_labels: number of segmentation classes.
        filter_num_down: per-level filter counts for the no-backbone encoder
                         fallback (`backbone=None`); ignored otherwise.
        feature_levels: a 3-tuple of 1-indexed encoder feature levels to use
                        as the (x8, x16, x32) inputs — the PP-LiteSeg
                        paper/reference's `backbone_indices`. Default `(3, 4,
                        5)` is tuned for a genuine 5-tap backbone (MobileNetV2,
                        MobileOne, MobileNetV4); a 4-tap custom backbone
                        (RepViT, FastViT) needs `(2, 3, 4)` instead (one less
                        than the 5-tap default, same as those backbones'
                        `deep_layer`/`shallow_layer` overrides elsewhere in
                        this codebase).
        sppm_out_channels: SPPM's per-scale and final output channel count
                           (128 in the reference).
        sppm_bin_sizes: SPPM's adaptive-pool grid sizes (`(1, 2, 4)` in the
                        reference) — every entry must evenly divide the
                        deepest selected feature level's spatial size.
        arm_out_channels: 2-tuple of output channel counts for the x16- and
                          x8-resolution UAFM stages respectively (`(64, 96)`
                          in the reference — note the reference orders this
                          `[x8, x16, x32]`-indexed as `(64, 96, 128)`; this
                          implementation only exposes the 2 values actually
                          used by the 2 UAFM stages, `sppm_out_channels`
                          already covers the 3rd/x32 entry).
        seg_head_mid_channels: SegHead's intermediate channel count.
        stack_num_down, activation, batch_norm, pool: passed through to the
            no-backbone (`backbone=None`) encoder fallback, same as
            `deeplab_v3_plus`/`lr_aspp`.
        backbone: `None` (default; plain stacked-conv encoder), or the name
                  of a backbone registered in
                  `swiss_army_keras._backbone_zoo.BACKBONE_REGISTRY`.
        weights: one of `None`, `'imagenet'`, or a path to a weights file.
        freeze_backbone: `True` to freeze the backbone's weights.
        freeze_batch_norm: `True` to also freeze the backbone's BN layers.
        name: prefix of the created Keras model and its layers.
        fusion: `'uafm'` (default) for the paper's attention fusion (channel
                mean/max statistics, sigmoid, blending), or `'add'` for a
                plain additive fusion of the two features. `'add'` exports
                only convolutions, resizes and additions in the fusion
                stages, for accelerators that do not run the attention
                operators (see the deployment guide). It is a different
                architecture and has to be trained as such.

    Output
    ----------
        A `tf.keras.Model` with a `.preprocessing` attribute, producing an
        `(H, W, n_labels)` logit map at the input's spatial resolution.
    '''
    if fusion not in ('uafm', 'add'):
        raise ValueError("`fusion` must be 'uafm' or 'add', got {!r}".format(fusion))

    depth_ = len(filter_num_down)
    X_encoder = []

    if backbone is None:
        X = input_tensor
        X = CONV_stack(X, filter_num_down[0], kernel_size=3, stack_num=stack_num_down,
                       activation=activation, batch_norm=batch_norm, name='{}_down0'.format(name))
        X_encoder.append(X)

        for i, f in enumerate(filter_num_down[1:]):
            X = UNET_left(X, f, kernel_size=3, stack_num=stack_num_down, activation=activation,
                          pool=pool, batch_norm=batch_norm, name='{}_down{}'.format(name, i + 1))
            X_encoder.append(X)

        preprocessing = dummy_preprocessing
    else:
        backbone_ = backbone_zoo(
            backbone, weights, input_tensor, max(feature_levels), freeze_backbone, freeze_batch_norm)
        X_encoder = backbone_([input_tensor, ])
        preprocessing = backbone_.preprocessing

    x8, x16, x32 = (X_encoder[i - 1] for i in feature_levels)

    high_feat = sppm_module(x32, out_channels=sppm_out_channels, bin_sizes=sppm_bin_sizes,
                            name='{}_sppm'.format(name))
    high_feat = uafm_block(x16, high_feat, out_channels=arm_out_channels[1], name='{}_uafm16'.format(name),
                           attention=(fusion == 'uafm'))
    high_feat = uafm_block(x8, high_feat, out_channels=arm_out_channels[0], name='{}_uafm8'.format(name),
                           attention=(fusion == 'uafm'))

    x = seg_head(high_feat, n_labels, mid_channels=seg_head_mid_channels, name='{}_seghead'.format(name))

    x = qat_annotate_layer(UpSampling2D(
        size=(input_tensor.shape[1] // x.shape[1],
              input_tensor.shape[2] // x.shape[2]),
        interpolation='bilinear', name='{}_upsample_to_input'.format(name),
    ))(x)

    m = Model([input_tensor, ], [x, ])

    with tfmot.quantization.keras.quantize_scope():
        m = tfmot.quantization.keras.quantize_apply(m)

    m.preprocessing = preprocessing

    return m
