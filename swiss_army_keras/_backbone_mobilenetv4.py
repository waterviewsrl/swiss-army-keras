'''MobileNetV4 (Google, 2024) — "MobileNetV4 - Universal Models for the
Mobile Ecosystem" (https://arxiv.org/abs/2404.10518).

Like MobileOne (§2.1), no Keras port exists anywhere (confirmed, plan.md
§12) and no official TF/Keras *checkpoint* exists either (Google only
shipped TFLite exports) — architecture verified directly against `timm`'s
port (`timm/models/mobilenetv3.py`'s `_gen_mobilenet_v4`, built on the
shared `EfficientNetBuilder`/block-string infrastructure, plus
`_efficientnet_blocks.py`'s `UniversalInvertedResidual`/`MobileAttention`),
per plan.md §7.

Core block: the Universal Inverted Bottleneck (UIB) —
`[optional DW(a), no activation] -> PW expand (BN+act) -> [optional DW(k),
BN+act] -> PW project (BN, no activation)`, plus optional `LayerScale`
(Hybrid variants only) and a residual when stride=1 and in==out channels.
Which of the two optional depthwise convs are present (a>0? k>0?) gives 4
modes purely by construction, no separate code path needed: `a=0,k>0` =
plain Inverted Bottleneck; `a>0,k=0` = ConvNeXt-like; `a>0,k>0` = "ExtraDW";
`a=0,k=0` = a plain FFN (no spatial mixing at all in this block). Which of
the two depthwise convs carries the block's stride is non-obvious when both
are present — see `uib_block`'s docstring. Stage 0 additionally uses two
other block kinds: a plain conv (`cn`) and a fused "edge residual" (`er`, a
single KxK conv doing spatial+expansion together, then a project conv — no
depthwise at all). No structural reparameterization anywhere in this
backbone (confirmed in the research) — nothing here needs `reparam_utils.py`,
unlike MobileOne/FastViT/RepViT.

Hybrid-Medium/Hybrid-Large add Mobile-MQA (multi-query attention) blocks in
stages 2-3: a single shared K/V projection (true multi-query, not
multi-head, for K/V) with Q having its own per-head projection, and K/V
each optionally spatially downsampled first via a strided depthwise conv+BN
(`kv_stride`, never applied to Q) — see `mobile_mqa_block`'s docstring for a
subtlety in how K and V's downsample convs relate. MQA blocks are
interleaved with UIB blocks within a stage, always at stride=1, with a
learnable per-channel `LayerScale` (shared with FastViT §2.3, imported
rather than redefined to avoid a second `register_keras_serializable`
registration) before their residual add. Hybrid-M/L's UIB blocks get this
same `LayerScale` too (`layer_scale_init=1e-5` for these two variants only,
`None`/no-op for Conv-S/M/L, matching the reference's
`layer_scale_init_value` being set only when `'hybrid' in variant`).

Native feature pyramid: unlike RepViT/FastViT's 4-tap patchify-stem
limitation, MobileNetV4's stem is a single stride-2 conv (2x tap) — a
genuine 5-tap 2x/4x/8x/16x/32x pyramid exists, same as MobileOne (§2.1).
Every stage's first block carries that stage's stride-2 downsampling
transition; every other block in the stage is stride=1 (confirmed from the
per-variant resolution progression in the source configs, not assumed).

Weights: real ImageNet-pretrained weights for all 5 variants are ported
from timm's checkpoints via `scripts/port_mobilenetv4_weights.py` (plan.md
§12.5) - pass the resulting `.weights.h5` file path as `weights=`.
`weights='imagenet'` (auto-download) is not implemented yet; `weights=None`
(random init) is fully supported.

The per-stage block sequences in `MOBILENETV4_CONFIGS` below are parsed
programmatically from `timm`'s own arch-def strings, not hand-transcribed —
see that dict's own comment for why that distinction matters.
'''

from __future__ import absolute_import

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Input,
    Layer,
    Reshape,
    ZeroPadding2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable

from swiss_army_keras.activations import GELU
from swiss_army_keras._backbone_fastvit import LayerScale


@register_keras_serializable(package='swiss_army_keras')
class MultiQueryAttention(Layer):
    '''`softmax(Q K^T / sqrt(d)) V` with Q holding its own per-head
    projection (B, Nq, heads, d) but K/V *shared* across all heads (B,
    Nkv, d), i.e. genuine multi-*query* attention rather than
    multi-head - the core of MobileNetV4's Mobile-MQA (Hybrid-M/L only).

    Implemented as a proper `Layer` subclass, not raw functional
    `tf.einsum` calls: a multi-tensor-input `tf.einsum`, auto-wrapped by
    Keras as a `TFOpLambda`, does not survive `tfmot`'s QAT model-cloning
    (`quantize_apply`) round-trip - confirmed here via
    `deeplab_v3_plus`+Hybrid-L integration testing (`ValueError: Got 0
    inputs for equation ... expecting 2` when `quantize_apply` tries to
    reconstruct the op from its serialized config). Exactly the same class
    of issue already solved for FastViT's attention block by using a real
    layer (`tf.keras.layers.MultiHeadAttention`) instead of raw ops - see
    `_backbone_fastvit.py`.
    '''

    def call(self, inputs):
        q, k, v = inputs
        scale = 1.0 / (float(k.shape[-1]) ** 0.5)
        attn_logits = tf.einsum('bqhd,bkd->bhqk', q, k) * scale
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        return tf.einsum('bhqk,bkd->bqhd', attn_weights, v)


def _pad_same(x, kernel_size, name):
    '''Explicit, symmetric zero-padding (`ZeroPadding2D(k // 2)` +
    `padding='valid'`), used instead of Keras's `padding='same'` on every
    conv in this file that can run at `stride > 1`: TF/Keras's `'same'` can
    pad asymmetrically at stride>1 (any odd remainder goes on the "after"
    side), while `timm`'s `create_conv2d` (this family's reference) always
    uses a fixed, symmetric `k // 2` pad - the same padding-alignment class
    of bug already relevant to MobileOne (§2.1) and FastViT (§2.3).
    '''
    pad = kernel_size // 2
    return ZeroPadding2D(pad, name=f'{name}_pad')(x) if pad > 0 else x


def _conv_bn_act(x, out_channels, kernel_size, stride, activation, use_act=True, name=''):
    x = _pad_same(x, kernel_size, name=f'{name}_conv')
    x = Conv2D(out_channels, kernel_size, strides=stride, padding='valid', use_bias=False, name=f'{name}_conv')(x)
    x = BatchNormalization(epsilon=1e-5, name=f'{name}_bn')(x)
    if use_act:
        x = activation(name=f'{name}_act')(x) if callable(activation) and not isinstance(activation, str) else Activation(activation, name=f'{name}_act')(x)
    return x


def _dw_bn_act(x, kernel_size, stride, activation, use_act, name):
    x = _pad_same(x, kernel_size, name=f'{name}_dw')
    x = DepthwiseConv2D(kernel_size, strides=stride, padding='valid', use_bias=False, name=f'{name}_dw')(x)
    x = BatchNormalization(epsilon=1e-5, name=f'{name}_dw_bn')(x)
    if use_act:
        x = _act(x, activation, name=f'{name}_dw_act')
    return x


def _act(x, activation, name):
    return GELU(name=name)(x) if activation == 'gelu' else Activation('relu', name=name)(x)


def _make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    '''Exact port of `timm.layers.helpers.make_divisible` - the formula the
    reference uses to size a UIB/edge-residual block's hidden (expanded)
    channel count (`make_divisible(in_chs * exp_ratio)`), not a plain
    `round()`. A no-op for every released variant's actual channel
    counts/expand ratios (all already land on multiples of 8), but ported
    exactly rather than assumed safe.
    '''
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def uib_block(x, out_channels, start_k, mid_k, stride, expand_ratio, activation='relu', layer_scale_init=None, name=''):
    '''Universal Inverted Bottleneck: `[optional DW(start_k), no act] -> PW
    expand (BN+act) -> [optional DW(mid_k), BN+act] -> PW project (BN, no
    act)`, + optional `LayerScale` (Hybrid variants only), + residual iff
    stride=1 and in_channels==out_channels. `start_k=0` or `mid_k=0` skips
    that depthwise conv entirely (an `Identity`, in the reference) - this
    alone is what gives plain-IB (start_k=0), ConvNeXt-like (mid_k=0),
    ExtraDW (both >0), or FFN (both 0).

    Which depthwise conv carries the block's stride is not "whichever one is
    `start`": the reference always gives `dw_mid` the full stride whenever
    `mid_k > 0`; `dw_start` only carries it when `dw_mid` doesn't exist at
    all (`mid_k == 0`). Easy to get backwards since ExtraDW blocks
    (`start_k>0` and `mid_k>0` together) are both the most common block kind
    and exactly the ones at every stage's stride-2 transition (plan.md §12.5).
    '''
    in_channels = x.shape[-1]
    use_residual = (stride == 1 and in_channels == out_channels)
    shortcut = x

    h = x
    if start_k > 0:
        # Only carries the stride when there's no dw_mid to carry it instead.
        start_stride = stride if mid_k == 0 else 1
        h = _dw_bn_act(h, start_k, start_stride, activation, use_act=False, name=f'{name}_dwstart')

    hidden_channels = _make_divisible(in_channels * expand_ratio)
    h = _conv_bn_act(h, hidden_channels, 1, 1, activation, name=f'{name}_pwexp')

    if mid_k > 0:
        # Always carries the block's full stride, regardless of dw_start.
        h = _dw_bn_act(h, mid_k, stride, activation, use_act=True, name=f'{name}_dwmid')

    h = Conv2D(out_channels, 1, padding='same', use_bias=False, name=f'{name}_pwproj_conv')(h)
    h = BatchNormalization(epsilon=1e-5, name=f'{name}_pwproj_bn')(h)

    if layer_scale_init is not None:
        h = LayerScale(layer_scale_init, name=f'{name}_scale')(h)

    if use_residual:
        h = Add(name=f'{name}_residual')([h, shortcut])
    return h


def edge_residual_block(x, out_channels, kernel_size, stride, expand_ratio, activation='relu', name=''):
    '''"er" block (fused-IB): a single KxK conv does spatial mixing AND
    channel expansion together (no separate depthwise), then a pointwise
    conv projects down. Used only for stage 0 in Conv-M/L/Hybrid-M/L.
    '''
    in_channels = x.shape[-1]
    use_residual = (stride == 1 and in_channels == out_channels)
    hidden_channels = _make_divisible(in_channels * expand_ratio)

    h = _conv_bn_act(x, hidden_channels, kernel_size, stride, activation, name=f'{name}_expand')
    h = Conv2D(out_channels, 1, padding='same', use_bias=False, name=f'{name}_proj_conv')(h)
    h = BatchNormalization(epsilon=1e-5, name=f'{name}_proj_bn')(h)

    if use_residual:
        h = Add(name=f'{name}_residual')([h, x])
    return h


def mobile_mqa_block(x, num_heads, key_dim, kv_stride, layer_scale_init=1e-5, activation='relu', name=''):
    '''Mobile-MQA: multi-*query* attention - Q gets its own per-head
    projection, but K and V share a *single* projection between all heads
    (the "multi-query" part). K and V each get their *own, independent*
    strided depthwise conv+BN downsample when `kv_stride > 1` - not one
    downsample conv shared between them, easy to assume by analogy to K/V
    otherwise being treated as a pair (plan.md §12.5). Q is never
    downsampled in any released MobileNetV4 config. BN (no activation)
    precedes the whole block; a learnable per-channel `LayerScale` (shared
    with FastViT §2.3) precedes the residual add - always present, stride=1,
    in==out channels, so the residual always applies here. Q/K/V/output
    projections are all bias-free, including the output proj (the reference
    defaults `use_bias=False` throughout `MultiQueryAttention2d`).
    '''
    channels = x.shape[-1]
    height, width = x.shape[1], x.shape[2]

    normed = BatchNormalization(epsilon=1e-5, name=f'{name}_norm')(x)

    q = Conv2D(num_heads * key_dim, 1, padding='same', use_bias=False, name=f'{name}_q_proj')(normed)
    q = Reshape((height * width, num_heads, key_dim), name=f'{name}_q_reshape')(q)

    def _kv_branch(prefix):
        kv_input = normed
        kv_h, kv_w = height, width
        if kv_stride > 1:
            kv_input = _pad_same(kv_input, 3, name=f'{name}_{prefix}_dw')
            kv_input = DepthwiseConv2D(3, strides=kv_stride, padding='valid', use_bias=False, name=f'{name}_{prefix}_dw')(kv_input)
            kv_input = BatchNormalization(epsilon=1e-5, name=f'{name}_{prefix}_dw_bn')(kv_input)
            kv_h, kv_w = kv_input.shape[1], kv_input.shape[2]
        proj = Conv2D(key_dim, 1, padding='same', use_bias=False, name=f'{name}_{prefix}_proj')(kv_input)
        return Reshape((kv_h * kv_w, key_dim), name=f'{name}_{prefix}_reshape')(proj)

    k = _kv_branch('k')
    v = _kv_branch('v')

    # q: (B, HW, heads, key_dim), k/v: (B, HW_kv, key_dim) shared across heads.
    attn_out = MultiQueryAttention(name=f'{name}_attn')([q, k, v])
    attn_out = Reshape((height, width, num_heads * key_dim), name=f'{name}_out_reshape')(attn_out)
    attn_out = Conv2D(channels, 1, padding='same', use_bias=False, name=f'{name}_out_proj')(attn_out)

    attn_out = LayerScale(layer_scale_init, name=f'{name}_scale')(attn_out)
    return Add(name=f'{name}_residual')([attn_out, x])


def _stage_output_name(model_name, stage_idx):
    return f'{model_name}_stage{stage_idx}_out'


def _build_stage(x, blocks, stage_idx, activation, layer_scale_init, model_name):
    spatial_seen = False
    for block_idx, spec in enumerate(blocks):
        kind = spec[0]
        name = f'{model_name}_stage{stage_idx}_block{block_idx}'

        if kind == 'mqa':
            _, num_heads, key_dim, kv_stride = spec
            x = mobile_mqa_block(x, num_heads, key_dim, kv_stride, layer_scale_init=layer_scale_init, activation=activation, name=name)
            continue

        stride = 2 if not spatial_seen else 1
        spatial_seen = True

        if kind == 'cn':
            _, out_channels, kernel_size = spec
            x = _conv_bn_act(x, out_channels, kernel_size, stride, activation, name=name)
        elif kind == 'er':
            _, out_channels, kernel_size, expand = spec
            x = edge_residual_block(x, out_channels, kernel_size, stride, expand, activation=activation, name=name)
        elif kind == 'uir':
            _, out_channels, start_k, mid_k, expand = spec
            x = uib_block(x, out_channels, start_k, mid_k, stride, expand, activation=activation, layer_scale_init=layer_scale_init, name=name)
        else:
            raise ValueError(f'Unknown block kind: {kind}')

    x = Activation('linear', name=_stage_output_name(model_name, stage_idx))(x)
    return x


def _mobilenetv4_network(input_tensor, stem_channels, stages, head_channels, activation, layer_scale_init=None,
                         include_top=False, pooling=None, num_classes=1000, name='MobileNetV4'):
    x = _conv_bn_act(input_tensor, stem_channels, 3, 2, activation, name=f'{name}_stem')
    x = Activation('linear', name=f'{name}_stem_out')(x)

    for stage_idx, blocks in enumerate(stages):
        x = _build_stage(x, blocks, stage_idx, activation, layer_scale_init, name)

    x = _conv_bn_act(x, head_channels, 1, 1, activation, name=f'{name}_head')

    if include_top:
        x = GlobalAveragePooling2D(name=f'{name}_head_gap')(x)
        x = Dropout(0.0, name=f'{name}_head_dropout')(x)
        x = Dense(num_classes, activation='softmax', name=f'{name}_head_dense')(x)
        return x

    if pooling == 'avg':
        return GlobalAveragePooling2D(name=f'{name}_pool')(x)
    if pooling == 'max':
        return GlobalMaxPooling2D(name=f'{name}_pool')(x)

    return x


def mobilenetv4_stage_output_names(model_name, num_stages=4):
    '''The 5 pyramid-tap layer names: stem, then each of the 4 main stages'
    output (stage 4/the head is a stride-1 1x1 conv, not a pyramid level).
    '''
    names = [f'{model_name}_stem_out']
    names += [_stage_output_name(model_name, i) for i in range(num_stages)]
    return tuple(names)


def _build_mobilenetv4(stem_channels, stages, head_channels, activation, model_name, layer_scale_init=None):
    def constructor(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs):
        if weights == 'imagenet':
            raise NotImplementedError(
                "weights='imagenet' auto-download isn't available yet for {} - that needs "
                "real, publicly-hosted weight files this package can fetch, which is a "
                "separate, not-yet-done piece of work (plan.md §12). Real, verified "
                "ImageNet-pretrained weights ARE available today, though: run "
                "`scripts/port_mobilenetv4_weights.py` (converts timm's pretrained checkpoint - "
                "verified against the PyTorch reference, plan.md §12.5) and pass the resulting "
                "`<model_name>.weights.h5` file path directly as `weights=`, e.g. "
                "weights='pretrained_weights/mobilenetv4/{}.weights.h5'.".format(model_name, model_name)
            )

        if input_tensor is None:
            if input_shape is None:
                raise ValueError('Either `input_tensor` or `input_shape` must be provided.')
            input_tensor = Input(shape=input_shape)

        output = _mobilenetv4_network(
            input_tensor, stem_channels, stages, head_channels, activation, layer_scale_init=layer_scale_init,
            include_top=include_top, pooling=pooling, num_classes=classes, name=model_name,
        )
        model = Model(inputs=input_tensor, outputs=output, name=model_name)

        if weights not in (None, 'none'):
            # Any other string is a local `.weights.h5` file path (e.g. from
            # `scripts/port_mobilenetv4_weights.py`) - built with
            # `include_top=False`, so `skip_mismatch=True` lets it load
            # cleanly into an `include_top=True` model too.
            model.load_weights(weights, skip_mismatch=True)
        return model

    constructor.__name__ = model_name
    return constructor


# Per-variant stage configs, programmatically parsed from timm's actual
# `_gen_mobilenet_v4` arch-def block strings (extracted directly from a
# running `timm` import, not hand-transcribed - see plan.md §12.5) - block
# kind tags:
#   ('cn', out_channels, kernel_size)                      - plain conv
#   ('er', out_channels, kernel_size, expand_ratio)         - fused edge residual
#   ('uir', out_channels, start_k, mid_k, expand_ratio)     - universal inverted bottleneck
#   ('mqa', num_heads, key_dim, kv_stride)                  - mobile multi-query attention
# Every stage's first spatial (non-mqa) block carries that stage's stride-2
# transition; every other block is stride=1 - confirmed from the source's
# per-variant resolution progression (112->56->28->14->7), not assumed.
#
# The original hand-transcription (plan.md §12 research phase) had real
# errors in 4 of these 5 variants - found while porting real ImageNet
# weights (plan.md §12.5) and confirmed by re-deriving every block
# programmatically from `timm`'s own source rather than trusting the
# by-eye transcription a second time: wrong expand ratios on isolated
# blocks (Conv-S, Conv-M, Hybrid-M), a `('cn', out, 3)` instead of
# `('cn', out, 1)` for two of Conv-S's plain-conv blocks (the arch string's
# own `k1` misread as the stage's other block's `k3`), and - the most
# significant - Conv-L's stage 2 silently missing 3 of its 5
# `uir_r5_a5_k3_s1_e4_c192` repeats (`r5` misread as `r2`), which
# understates the whole variant's depth, not just one block's parameters.
MOBILENETV4_CONFIGS = {
    'MobileNetV4-Conv-S': dict(
        stem_channels=32, head_channels=960, activation='relu', model_name='MobileNetV4_ConvS',
        stages=[
            [('cn', 32, 3), ('cn', 32, 1)],
            [('cn', 96, 3), ('cn', 64, 1)],
            [('uir', 96, 5, 5, 3), ('uir', 96, 0, 3, 2), ('uir', 96, 0, 3, 2), ('uir', 96, 0, 3, 2),
             ('uir', 96, 0, 3, 2), ('uir', 96, 3, 0, 4)],
            [('uir', 128, 3, 3, 6), ('uir', 128, 5, 5, 4), ('uir', 128, 0, 5, 4), ('uir', 128, 0, 5, 3),
             ('uir', 128, 0, 3, 4), ('uir', 128, 0, 3, 4)],
        ],
    ),
    'MobileNetV4-Conv-M': dict(
        stem_channels=32, head_channels=960, activation='relu', model_name='MobileNetV4_ConvM',
        stages=[
            [('er', 48, 3, 4)],
            [('uir', 80, 3, 5, 4), ('uir', 80, 3, 3, 2)],
            [('uir', 160, 3, 5, 6), ('uir', 160, 3, 3, 4), ('uir', 160, 3, 3, 4), ('uir', 160, 3, 5, 4),
             ('uir', 160, 3, 3, 4), ('uir', 160, 3, 0, 4), ('uir', 160, 0, 0, 2), ('uir', 160, 3, 0, 4)],
            [('uir', 256, 5, 5, 6), ('uir', 256, 5, 5, 4), ('uir', 256, 3, 5, 4), ('uir', 256, 3, 5, 4),
             ('uir', 256, 0, 0, 4), ('uir', 256, 3, 0, 4), ('uir', 256, 3, 5, 2), ('uir', 256, 5, 5, 4),
             ('uir', 256, 0, 0, 4), ('uir', 256, 0, 0, 4), ('uir', 256, 5, 0, 2)],
        ],
    ),
    'MobileNetV4-Conv-L': dict(
        stem_channels=24, head_channels=960, activation='relu', model_name='MobileNetV4_ConvL',
        stages=[
            [('er', 48, 3, 4)],
            [('uir', 96, 3, 5, 4), ('uir', 96, 3, 3, 4)],
            [('uir', 192, 3, 5, 4), ('uir', 192, 3, 3, 4), ('uir', 192, 3, 3, 4), ('uir', 192, 3, 3, 4),
             ('uir', 192, 3, 5, 4), ('uir', 192, 5, 3, 4), ('uir', 192, 5, 3, 4), ('uir', 192, 5, 3, 4),
             ('uir', 192, 5, 3, 4), ('uir', 192, 5, 3, 4), ('uir', 192, 3, 0, 4)],
            [('uir', 512, 5, 5, 4), ('uir', 512, 5, 5, 4), ('uir', 512, 5, 5, 4), ('uir', 512, 5, 5, 4),
             ('uir', 512, 5, 0, 4), ('uir', 512, 5, 3, 4), ('uir', 512, 5, 0, 4), ('uir', 512, 5, 0, 4),
             ('uir', 512, 5, 3, 4), ('uir', 512, 5, 5, 4), ('uir', 512, 5, 0, 4), ('uir', 512, 5, 0, 4),
             ('uir', 512, 5, 0, 4)],
        ],
    ),
    'MobileNetV4-Hybrid-M': dict(
        stem_channels=32, head_channels=960, activation='relu', model_name='MobileNetV4_HybridM',
        layer_scale_init=1e-5,
        stages=[
            [('er', 48, 3, 4)],
            [('uir', 80, 3, 5, 4), ('uir', 80, 3, 3, 2)],
            [('uir', 160, 3, 5, 6), ('uir', 160, 0, 0, 2), ('uir', 160, 3, 3, 4), ('uir', 160, 3, 5, 4),
             ('mqa', 4, 64, 2), ('uir', 160, 3, 3, 4), ('mqa', 4, 64, 2), ('uir', 160, 3, 0, 4),
             ('mqa', 4, 64, 2), ('uir', 160, 3, 3, 4), ('mqa', 4, 64, 2), ('uir', 160, 3, 0, 4)],
            [('uir', 256, 5, 5, 6), ('uir', 256, 5, 5, 4), ('uir', 256, 3, 5, 4), ('uir', 256, 3, 5, 4),
             ('uir', 256, 0, 0, 2), ('uir', 256, 3, 5, 2), ('uir', 256, 0, 0, 2), ('uir', 256, 0, 0, 4),
             ('mqa', 4, 64, 1), ('uir', 256, 3, 0, 4), ('mqa', 4, 64, 1), ('uir', 256, 5, 5, 4),
             ('mqa', 4, 64, 1), ('uir', 256, 5, 0, 4), ('mqa', 4, 64, 1), ('uir', 256, 5, 0, 4)],
        ],
    ),
    'MobileNetV4-Hybrid-L': dict(
        stem_channels=24, head_channels=960, activation='gelu', model_name='MobileNetV4_HybridL',
        layer_scale_init=1e-5,
        stages=[
            [('er', 48, 3, 4)],
            [('uir', 96, 3, 5, 4), ('uir', 96, 3, 3, 4)],
            [('uir', 192, 3, 5, 4), ('uir', 192, 3, 3, 4), ('uir', 192, 3, 3, 4), ('uir', 192, 3, 3, 4),
             ('uir', 192, 3, 5, 4), ('uir', 192, 5, 3, 4), ('uir', 192, 5, 3, 4), ('mqa', 8, 48, 2),
             ('uir', 192, 5, 3, 4), ('mqa', 8, 48, 2), ('uir', 192, 5, 3, 4), ('mqa', 8, 48, 2),
             ('uir', 192, 5, 3, 4), ('mqa', 8, 48, 2), ('uir', 192, 3, 0, 4)],
            [('uir', 512, 5, 5, 4), ('uir', 512, 5, 5, 4), ('uir', 512, 5, 5, 4), ('uir', 512, 5, 5, 4),
             ('uir', 512, 5, 0, 4), ('uir', 512, 5, 3, 4), ('uir', 512, 5, 0, 4), ('uir', 512, 5, 0, 4),
             ('uir', 512, 5, 3, 4), ('uir', 512, 5, 5, 4), ('mqa', 8, 64, 1), ('uir', 512, 5, 0, 4),
             ('mqa', 8, 64, 1), ('uir', 512, 5, 0, 4), ('mqa', 8, 64, 1), ('uir', 512, 5, 0, 4),
             ('mqa', 8, 64, 1), ('uir', 512, 5, 0, 4)],
        ],
    ),
}

MobileNetV4_ConvS = _build_mobilenetv4(**{k: v for k, v in MOBILENETV4_CONFIGS['MobileNetV4-Conv-S'].items()})
MobileNetV4_ConvM = _build_mobilenetv4(**{k: v for k, v in MOBILENETV4_CONFIGS['MobileNetV4-Conv-M'].items()})
MobileNetV4_ConvL = _build_mobilenetv4(**{k: v for k, v in MOBILENETV4_CONFIGS['MobileNetV4-Conv-L'].items()})
MobileNetV4_HybridM = _build_mobilenetv4(**{k: v for k, v in MOBILENETV4_CONFIGS['MobileNetV4-Hybrid-M'].items()})
MobileNetV4_HybridL = _build_mobilenetv4(**{k: v for k, v in MOBILENETV4_CONFIGS['MobileNetV4-Hybrid-L'].items()})


# Plain tuples, not `tf.constant(...)` - see `_backbone_repvit.py`'s
# identical constants for why a module-level eager tensor is a real bug
# (forces TF's eager context to initialize at import time, breaking a
# later `tf.config.experimental.set_memory_growth` call in user code).
_MOBILENETV4_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_MOBILENETV4_IMAGENET_STD = (0.229, 0.224, 0.225)


def mobilenetv4_preprocess_input(x):
    '''Real ImageNet normalization (`[0, 1]` scaling, then per-channel
    mean/std) that `scripts/port_mobilenetv4_weights.py`'s ported weights
    expect - confirmed via `timm`'s own `_cfg()` defaults
    (`IMAGENET_DEFAULT_MEAN`/`STD`) for this family (plan.md §12.5),
    identical to every other backbone in this plan
    (`repvit_preprocess_input`/`fastvit_preprocess_input`/
    `mobileone_preprocess_input`).
    '''
    x = x / 255.0
    return (x - _MOBILENETV4_IMAGENET_MEAN) / _MOBILENETV4_IMAGENET_STD
