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
BN+act] -> PW project (BN, no activation)`, plus a residual when
stride=1 and in==out channels. Which of the two optional depthwise convs
are present (a>0? k>0?) gives 4 modes purely by construction, no separate
code path needed: `a=0,k>0` = plain Inverted Bottleneck; `a>0,k=0` =
ConvNeXt-like; `a>0,k>0` = "ExtraDW"; `a=0,k=0` = a plain FFN (no spatial
mixing at all in this block). Stage 0 additionally uses two other block
kinds: a plain conv (`cn`) and a fused "edge residual" (`er`, a single KxK
conv doing spatial+expansion together, then a project conv — no depthwise
at all). **No structural reparameterization anywhere in this backbone**
(confirmed in the research) — nothing here needs `reparam_utils.py`, unlike
MobileOne/FastViT/RepViT.

Hybrid-Medium/Hybrid-Large add Mobile-MQA (multi-query attention) blocks in
stages 2-3: a single shared K/V projection (true multi-query, not
multi-head, for K/V) with Q having its own per-head projection, and K/V
optionally spatially downsampled first via a strided depthwise conv+BN
(`kv_stride`) before their 1x1 projection — Q is never downsampled in any
released config. MQA blocks are interleaved with UIB blocks within a stage,
always at stride=1 (attention never downsamples spatially here), with a
learnable per-channel "layer scale" before their residual add, same
`LayerScale` layer used for FastViT (defined there; imported here rather
than redefined to avoid a second registration of a layer with the same
serialized name under `register_keras_serializable`).

Native feature pyramid: **unlike RepViT/FastViT's 4-tap patchify-stem
limitation, MobileNetV4's stem is a single stride-2 conv (2x tap)** — a
genuine 5-tap 2x/4x/8x/16x/32x pyramid exists, same as MobileOne (§2.1).
Every stage's first block carries that stage's stride-2 downsampling
transition; every other block in the stage is stride=1 (confirmed from the
per-variant resolution progression in the source configs, not assumed).

Weights: no `weights='imagenet'` support in this pass — same status as
every other backbone in this plan (plan.md §12/§12.1): `timm` has PyTorch
weights for all 5 variants, but no Keras checkpoint exists to port from
directly, and porting isn't done here. `weights=None` (random init) is
fully supported.

Per-block expand-ratio note: the per-stage block sequences below (channel
counts, stride locations, block-mode diversity — start/mid kernel sizes)
are transcribed directly from `timm`'s arch-def strings. A handful of
individual blocks' *expand ratios* specifically were ambiguous in the
compressed notation this was researched through and default to `4` (the
most common value in the source) where not unambiguously given — this has
zero effect on the feature-pyramid shape (channel counts/strides, which
`BACKBONE_REGISTRY` depends on, are exact) and no weight-matching
consequence yet, since `weights=None` is all that's supported so far;
revisit if/when real weight porting is attempted (plan.md §12.1).
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


def _conv_bn_act(x, out_channels, kernel_size, stride, activation, use_act=True, name=''):
    x = Conv2D(out_channels, kernel_size, strides=stride, padding='same', use_bias=False, name=f'{name}_conv')(x)
    x = BatchNormalization(epsilon=1e-3, name=f'{name}_bn')(x)
    if use_act:
        x = activation(name=f'{name}_act')(x) if callable(activation) and not isinstance(activation, str) else Activation(activation, name=f'{name}_act')(x)
    return x


def _dw_bn_act(x, kernel_size, stride, activation, use_act, name):
    x = DepthwiseConv2D(kernel_size, strides=stride, padding='same', use_bias=False, name=f'{name}_dw')(x)
    x = BatchNormalization(epsilon=1e-3, name=f'{name}_dw_bn')(x)
    if use_act:
        x = _act(x, activation, name=f'{name}_dw_act')
    return x


def _act(x, activation, name):
    return GELU(name=name)(x) if activation == 'gelu' else Activation('relu', name=name)(x)


def uib_block(x, out_channels, start_k, mid_k, stride, expand_ratio, activation='relu', name=''):
    '''Universal Inverted Bottleneck: `[optional DW(start_k), no act] -> PW
    expand (BN+act) -> [optional DW(mid_k), BN+act] -> PW project (BN, no
    act)`, + residual iff stride=1 and in_channels==out_channels.
    `start_k=0` or `mid_k=0` skips that depthwise conv entirely (an
    `Identity`, in the reference) - this alone is what gives plain-IB
    (start_k=0), ConvNeXt-like (mid_k=0), ExtraDW (both >0), or FFN (both 0).
    '''
    in_channels = x.shape[-1]
    use_residual = (stride == 1 and in_channels == out_channels)
    shortcut = x

    h = x
    if start_k > 0:
        h = _dw_bn_act(h, start_k, stride, activation, use_act=False, name=f'{name}_dwstart')
    elif stride > 1:
        # No start-DW to carry the stride (plain IB/FFN case) - the
        # reference still needs *some* op to downsample when this stage's
        # first block has stride=2; the expand/mid depthwise below carries
        # it instead in that case (mid_k is guaranteed >0 whenever start_k
        # is 0 and stride>1 in every released config - plain FFN blocks
        # never sit at a stage's stride-2 position).
        pass

    hidden_channels = int(round(in_channels * expand_ratio))
    h = _conv_bn_act(h, hidden_channels, 1, 1, activation, name=f'{name}_pwexp')

    if mid_k > 0:
        mid_stride = stride if start_k == 0 else 1
        h = _dw_bn_act(h, mid_k, mid_stride, activation, use_act=True, name=f'{name}_dwmid')

    h = Conv2D(out_channels, 1, padding='same', use_bias=False, name=f'{name}_pwproj_conv')(h)
    h = BatchNormalization(epsilon=1e-3, name=f'{name}_pwproj_bn')(h)

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
    hidden_channels = int(round(in_channels * expand_ratio))

    h = _conv_bn_act(x, hidden_channels, kernel_size, stride, activation, name=f'{name}_expand')
    h = Conv2D(out_channels, 1, padding='same', use_bias=False, name=f'{name}_proj_conv')(h)
    h = BatchNormalization(epsilon=1e-3, name=f'{name}_proj_bn')(h)

    if use_residual:
        h = Add(name=f'{name}_residual')([h, x])
    return h


def mobile_mqa_block(x, num_heads, key_dim, kv_stride, layer_scale_init=1e-5, activation='relu', name=''):
    '''Mobile-MQA: multi-*query* attention - Q gets its own per-head
    projection, but K and V share a *single* projection between all heads
    (the "multi-query" part). K/V are optionally spatially downsampled
    first via a strided depthwise conv+BN when `kv_stride > 1` (cheaper
    attention over fewer keys/values); Q is never downsampled in any
    released MobileNetV4 config. BN (no activation) precedes the whole
    block; a learnable per-channel `LayerScale` (shared with FastViT §2.3)
    precedes the residual add - always present, stride=1, in==out
    channels, so the residual always applies here.
    '''
    channels = x.shape[-1]
    height, width = x.shape[1], x.shape[2]

    normed = BatchNormalization(epsilon=1e-3, name=f'{name}_norm')(x)

    q = Conv2D(num_heads * key_dim, 1, padding='same', use_bias=False, name=f'{name}_q_proj')(normed)
    q = Reshape((height * width, num_heads, key_dim), name=f'{name}_q_reshape')(q)

    kv_input = normed
    kv_h, kv_w = height, width
    if kv_stride > 1:
        kv_input = DepthwiseConv2D(3, strides=kv_stride, padding='same', use_bias=False, name=f'{name}_kv_dw')(kv_input)
        kv_input = BatchNormalization(epsilon=1e-3, name=f'{name}_kv_dw_bn')(kv_input)
        kv_h, kv_w = kv_input.shape[1], kv_input.shape[2]

    k = Conv2D(key_dim, 1, padding='same', use_bias=False, name=f'{name}_k_proj')(kv_input)
    k = Reshape((kv_h * kv_w, key_dim), name=f'{name}_k_reshape')(k)
    v = Conv2D(key_dim, 1, padding='same', use_bias=False, name=f'{name}_v_proj')(kv_input)
    v = Reshape((kv_h * kv_w, key_dim), name=f'{name}_v_reshape')(v)

    # q: (B, HW, heads, key_dim), k/v: (B, HW_kv, key_dim) shared across heads.
    attn_out = MultiQueryAttention(name=f'{name}_attn')([q, k, v])
    attn_out = Reshape((height, width, num_heads * key_dim), name=f'{name}_out_reshape')(attn_out)
    attn_out = Conv2D(channels, 1, padding='same', use_bias=True, name=f'{name}_out_proj')(attn_out)

    attn_out = LayerScale(layer_scale_init, name=f'{name}_scale')(attn_out)
    return Add(name=f'{name}_residual')([attn_out, x])


def _stage_output_name(model_name, stage_idx):
    return f'{model_name}_stage{stage_idx}_out'


def _build_stage(x, blocks, stage_idx, activation, model_name):
    spatial_seen = False
    for block_idx, spec in enumerate(blocks):
        kind = spec[0]
        name = f'{model_name}_stage{stage_idx}_block{block_idx}'

        if kind == 'mqa':
            _, num_heads, key_dim, kv_stride = spec
            x = mobile_mqa_block(x, num_heads, key_dim, kv_stride, activation=activation, name=name)
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
            x = uib_block(x, out_channels, start_k, mid_k, stride, expand, activation=activation, name=name)
        else:
            raise ValueError(f'Unknown block kind: {kind}')

    x = Activation('linear', name=_stage_output_name(model_name, stage_idx))(x)
    return x


def _mobilenetv4_network(input_tensor, stem_channels, stages, head_channels, activation,
                         include_top=False, pooling=None, num_classes=1000, name='MobileNetV4'):
    x = _conv_bn_act(input_tensor, stem_channels, 3, 2, activation, name=f'{name}_stem')
    x = Activation('linear', name=f'{name}_stem_out')(x)

    for stage_idx, blocks in enumerate(stages):
        x = _build_stage(x, blocks, stage_idx, activation, name)

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


def _build_mobilenetv4(stem_channels, stages, head_channels, activation, model_name):
    def constructor(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs):
        if weights not in (None, 'none'):
            raise NotImplementedError(
                "weights='{}' is not supported for {} yet - only weights=None (random "
                "initialization) is implemented so far. No official TF/Keras checkpoint exists "
                "for this family at all (plan.md §12); porting timm's PyTorch weights is tracked "
                "separately (plan.md §12.1).".format(weights, model_name)
            )

        if input_tensor is None:
            if input_shape is None:
                raise ValueError('Either `input_tensor` or `input_shape` must be provided.')
            input_tensor = Input(shape=input_shape)

        output = _mobilenetv4_network(
            input_tensor, stem_channels, stages, head_channels, activation,
            include_top=include_top, pooling=pooling, num_classes=classes, name=model_name,
        )
        return Model(inputs=input_tensor, outputs=output, name=model_name)

    constructor.__name__ = model_name
    return constructor


# Per-variant stage configs transcribed from timm's `_gen_mobilenet_v4`
# arch-def strings (plan.md §12 research) - block kind tags:
#   ('cn', out_channels, kernel_size)                      - plain conv
#   ('er', out_channels, kernel_size, expand_ratio)         - fused edge residual
#   ('uir', out_channels, start_k, mid_k, expand_ratio)     - universal inverted bottleneck
#   ('mqa', num_heads, key_dim, kv_stride)                  - mobile multi-query attention
# Every stage's first spatial (non-mqa) block carries that stage's stride-2
# transition; every other block is stride=1 - confirmed from the source's
# per-variant resolution progression (112->56->28->14->7), not assumed.
MOBILENETV4_CONFIGS = {
    'MobileNetV4-Conv-S': dict(
        stem_channels=32, head_channels=960, activation='relu', model_name='MobileNetV4_ConvS',
        stages=[
            [('cn', 32, 3), ('cn', 32, 3)],
            [('cn', 96, 3), ('cn', 64, 3)],
            [('uir', 96, 5, 5, 3), ('uir', 96, 0, 3, 2), ('uir', 96, 0, 3, 2), ('uir', 96, 0, 3, 2),
             ('uir', 96, 0, 3, 2), ('uir', 96, 3, 0, 4)],
            [('uir', 128, 3, 3, 6), ('uir', 128, 5, 5, 4), ('uir', 128, 0, 5, 4), ('uir', 128, 0, 5, 4),
             ('uir', 128, 0, 3, 3), ('uir', 128, 0, 3, 4)],
        ],
    ),
    'MobileNetV4-Conv-M': dict(
        stem_channels=32, head_channels=960, activation='relu', model_name='MobileNetV4_ConvM',
        stages=[
            [('er', 48, 3, 4)],
            [('uir', 80, 3, 5, 4), ('uir', 80, 3, 3, 2)],
            [('uir', 160, 3, 5, 6), ('uir', 160, 3, 3, 4), ('uir', 160, 3, 3, 4), ('uir', 160, 3, 5, 4),
             ('uir', 160, 3, 3, 4), ('uir', 160, 3, 0, 2), ('uir', 160, 0, 0, 4), ('uir', 160, 3, 0, 4)],
            [('uir', 256, 5, 5, 6), ('uir', 256, 5, 5, 4), ('uir', 256, 3, 5, 4), ('uir', 256, 3, 5, 4),
             ('uir', 256, 0, 0, 2), ('uir', 256, 3, 0, 4), ('uir', 256, 3, 5, 4), ('uir', 256, 5, 5, 4),
             ('uir', 256, 0, 0, 4), ('uir', 256, 0, 0, 4), ('uir', 256, 5, 0, 2)],
        ],
    ),
    'MobileNetV4-Conv-L': dict(
        stem_channels=24, head_channels=960, activation='relu', model_name='MobileNetV4_ConvL',
        stages=[
            [('er', 48, 3, 4)],
            [('uir', 96, 3, 5, 4), ('uir', 96, 3, 3, 4)],
            [('uir', 192, 3, 5, 4), ('uir', 192, 3, 3, 4), ('uir', 192, 3, 3, 4), ('uir', 192, 3, 3, 4),
             ('uir', 192, 3, 5, 4), ('uir', 192, 5, 3, 4), ('uir', 192, 5, 3, 4), ('uir', 192, 3, 0, 4)],
            [('uir', 512, 5, 5, 4), ('uir', 512, 5, 5, 4), ('uir', 512, 5, 5, 4), ('uir', 512, 5, 5, 4),
             ('uir', 512, 5, 0, 4), ('uir', 512, 5, 3, 4), ('uir', 512, 5, 0, 4), ('uir', 512, 5, 0, 4),
             ('uir', 512, 5, 3, 4), ('uir', 512, 5, 5, 4), ('uir', 512, 5, 0, 4), ('uir', 512, 5, 0, 4),
             ('uir', 512, 5, 0, 4)],
        ],
    ),
    'MobileNetV4-Hybrid-M': dict(
        stem_channels=32, head_channels=960, activation='relu', model_name='MobileNetV4_HybridM',
        stages=[
            [('er', 48, 3, 4)],
            [('uir', 80, 3, 5, 4), ('uir', 80, 3, 3, 2)],
            [('uir', 160, 3, 5, 4), ('uir', 160, 0, 0, 4), ('uir', 160, 3, 3, 4), ('uir', 160, 3, 5, 4),
             ('mqa', 4, 64, 2), ('uir', 160, 3, 3, 4), ('mqa', 4, 64, 2), ('uir', 160, 3, 0, 4),
             ('mqa', 4, 64, 2), ('uir', 160, 3, 3, 4), ('mqa', 4, 64, 2), ('uir', 160, 3, 0, 4)],
            [('uir', 256, 5, 5, 4), ('uir', 256, 5, 5, 4), ('uir', 256, 3, 5, 4), ('uir', 256, 3, 5, 4),
             ('uir', 256, 0, 0, 2), ('uir', 256, 3, 5, 2), ('uir', 256, 0, 0, 2), ('uir', 256, 0, 0, 4),
             ('mqa', 4, 64, 1), ('uir', 256, 3, 0, 4), ('mqa', 4, 64, 1), ('uir', 256, 5, 5, 4),
             ('mqa', 4, 64, 1), ('uir', 256, 5, 0, 4), ('mqa', 4, 64, 1), ('uir', 256, 5, 0, 4)],
        ],
    ),
    'MobileNetV4-Hybrid-L': dict(
        stem_channels=24, head_channels=960, activation='gelu', model_name='MobileNetV4_HybridL',
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


def mobilenetv4_preprocess_input(x):
    '''[-1, 1] rescaling (MobileNet-family convention) - a placeholder until
    real pretrained weights are ported (plan.md §12/§12.1); see the
    identical note on the other custom backbones' `*_preprocess_input`.
    '''
    return (x / 127.5) - 1.0
