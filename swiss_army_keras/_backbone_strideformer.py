'''StrideFormer (Baidu PaddleSeg, 2023, "PP-MobileSeg: Explore the Fast and
Accurate Semantic Segmentation Model on Mobile Devices",
arXiv:2304.05152).

Architecture verified directly against the official reference source
(`PaddlePaddle/PaddleSeg`, `paddleseg/models/backbones/strideformer.py`,
fetched and read verbatim, cross-checked against the two per-variant
training configs — `configs/pp_mobileseg/pp_mobileseg_{tiny,base}_
ade20k_512x512_80k.yml` — for the values the Python source itself leaves as
required kwargs with no default) — same rigor as every other backbone in
this plan.

**A real, deliberate departure from the reference's own module boundary**:
in the reference, AAM (the 3-level fusion module) lives *inside*
`StrideFormer.forward()` itself — the paper's own decode head
(`PPMobileSegHead`) is trivial (a single 1x1 conv → dropout → classifier
conv), consuming one already-fused tensor. That doesn't fit this repo's
`backbone_zoo` convention (every registered backbone exposes a *multi-level*
pyramid, consumed by a decoder's `feature_levels`/`deep_layer`). Resolved
by moving AAM into the `pp_mobileseg` *decoder* (`_model_pp_mobileseg.py`)
instead, and having this file expose the same 3 raw feature maps AAM itself
would otherwise consume as a genuine, swappable 3-tap backbone — usable by
*any* decoder in this repo (`lr_aspp`/`pp_liteseg`/`deeplab_v3_plus`), not
just `pp_mobileseg`, consistent with every other backbone here.

- **4 CNN stages (`StackedMV3Block`, MobileNetV3-style inverted-residual)
  interleaved sequentially with 2 transformer stages (`BasicLayer`, SEA
  attention) — not parallel branches, confirmed from source**: stem+stage1
  (`smb1`) → stage2 (`smb2`, **tap 1, stride8**, a raw CNN feature, no
  transformer applied) → stage3 (`smb3`) → SEA attention (`trans1`, applied
  to `smb3`'s output, **tap 2, stride16**) → stage4 (`smb4`, consumes
  `trans1`'s *output*, not `smb3`'s raw one) → SEA attention (`trans2`,
  applied to `smb4`'s output, **tap 3, stride32**).
- **`ResidualUnit`**: expand 1x1 conv+BN+act → depthwise KxK conv+BN+act →
  optional SE (**reduction is a fixed `channels // 4` divisor, not a
  ratio** — confirmed from source, unlike this repo's other two SE
  implementations, `_backbone_mobileone.py`/`_backbone_repvit.py`, which
  both use a ratio; gated by `HardSigmoid(slope=0.2, offset=0.5)`, **not**
  the more common MobileNetV3-paper hard-sigmoid formula
  `relu6(x+3)/6` (slope 1/6) — a different, deliberately-chosen slope,
  confirmed from `paddleseg/models/backbones/mobilenetv3.py::Hardsigmoid`)
  → project 1x1 conv+BN (no activation), residual iff stride1+channel-match.
- **`Sea_Attention`** ("squeeze-axial" row/column attention — the one
  genuinely novel mechanism in this repo, not textbook multi-head or
  multi-query attention):
  1. Optional stride-2 depthwise+BN downsample (`stride_attention=True` in
     both released variants), remember the pre-downsample H/W for a final
     upsample back.
  2. `q,k,v` via 1x1 conv+BN (`to_q`/`to_k` → `num_heads*key_dim` channels,
     `to_v` → `num_heads*key_dim*attn_ratio` channels).
  3. A parallel local-mixing branch, independent of the attention math
     below: `concat([q,k,v], channel) → depthwise 3x3 conv+BN → activation
     → pointwise conv → dim` channels.
  4. **Row branch**: mean-pool `q`/`k`/`v` over the *width* axis →
     `SqueezeAxialPositionalEmbedding` (a learned, linearly-interpolated 1D
     positional bias) added to pooled `q`/`k` → multi-head scaled
     dot-product attention over the *height* axis → `(B, H, 1, dh)`.
  5. **Column branch**: identical, pooling over *height*, attention over
     *width* → `(B, 1, W, dh)`.
  6. `xx = row + column` (broadcasts to `(B,H,W,dh)`) `+ v` (residual) →
     activation+conv (`proj`) → sigmoid → **gates the step-3 local-mixing
     branch elementwise** (not summed with it) → final `dim`-channel
     output.
  7. Bilinear upsample back to the pre-downsample H/W if `stride_attention`.
  Q/K/V projection convs and every plain conv/BN/activation stay in the
  functional API (individually `qat_annotate_layer`-wrapped, matching this
  repo's universal convention) — only the actual attention math (mean-pool,
  positional embedding, reshape/transpose, matmul+softmax) lives inside
  proper `Layer` subclasses (`SqueezeAxialPositionalEmbedding`,
  `AxialAttention`), the same QAT-model-cloning-safety split already
  established by MobileNetV4's `MultiQueryAttention` (§2.2): convs
  functional and individually annotated, only the "pure math" wrapped in a
  `Layer.call()`.

Native feature pyramid: a genuine **3-tap** pyramid (stride 8/16/32) — no
single-stride-2-conv stem tap or stride-4 tap the way MobileOne/MobileNetV4
expose (this family's first tap is already at stride8, `smb2`'s output),
fewer taps than RepViT/FastViT's 4-tap patchify-stem situation even.
Decoders consuming this backbone need an explicit `feature_levels=(1,2,3)`/
`deep_layer=3` override, the same pattern RepViT/FastViT's 4-tap pyramid
already requires elsewhere in this codebase.

Weights: `weights=None` (random init) only in this pass — real
ImageNet-pretrained weights exist (linked from the reference's own YAML
configs) but porting them is out of scope here, the same phased treatment
(architecture first) every other backbone/decoder in this plan got.
'''

from __future__ import absolute_import

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    DepthwiseConv2D,
    GlobalAveragePooling2D,
    Input,
    Layer,
    Multiply,
    ZeroPadding2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.utils import register_keras_serializable

from swiss_army_keras.layer_utils import qat_annotate_layer
from swiss_army_keras._backbone_mobilenetv4 import _conv_bn_act, _pad_same


@register_keras_serializable(package='swiss_army_keras')
class ReLU6(Layer):
    '''`min(max(x, 0), 6)` - `tf_keras`'s `Activation('relu6')` string form
    isn't registered under this repo's forced legacy-Keras-2 backend
    (`TF_USE_LEGACY_KERAS=1`, confirmed empirically: `tf_keras.activations`
    only exposes `hard_sigmoid`/`swish`, not `hard_swish`/`relu6`) - a
    proper `Layer` avoids depending on a string name that doesn't resolve
    in this repo's actual runtime.
    '''

    def call(self, x):
        return tf.nn.relu6(x)


@register_keras_serializable(package='swiss_army_keras')
class HardSwish(Layer):
    '''`x * relu6(x + 3) / 6` - the standard MobileNetV3 hard-swish, same
    "`tf_keras` doesn't register this activation string" reasoning as
    `ReLU6` above.
    '''

    def call(self, x):
        return x * tf.nn.relu6(x + 3.0) / 6.0


_ACT_NAME_MAP = {'relu': 'relu', 'hardswish': HardSwish, 'relu6': ReLU6}


@register_keras_serializable(package='swiss_army_keras')
class HardSigmoid(Layer):
    '''`clip(slope*x + offset, 0, 1)`, `slope=0.2, offset=0.5` - PaddleSeg's
    own hard-sigmoid variant (`paddleseg/models/backbones/mobilenetv3.py::
    Hardsigmoid`), confirmed from source: **not** the more common
    MobileNetV3-paper hard-sigmoid (`relu6(x+3)/6`, slope 1/6) - a
    different, deliberately-chosen slope.
    '''

    def __init__(self, slope=0.2, offset=0.5, **kwargs):
        super().__init__(**kwargs)
        self.slope = slope
        self.offset = offset

    def call(self, x):
        return tf.clip_by_value(self.slope * x + self.offset, 0.0, 1.0)

    def get_config(self):
        config = super().get_config()
        config.update({'slope': self.slope, 'offset': self.offset})
        return config


def _apply_activation(x, activation, name):
    '''`activation` may be a plain string (`Activation(activation, ...)`) or
    a callable `Layer` class (`ReLU6`/`HardSwish`, called directly) - the
    same dispatch `_backbone_mobilenetv4._conv_bn_act` already uses
    internally, factored out here since this file also needs it in several
    places `_conv_bn_act` itself doesn't cover (depthwise convs, the MLP,
    SEA attention's own activations).
    '''
    if callable(activation) and not isinstance(activation, str):
        return qat_annotate_layer(activation(name=name))(x)
    return qat_annotate_layer(Activation(activation, name=name))(x)


def _dw_conv_bn_act(x, kernel_size, stride, activation, name=''):
    x = _pad_same(x, kernel_size, name=f'{name}_dw')
    x = qat_annotate_layer(DepthwiseConv2D(
        kernel_size, strides=stride, padding='valid', use_bias=False, name=f'{name}_dw',
    ))(x)
    x = qat_annotate_layer(BatchNormalization(epsilon=1e-5, name=f'{name}_dw_bn'))(x)
    if activation is not None:
        x = _apply_activation(x, activation, name=f'{name}_dw_act')
    return x


def _se_module(x, channels, name=''):
    '''Squeeze-excite with a fixed `channels // 4` reduction (not a ratio)
    and `HardSigmoid(slope=0.2, offset=0.5)` gate - both confirmed from
    source, see module docstring.
    '''
    reduction = max(1, channels // 4)
    s = GlobalAveragePooling2D(keepdims=True, name=f'{name}_gap')(x)
    s = qat_annotate_layer(Conv2D(reduction, 1, padding='same', name=f'{name}_reduce'))(s)
    s = qat_annotate_layer(Activation('relu', name=f'{name}_relu'))(s)
    s = qat_annotate_layer(Conv2D(channels, 1, padding='same', name=f'{name}_expand'))(s)
    s = HardSigmoid(name=f'{name}_hsigmoid')(s)
    return Multiply(name=f'{name}_scale')([x, s])


def _residual_unit(x, mid_channels, out_channels, kernel_size, stride, use_se, activation, name=''):
    '''MobileNetV3's standard inverted-residual block (`ResidualUnit`):
    expand 1x1 -> depthwise KxK -> optional SE -> project 1x1 (no
    activation), residual iff stride1 and in==out channels.
    '''
    in_channels = x.shape[-1]
    act = _ACT_NAME_MAP[activation]
    use_shortcut = (stride == 1 and in_channels == out_channels)

    h = _conv_bn_act(x, mid_channels, 1, 1, act, name=f'{name}_expand')
    h = _dw_conv_bn_act(h, kernel_size, stride, act, name=f'{name}_bottleneck')
    if use_se:
        h = _se_module(h, mid_channels, name=f'{name}_se')
    h = _conv_bn_act(h, out_channels, 1, 1, act, use_act=False, name=f'{name}_project')

    if use_shortcut:
        h = Add(name=f'{name}_add')([h, x])
    return h


def _stacked_mv3_stage(x, cfgs, stem, stem_channels, name=''):
    '''One `StackedMV3Block` stage: an optional stride-2 3x3 stem
    conv+BN+hardswish (only the very first stage), then `cfgs`'s own
    `ResidualUnit`s in sequence. `cfgs`: list of
    `(kernel_size, mid_channels, out_channels, use_se, activation, stride)`
    tuples, transcribed directly from the reference's own per-variant
    `cfg1..cfg4` lists (`k, t, c, se, act, s` in the source's own naming).
    '''
    if stem:
        x = _conv_bn_act(x, stem_channels, 3, 2, HardSwish, name=f'{name}_stem')
    for i, (k, mid_c, out_c, se, act, s) in enumerate(cfgs):
        x = _residual_unit(x, mid_c, out_c, k, s, se, act, name=f'{name}_block{i}')
    return x


@register_keras_serializable(package='swiss_army_keras')
class SqueezeAxialPositionalEmbedding(Layer):
    '''A learned 1D positional bias, linearly interpolated to the actual
    sequence length at call time via `tf.image.resize` with a size-1 fake
    axis (bilinear resize with one degenerate spatial axis is exactly 1D
    linear interpolation) - confirmed from source
    (`SqueezeAxialPositionalEmbedding.forward`:
    `x + F.interpolate(pos_embed, size=(N,), mode='linear')`). Operates on
    `(batch, N, channels)` tensors (channels-last, unlike the reference's
    channels-first `(batch, channels, N)` - same values, different axis
    order).
    '''

    def __init__(self, base_size=16, **kwargs):
        super().__init__(**kwargs)
        self.base_size = base_size

    def build(self, input_shape):
        self.dim = int(input_shape[-1])
        self.pos_embed = self.add_weight(
            name='pos_embed', shape=(1, self.base_size, self.dim),
            initializer=tf.keras.initializers.RandomNormal(), trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        n = tf.shape(x)[1]
        pos = tf.reshape(self.pos_embed, (1, self.base_size, 1, self.dim))
        pos = tf.image.resize(pos, size=(n, 1), method='bilinear')
        pos = tf.reshape(pos, (1, n, self.dim))
        return x + pos

    def get_config(self):
        config = super().get_config()
        config.update({'base_size': self.base_size})
        return config


@register_keras_serializable(package='swiss_army_keras')
class AxialAttention(Layer):
    '''The row-or-column half of SEA attention's squeeze-axial mechanism -
    mean-pool `q`/`k`/`v` over one spatial axis, add a learned positional
    bias to the pooled `q`/`k`, run scaled dot-product multi-head attention
    over the *other* spatial axis, reshape back to a `(B,H,1,dh)` (row) or
    `(B,1,W,dh)` (column) map ready to broadcast-add with its counterpart -
    confirmed from source (`Sea_Attention.forward`'s row/column branches,
    identical up to which axis is pooled/attended). A proper `Layer`
    subclass (not raw functional `tf.reduce_mean`/`tf.matmul`/`Lambda`
    calls), for the same QAT-model-cloning-safety reason
    `MultiQueryAttention` (`_backbone_mobilenetv4.py`, §2.2) already is one
    - q/k/v projection convs stay functional/individually annotated
    outside this layer, only the attention math itself lives here.
    '''

    def __init__(self, num_heads, key_dim, axis, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.axis = axis  # 'row' (pool over width, attend over height) or 'column' (vice versa)
        self.scale = key_dim ** -0.5
        self.pos_embed_q = SqueezeAxialPositionalEmbedding(name=f'{self.name}_pos_q')
        self.pos_embed_k = SqueezeAxialPositionalEmbedding(name=f'{self.name}_pos_k')

    def _pool(self, x):
        # (B, H, W, C) -> (B, N, C), N = H (row branch) or W (column branch).
        pool_axis = 2 if self.axis == 'row' else 1
        return tf.reduce_mean(x, axis=pool_axis)

    def _split_heads(self, x):
        # (B, N, num_heads*key_dim) -> (B, num_heads, N, key_dim)
        n = tf.shape(x)[1]
        x = tf.reshape(x, (-1, n, self.num_heads, self.key_dim))
        return tf.transpose(x, (0, 2, 1, 3))

    def call(self, inputs):
        q, k, v = inputs
        value_dim = v.shape[-1] // self.num_heads

        q_pooled = self.pos_embed_q(self._pool(q))
        k_pooled = self.pos_embed_k(self._pool(k))
        v_pooled = self._pool(v)

        q_heads = self._split_heads(q_pooled)  # (B, heads, N, key_dim)
        k_heads = self._split_heads(k_pooled)  # (B, heads, N, key_dim)
        n = tf.shape(v_pooled)[1]
        v_heads = tf.transpose(
            tf.reshape(v_pooled, (-1, n, self.num_heads, value_dim)), (0, 2, 1, 3),
        )  # (B, heads, N, value_dim)

        attn = tf.matmul(q_heads, k_heads, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        out = tf.matmul(attn, v_heads)  # (B, heads, N, value_dim)

        out = tf.transpose(out, (0, 2, 1, 3))
        out = tf.reshape(out, (-1, n, self.num_heads * value_dim))  # (B, N, dh)

        if self.axis == 'row':
            return out[:, :, tf.newaxis, :]  # (B, H, 1, dh)
        return out[:, tf.newaxis, :, :]  # (B, 1, W, dh)

    def get_config(self):
        config = super().get_config()
        config.update({'num_heads': self.num_heads, 'key_dim': self.key_dim, 'axis': self.axis})
        return config


def _sea_attention(x, dim, key_dim, num_heads, attn_ratio, activation, stride_attention, name=''):
    '''One `Sea_Attention` block - see module docstring for the full
    mechanism. `dim` == `x`'s own channel count (attention is channel-
    preserving end to end, confirmed from source).
    '''
    act = _ACT_NAME_MAP.get(activation, activation)
    orig_h, orig_w = x.shape[1], x.shape[2]

    if stride_attention:
        x = _dw_conv_bn_act(x, 3, 2, None, name=f'{name}_stride')

    nh_kd = num_heads * key_dim
    dh = num_heads * key_dim * attn_ratio

    q = _conv_bn_act(x, nh_kd, 1, 1, act, use_act=False, name=f'{name}_to_q')
    k = _conv_bn_act(x, nh_kd, 1, 1, act, use_act=False, name=f'{name}_to_k')
    v = _conv_bn_act(x, dh, 1, 1, act, use_act=False, name=f'{name}_to_v')

    # Local-mixing branch (independent of the attention math below).
    qkv = Concatenate(axis=-1, name=f'{name}_qkv_concat')([q, k, v])
    qkv = _dw_conv_bn_act(qkv, 3, 1, act, name=f'{name}_qkv_dw')
    qkv = _conv_bn_act(qkv, dim, 1, 1, act, use_act=False, name=f'{name}_qkv_pw')

    xx_row = AxialAttention(num_heads, key_dim, axis='row', name=f'{name}_row_attn')([q, k, v])
    xx_col = AxialAttention(num_heads, key_dim, axis='column', name=f'{name}_col_attn')([q, k, v])
    xx = Add(name=f'{name}_row_col_add')([xx_row, xx_col])
    xx = Add(name=f'{name}_v_residual')([xx, v])

    xx = _apply_activation(xx, act, name=f'{name}_proj_act')
    xx = _conv_bn_act(xx, dim, 1, 1, act, use_act=False, name=f'{name}_proj')

    gate = qat_annotate_layer(Activation('sigmoid', name=f'{name}_gate'))(xx)
    out = Multiply(name=f'{name}_gated')([gate, qkv])

    if stride_attention:
        out = qat_annotate_layer(tf.keras.layers.UpSampling2D(
            size=(orig_h // out.shape[1], orig_w // out.shape[2]),
            interpolation='bilinear', name=f'{name}_upsample',
        ))(out)
    return out


def _mlp(x, hidden_features, out_features, activation, name=''):
    x = _conv_bn_act(x, hidden_features, 1, 1, activation, use_act=False, name=f'{name}_fc1')
    x = _dw_conv_bn_act(x, 3, 1, None, name=f'{name}_dw')
    x = _apply_activation(x, activation, name=f'{name}_act')
    x = _conv_bn_act(x, out_features, 1, 1, activation, use_act=False, name=f'{name}_fc2')
    return x


def _transformer_block(x, key_dim, num_heads, mlp_ratio, attn_ratio, activation, stride_attention, name=''):
    dim = x.shape[-1]
    act = _ACT_NAME_MAP.get(activation, activation)

    attn_out = _sea_attention(x, dim, key_dim, num_heads, attn_ratio, activation, stride_attention, name=f'{name}_attn')
    x = Add(name=f'{name}_attn_residual')([x, attn_out])

    mlp_out = _mlp(x, int(dim * mlp_ratio), dim, act, name=f'{name}_mlp')
    x = Add(name=f'{name}_mlp_residual')([x, mlp_out])
    return x


def _basic_layer(x, block_num, key_dim, num_heads, mlp_ratio, attn_ratio, activation, stride_attention, name=''):
    for i in range(block_num):
        x = _transformer_block(
            x, key_dim, num_heads, mlp_ratio, attn_ratio, activation, stride_attention, name=f'{name}_block{i}',
        )
    return x


def strideformer_stage_output_names(model_name):
    '''The 3 pyramid-tap layer names - stride8/16/32 (`smb2`/`trans1`/`trans2`'s
    own outputs), the same 3 tensors the reference's own AAM module
    consumes (see module docstring for why AAM itself lives in the decoder
    instead, not here).
    '''
    return (
        f'{model_name}_tap1_stride8',
        f'{model_name}_tap2_stride16',
        f'{model_name}_tap3_stride32',
    )


def _strideformer_network(input_tensor, cfgs, channels, depths, embed_dims, key_dims, num_heads,
                          mlp_ratios, attn_ratio, act_layer, stride_attention,
                          include_top=False, pooling=None, num_classes=1000, name='strideformer'):
    tap_names = strideformer_stage_output_names(name)

    x = _stacked_mv3_stage(input_tensor, cfgs[0], stem=True, stem_channels=channels[0], name=f'{name}_smb1')
    x = _stacked_mv3_stage(x, cfgs[1], stem=False, stem_channels=None, name=f'{name}_smb2')
    x = Activation('linear', name=tap_names[0])(x)
    tap1 = x

    x = _stacked_mv3_stage(x, cfgs[2], stem=False, stem_channels=None, name=f'{name}_smb3')
    x = _basic_layer(
        x, depths[0], key_dims[0], num_heads, mlp_ratios[0], attn_ratio, act_layer, stride_attention,
        name=f'{name}_trans1',
    )
    x = Activation('linear', name=tap_names[1])(x)
    tap2 = x

    x = _stacked_mv3_stage(x, cfgs[3], stem=False, stem_channels=None, name=f'{name}_smb4')
    x = _basic_layer(
        x, depths[1], key_dims[1], num_heads, mlp_ratios[1], attn_ratio, act_layer, stride_attention,
        name=f'{name}_trans2',
    )
    x = Activation('linear', name=tap_names[2])(x)
    tap3 = x

    if include_top or pooling is not None:
        # Matches every other custom backbone's `include_top`/`pooling`
        # convention, operating on the deepest tap - StrideFormer's own
        # reference has no classifier head at all (it's segmentation-only),
        # so this is this repo's own addition for interface consistency,
        # not a reference-faithful reproduction.
        if pooling == 'max':
            pooled = tf.keras.layers.GlobalMaxPooling2D(name=f'{name}_pool')(tap3)
        else:
            pooled = tf.keras.layers.GlobalAveragePooling2D(name=f'{name}_pool')(tap3)
        if include_top:
            return tf.keras.layers.Dense(num_classes, activation='softmax', name=f'{name}_head_dense')(pooled)
        return pooled

    return [tap1, tap2, tap3]


_ACT_NAME_MAP.setdefault('relu6', 'relu6')


def strideformer_preprocess_input(x):
    '''Standard ImageNet normalization (`x/255` then per-channel
    `(x-mean)/std`) - confirmed from the reference's own training configs
    (`configs/pp_mobileseg/*.yml`'s `Normalize` transform,
    `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`), the same
    normalization every other backbone in this plan's real (non-placeholder)
    preprocessing uses.
    '''
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    x = x / 255.0
    return (x - mean) / std


# Per-variant configs transcribed directly from `MobileSeg_Tiny`/
# `MobileSeg_Base` (`strideformer.py`'s own factory functions) - `cfgs`
# entries are `(kernel_size, mid_channels, out_channels, use_se, activation,
# stride)`, the source's own `(k, t, c, se, act, s)` tuples, re-ordered
# here so mid/out channels sit next to each other (`t`=expansion/mid
# channels, `c`=output channels - the source's own naming, not obvious
# without reading `ResidualUnit`'s constructor).
STRIDEFORMER_CONFIGS = {
    'StrideFormer-Tiny': dict(
        model_name='StrideFormer_Tiny',
        channels=[16, 24, 32, 64, 128],
        cfgs=[
            [(3, 16, 16, True, 'relu', 1), (3, 64, 32, False, 'relu', 2), (3, 48, 24, False, 'relu', 1)],
            [(5, 96, 32, True, 'hardswish', 2), (5, 96, 32, True, 'hardswish', 1)],
            [(5, 160, 64, True, 'hardswish', 2), (5, 160, 64, True, 'hardswish', 1)],
            [(3, 384, 128, True, 'hardswish', 2), (3, 384, 128, True, 'hardswish', 1)],
        ],
        depths=[2, 2],
        embed_dims=[64, 128],
        num_heads=4,
        key_dims=[16, 24],
    ),
    'StrideFormer-Base': dict(
        model_name='StrideFormer_Base',
        channels=[16, 32, 64, 128, 192],
        cfgs=[
            [(3, 16, 16, True, 'relu', 1), (3, 64, 32, False, 'relu', 2), (3, 96, 32, False, 'relu', 1)],
            [(5, 128, 64, True, 'hardswish', 2), (5, 240, 64, True, 'hardswish', 1)],
            [(5, 384, 128, True, 'hardswish', 2), (5, 384, 128, True, 'hardswish', 1)],
            [(5, 768, 192, True, 'hardswish', 2), (5, 768, 192, True, 'hardswish', 1)],
        ],
        depths=[3, 3],
        embed_dims=[128, 192],
        num_heads=8,
        key_dims=[16, 24],
    ),
}


def _build_strideformer(model_name, channels, cfgs, depths, embed_dims, num_heads, key_dims,
                        mlp_ratios=(2, 4), attn_ratio=2, act_layer='relu6', stride_attention=True):
    def constructor(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs):
        if weights == 'imagenet':
            raise NotImplementedError(
                "weights='imagenet' auto-download isn't available yet for {} - that needs "
                "real, publicly-hosted weight files this package can fetch, which is a "
                "separate, not-yet-done piece of work (plan.md §12). Porting the reference's "
                "own released PyTorch/Paddle checkpoints is tracked separately (plan.md §3.5)."
                .format(model_name)
            )

        if input_tensor is None:
            if input_shape is None:
                raise ValueError('Either `input_tensor` or `input_shape` must be provided.')
            input_tensor = Input(shape=input_shape)

        output = _strideformer_network(
            input_tensor, cfgs, channels, depths, embed_dims, key_dims, num_heads,
            mlp_ratios, attn_ratio, act_layer, stride_attention,
            include_top=include_top, pooling=pooling, num_classes=classes, name=model_name,
        )
        model = Model(inputs=input_tensor, outputs=output, name=model_name)

        if weights not in (None, 'none'):
            model.load_weights(weights, skip_mismatch=True)
        return model

    constructor.__name__ = model_name
    return constructor


StrideFormer_Tiny = _build_strideformer(**STRIDEFORMER_CONFIGS['StrideFormer-Tiny'])
StrideFormer_Base = _build_strideformer(**STRIDEFORMER_CONFIGS['StrideFormer-Base'])
