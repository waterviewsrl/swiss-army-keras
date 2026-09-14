'''FastViT (Apple, 2023) — "FastViT: A Fast Hybrid Vision Transformer using
Structural Reparameterization" (https://arxiv.org/abs/2303.14189).

Architecture verified against the reference Keras port in
`keras_cv_attention_models` (leondgarse/keras_cv_attention_models,
`fastvit/fastvit.py`) per plan.md §12 — per-stage channel/depth/mlp-ratio
configs, block structure, activation, and which stages get attention are
taken from that source, not guessed (plan.md §7).

4 stages, all variants: stages 1-3 are always RepMixer (conv-only, token
mixer = a reparameterizable depthwise conv); stage 4 is RepMixer for
T8/T12/S12 and self-attention for SA12/SA24/SA36/MA36. Every block (RepMixer
or attention) is followed by the same ConvFFN tail.

Reparameterization is more involved here than RepViT's (plan.md §2.4): a
RepMixer block's training-time token mixer has *four* linear branches, one
of which is *subtracted* (not summed), plus a learnable per-channel
"layer-scale" and a separate residual add — all algebraically foldable into
one `DepthwiseConv2D`, using the same `reparam_utils.py` primitives
(`fuse_conv_bn`, `identity_kernel`, `scale_kernel_bias`, `sum_branches` —
subtraction via negating a branch before summing) plus one more primitive
this backbone needed and `reparam_utils.py` didn't have yet
(`scale_kernel_bias`, for the layer-scale fold). The stem and inter-stage
downsample modules are separately reparameterizable (MobileOne-style 2-3
branch sums), *not* wrapped in the subtract/layer-scale/residual dance that
only applies to RepMixer's *token-mixer* usage of the shared multi-branch
`rep_conv_block` primitive.

Native feature pyramid: like RepViT, FastViT's stem is a 3-conv patchify
stem reaching 4x downsampling before stage 1, with no exposed intermediate
2x tap — 4 native taps (4x/8x/16x/32x), not 5. Same
`backbone_zoo()`/`depth_max` clamp handles this as RepViT's registration
does.

Weights: no `weights='imagenet'` support in this pass — porting the
`keras_cv_attention_models` checkpoints is separate follow-up work (plan.md
§12/§12.1). `weights=None` (random init) is fully supported.
'''

from __future__ import absolute_import

import numpy as np
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
    MultiHeadAttention,
    Reshape,
    Subtract,
    ZeroPadding2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable

from swiss_army_keras.activations import GELU
from swiss_army_keras.reparam_utils import (
    fuse_conv_bn,
    fuse_conv_bn_layers,
    identity_kernel,
    pad_kernel_to,
    scale_kernel_bias,
    sum_branches,
)


def fastvit_preprocess_input(x):
    '''[-1, 1] rescaling (MobileNet-family convention) - a placeholder until
    real pretrained weights are ported (plan.md §12/§12.1); see the
    identical note on `repvit_preprocess_input` in `_backbone_repvit.py`.
    '''
    return (x / 127.5) - 1.0


@register_keras_serializable(package='swiss_army_keras')
class LayerScale(Layer):
    '''Per-channel learnable scale, applied to a block's mixer/attention
    output before its residual add ("layer scale", Touvron et al. 2021) -
    used throughout FastViT's RepMixer, attention, and ConvFFN blocks.
    Registered via `register_keras_serializable` from the start (unlike
    `GELU`/`Snake`, whose missing registration broke QAT integration for
    RepViT §2.4 in a way that was only caught by exercising that exact code
    path — see plan.md §10 — this class is registered proactively instead
    of waiting to rediscover the same bug).
    '''

    def __init__(self, init_value=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma', shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Constant(self.init_value), trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update({'init_value': self.init_value})
        return config


def rep_conv_block(x, out_channels, kernel_size=3, stride=1, groups=1, deploy=False, epsilon=1e-3, name=''):
    '''General MobileOne-style multi-branch reparameterizable conv: a
    `kernel_size`x`kernel_size` conv+BN branch, a 1x1 conv+BN branch (same
    `groups`/`stride`), and — only when `stride==1` and `in_channels ==
    out_channels` (the standard RepVGG-style rule) — an identity+BN branch,
    all summed. No trailing BN, no activation, no residual here (unlike
    RepViT's RepVGGDW) - those are added by the caller where applicable
    (`fastvit_repmixer_block` for the RepMixer-usage case; the stem/downsample
    callers just apply an activation directly). At `deploy=True`, a single
    fused conv (`{name}_fused`) replaces the whole branch structure.
    '''
    in_channels = x.shape[-1]
    use_identity = (stride == 1 and in_channels == out_channels)
    is_depthwise = (groups == out_channels)

    def make_conv(k, s, use_bias, layer_name):
        # `Conv2D(filters, kernel_size, ...)` and `DepthwiseConv2D(kernel_size, ...)`
        # take a *different* first positional argument - passing `kernel_size`
        # positionally into `Conv2D` silently collides with `filters` if both
        # are supplied, so build each branch's call explicitly instead of
        # sharing one positional-argument call shape between the two classes.
        #
        # Explicit symmetric padding (`ZeroPadding2D(k // 2)` + `valid`)
        # instead of `padding='same'`: at `stride > 1`, `'same'` padding can
        # be asymmetric in a way that depends on kernel size, so a KxK
        # branch and a 1x1 branch summed here (at the stem's stride-2 calls)
        # don't necessarily sample the same input window - `pad_kernel_to`'s
        # symmetric-centering assumption then silently breaks. Confirmed via
        # an isolated stem test (see plan.md §13 Phase 2's MobileOne entry,
        # where this was found first, then audited and fixed here too).
        # Harmless at `stride == 1` (where `'same'` for an odd kernel is
        # always exactly symmetric already) - applied unconditionally anyway
        # for one uniform code path.
        def apply(inputs):
            padded = ZeroPadding2D(k // 2, name=f'{layer_name}_pad')(inputs) if k > 1 else inputs
            if is_depthwise:
                return DepthwiseConv2D(k, strides=s, padding='valid', use_bias=use_bias, name=layer_name)(padded)
            return Conv2D(out_channels, k, strides=s, padding='valid', use_bias=use_bias, name=layer_name)(padded)
        return apply

    if deploy:
        return make_conv(kernel_size, stride, True, f'{name}_fused')(x)

    kxk = make_conv(kernel_size, stride, False, f'{name}_kxk')(x)
    kxk = BatchNormalization(epsilon=epsilon, name=f'{name}_kxk_bn')(kxk)

    onexone = make_conv(1, stride, False, f'{name}_1x1')(x)
    onexone = BatchNormalization(epsilon=epsilon, name=f'{name}_1x1_bn')(onexone)

    branches = [kxk, onexone]
    if use_identity:
        branches.append(BatchNormalization(epsilon=epsilon, name=f'{name}_identity_bn')(x))

    return Add(name=f'{name}_add')(branches) if len(branches) > 1 else branches[0]


def rep_downsample(x, out_channels, deploy=False, epsilon=1e-3, name=''):
    '''Inter-stage transition: a 2-branch (7x7 + 3x3, both depthwise,
    stride-2) reparameterizable token mixer (channel-preserving), followed
    by a `rep_conv_block(kernel_size=1, groups=1)` pointwise projection to
    `out_channels` + GELU. Stage 1 has no downsample module (the stem
    already reaches stage 1's resolution).
    '''
    in_channels = x.shape[-1]

    # Explicit symmetric padding, not `padding='same'` - the 7x7 and 3x3
    # branches are summed (kernel-fused via `pad_kernel_to`, which assumes
    # both sample the same input window), and at stride=2 `'same'`'s
    # asymmetry can desynchronize that assumption depending on kernel size
    # (see the identical note on `rep_conv_block.make_conv` above).
    if deploy:
        mixed = ZeroPadding2D(3, name=f'{name}_dw_pad')(x)
        mixed = DepthwiseConv2D(7, strides=2, padding='valid', use_bias=True, name=f'{name}_dw_fused')(mixed)
    else:
        dw7 = ZeroPadding2D(3, name=f'{name}_dw7_pad')(x)
        dw7 = DepthwiseConv2D(7, strides=2, padding='valid', use_bias=False, name=f'{name}_dw7')(dw7)
        dw7 = BatchNormalization(epsilon=epsilon, name=f'{name}_dw7_bn')(dw7)
        dw3 = ZeroPadding2D(1, name=f'{name}_dw3_pad')(x)
        dw3 = DepthwiseConv2D(3, strides=2, padding='valid', use_bias=False, name=f'{name}_dw3')(dw3)
        dw3 = BatchNormalization(epsilon=epsilon, name=f'{name}_dw3_bn')(dw3)
        mixed = Add(name=f'{name}_dw_add')([dw7, dw3])

    projected = rep_conv_block(mixed, out_channels, kernel_size=1, stride=1, groups=1, deploy=deploy, epsilon=epsilon, name=f'{name}_proj')
    return GELU(name=f'{name}_act')(projected)


def rep_cpe(x, deploy=False, name=''):
    '''Conditional positional encoding: a single depthwise 7x7 conv (with
    bias, no BN) plus a raw residual at train time, collapsing to just the
    fused conv at deploy. Applied once per attention-bearing stage, before
    its first attention block - never for pure-RepMixer stages/variants.
    '''
    channels = x.shape[-1]
    if deploy:
        return DepthwiseConv2D(7, padding='same', use_bias=True, name=f'{name}_fused')(x)

    dw = DepthwiseConv2D(7, padding='same', use_bias=True, name=f'{name}_dw')(x)
    return Add(name=f'{name}_add')([dw, x])


def fastvit_convffn(x, mlp_ratio, layer_scale_init=1e-5, epsilon=1e-3, name=''):
    '''Shared FFN tail after every block type (RepMixer or attention): a
    plain depthwise 7x7 "positional" conv + BN (NOT multi-branch — ordinary
    conv-BN folding applies at deploy, no separate reparameterization),
    then a 2-layer 1x1-conv MLP (expand by `mlp_ratio`, GELU, contract),
    layer-scaled, with its own residual (always present, at both train and
    deploy time — never fused away).
    '''
    channels = x.shape[-1]
    mlp = DepthwiseConv2D(7, padding='same', use_bias=False, name=f'{name}_pos_dw')(x)
    mlp = BatchNormalization(epsilon=epsilon, name=f'{name}_pos_bn')(mlp)

    y = Conv2D(int(channels * mlp_ratio), 1, padding='same', name=f'{name}_fc1')(mlp)
    y = GELU(name=f'{name}_act')(y)
    y = Conv2D(channels, 1, padding='same', name=f'{name}_fc2')(y)
    y = LayerScale(layer_scale_init, name=f'{name}_scale')(y)
    return Add(name=f'{name}_residual')([y, mlp])


def fastvit_repmixer_block(x, channels, mlp_ratio, layer_scale_init=1e-5, deploy=False, epsilon=1e-3, name=''):
    '''One RepMixer block: reparameterizable token mixer + ConvFFN.

    Training-time token mixer (4 linear branches):
        deep_mixer = rep_conv_block(x, channels, k=3, groups=channels)   # 3-way sum: 3x3+BN, 1x1+BN, identity+BN
        norm_mixer = BN(x)                                               # a *4th*, separate branch
        mixer = deep_mixer - norm_mixer                                  # subtracted, not summed
        mixer = LayerScale(mixer)
        token_mixed = mixer + x                                         # residual
    At `deploy=True` this entire chain (all 4 branches, the subtraction,
    the layer-scale, and the residual) collapses to one fused
    `DepthwiseConv2D` — see `reparameterize_fastvit` for the fusion math.
    '''
    if deploy:
        token_mixed = DepthwiseConv2D(3, padding='same', use_bias=True, name=f'{name}_token_fused')(x)
    else:
        deep_mixer = rep_conv_block(x, channels, kernel_size=3, stride=1, groups=channels, deploy=False, epsilon=epsilon, name=f'{name}_token_deep')
        norm_mixer = BatchNormalization(epsilon=epsilon, name=f'{name}_token_norm')(x)
        mixer = Subtract(name=f'{name}_token_sub')([deep_mixer, norm_mixer])
        mixer = LayerScale(layer_scale_init, name=f'{name}_token_scale')(mixer)
        token_mixed = Add(name=f'{name}_token_residual')([mixer, x])

    return fastvit_convffn(token_mixed, mlp_ratio, layer_scale_init, epsilon=epsilon, name=f'{name}_convffn')


def fastvit_attention_block(x, num_heads, mlp_ratio, layer_scale_init=1e-5, epsilon=1e-3, name=''):
    '''One attention block (stage 4 of SA12/SA24/SA36/MA36 only): BN ->
    global multi-head self-attention over the flattened spatial grid
    (head_dim=32 fixed, per the reference) -> layer-scale -> residual
    (never reparameterized/fused, unlike the RepMixer path) -> ConvFFN.
    '''
    height, width, channels = x.shape[1], x.shape[2], x.shape[3]

    normed = BatchNormalization(epsilon=epsilon, name=f'{name}_norm')(x)
    seq = Reshape((height * width, channels), name=f'{name}_flatten')(normed)
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=32, name=f'{name}_mhsa')(seq, seq)
    attn = Reshape((height, width, channels), name=f'{name}_unflatten')(attn)
    attn = LayerScale(layer_scale_init, name=f'{name}_scale')(attn)
    mixed = Add(name=f'{name}_residual')([attn, x])

    return fastvit_convffn(mixed, mlp_ratio, layer_scale_init, epsilon=epsilon, name=f'{name}_convffn')


def fastvit_stem(x, stem_width, deploy=False, epsilon=1e-3, name=''):
    '''3-conv patchify stem, all reparameterizable: regular 3x3 stride-2
    (3->stem_width) + GELU, depthwise 3x3 stride-2 + GELU, pointwise 1x1
    stride-1 + GELU. Total stride 4x, no exposed intermediate tap.
    '''
    x = rep_conv_block(x, stem_width, kernel_size=3, stride=2, groups=1, deploy=deploy, epsilon=epsilon, name=f'{name}_1')
    x = GELU(name=f'{name}_1_act')(x)
    x = rep_conv_block(x, stem_width, kernel_size=3, stride=2, groups=stem_width, deploy=deploy, epsilon=epsilon, name=f'{name}_2')
    x = GELU(name=f'{name}_2_act')(x)
    x = rep_conv_block(x, stem_width, kernel_size=1, stride=1, groups=1, deploy=deploy, epsilon=epsilon, name=f'{name}_3')
    x = GELU(name=f'{name}_3_act')(x)
    return x


def fastvit_stage_output_names(model_name, num_stages=4):
    '''The 4 pyramid-tap layer names for a given model - the output of the
    last block in each stage (a ConvFFN residual `Add`, for every block
    type). Used both to register `BACKBONE_REGISTRY` entries and internally
    by `reparameterize_fastvit`.
    '''
    return tuple(f'{model_name}_stage{i + 1}_out' for i in range(num_stages))


def _fastvit_network(input_tensor, stage_channels, stage_blocks, stage_types, mlp_ratio, stem_width,
                     layer_scale_init=1e-5, include_top=False, pooling=None, num_classes=1000, deploy=False, name='FastViT'):
    x = fastvit_stem(input_tensor, stem_width, deploy=deploy, name=f'{name}_stem')

    for stage_idx, (channels, num_blocks, stage_type) in enumerate(zip(stage_channels, stage_blocks, stage_types)):
        stage_name = f'{name}_stage{stage_idx + 1}'

        if stage_idx > 0:
            x = rep_downsample(x, channels, deploy=deploy, name=f'{stage_name}_downsample')

        if stage_type == 'transformer':
            x = rep_cpe(x, deploy=deploy, name=f'{stage_name}_cpe')
            num_heads = channels // 32
            for block_id in range(num_blocks):
                x = fastvit_attention_block(x, num_heads, mlp_ratio, layer_scale_init, name=f'{stage_name}_block{block_id}')
        else:
            for block_id in range(num_blocks):
                x = fastvit_repmixer_block(x, channels, mlp_ratio, layer_scale_init, deploy=deploy, name=f'{stage_name}_block{block_id}')

        # Renamed via an identity-preserving op so the tap has a fixed,
        # variant-independent name (`{name}_stageN_out`) regardless of how
        # many blocks the stage has (the last block's own layer name varies
        # by block count/type) - see `fastvit_stage_output_names`.
        x = Activation('linear', name=f'{stage_name}_out')(x)

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


def _build_fastvit(stage_channels, stage_blocks, stage_types, mlp_ratio, stem_width, layer_scale_init, model_name):
    def constructor(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs):
        if weights not in (None, 'none'):
            raise NotImplementedError(
                "weights='{}' is not supported for {} yet - only weights=None (random "
                "initialization) is implemented so far. Porting the keras_cv_attention_models "
                "checkpoints is tracked separately (plan.md §12/§12.1).".format(weights, model_name)
            )

        if input_tensor is None:
            if input_shape is None:
                raise ValueError('Either `input_tensor` or `input_shape` must be provided.')
            input_tensor = Input(shape=input_shape)

        output = _fastvit_network(
            input_tensor, stage_channels, stage_blocks, stage_types, mlp_ratio, stem_width,
            layer_scale_init=layer_scale_init, include_top=include_top, pooling=pooling,
            num_classes=classes, name=model_name,
        )
        model = Model(inputs=input_tensor, outputs=output, name=model_name)
        model._fastvit_config = dict(
            stage_channels=stage_channels, stage_blocks=stage_blocks, stage_types=stage_types,
            mlp_ratio=mlp_ratio, stem_width=stem_width, layer_scale_init=layer_scale_init, model_name=model_name,
        )
        return model

    constructor.__name__ = model_name
    return constructor


# Per-variant configs verified against keras_cv_attention_models' `fastvit.py`
# (plan.md §12 research). Stages 1-3 are always 'conv' (RepMixer); stage 4 is
# 'conv' for T8/T12/S12, 'transformer' (attention) for SA12/SA24/SA36/MA36.
FASTVIT_CONFIGS = {
    'FastViT-T8': dict(
        stage_channels=[48, 96, 192, 384], stage_blocks=[2, 2, 4, 2],
        stage_types=['conv', 'conv', 'conv', 'conv'], mlp_ratio=3, stem_width=48,
        layer_scale_init=1e-5, model_name='FastViT_T8',
    ),
    'FastViT-T12': dict(
        stage_channels=[64, 128, 256, 512], stage_blocks=[2, 2, 6, 2],
        stage_types=['conv', 'conv', 'conv', 'conv'], mlp_ratio=3, stem_width=64,
        layer_scale_init=1e-5, model_name='FastViT_T12',
    ),
    'FastViT-S12': dict(
        stage_channels=[64, 128, 256, 512], stage_blocks=[2, 2, 6, 2],
        stage_types=['conv', 'conv', 'conv', 'conv'], mlp_ratio=4, stem_width=64,
        layer_scale_init=1e-5, model_name='FastViT_S12',
    ),
    'FastViT-SA12': dict(
        stage_channels=[64, 128, 256, 512], stage_blocks=[2, 2, 6, 2],
        stage_types=['conv', 'conv', 'conv', 'transformer'], mlp_ratio=4, stem_width=64,
        layer_scale_init=1e-5, model_name='FastViT_SA12',
    ),
    'FastViT-SA24': dict(
        stage_channels=[64, 128, 256, 512], stage_blocks=[4, 4, 12, 4],
        stage_types=['conv', 'conv', 'conv', 'transformer'], mlp_ratio=4, stem_width=64,
        layer_scale_init=1e-5, model_name='FastViT_SA24',
    ),
    'FastViT-SA36': dict(
        stage_channels=[64, 128, 256, 512], stage_blocks=[6, 6, 18, 6],
        stage_types=['conv', 'conv', 'conv', 'transformer'], mlp_ratio=4, stem_width=64,
        layer_scale_init=1e-6, model_name='FastViT_SA36',
    ),
    'FastViT-MA36': dict(
        stage_channels=[76, 152, 304, 608], stage_blocks=[6, 6, 18, 6],
        stage_types=['conv', 'conv', 'conv', 'transformer'], mlp_ratio=4, stem_width=76,
        layer_scale_init=1e-6, model_name='FastViT_MA36',
    ),
}

FastViT_T8 = _build_fastvit(**FASTVIT_CONFIGS['FastViT-T8'])
FastViT_T12 = _build_fastvit(**FASTVIT_CONFIGS['FastViT-T12'])
FastViT_S12 = _build_fastvit(**FASTVIT_CONFIGS['FastViT-S12'])
FastViT_SA12 = _build_fastvit(**FASTVIT_CONFIGS['FastViT-SA12'])
FastViT_SA24 = _build_fastvit(**FASTVIT_CONFIGS['FastViT-SA24'])
FastViT_SA36 = _build_fastvit(**FASTVIT_CONFIGS['FastViT-SA36'])
FastViT_MA36 = _build_fastvit(**FASTVIT_CONFIGS['FastViT-MA36'])


def _fuse_rep_conv_block(model, prefix, deploy_layer_name_suffix='_fused'):
    '''Shared fusion logic for any `rep_conv_block` usage (stem, downsample
    projection, or - via `_fuse_repmixer_token_mixer` - inside a RepMixer
    token mixer): fuse the KxK+BN branch, the 1x1+BN branch (padded to
    KxK), and, if present, the identity+BN branch, and sum them.
    '''
    kxk_layer = model.get_layer(f'{prefix}_kxk')
    kxk_bn = model.get_layer(f'{prefix}_kxk_bn')
    onexone_layer = model.get_layer(f'{prefix}_1x1')
    onexone_bn = model.get_layer(f'{prefix}_1x1_bn')

    k_kxk, b_kxk = fuse_conv_bn_layers(kxk_layer, kxk_bn)
    k_1x1, b_1x1 = fuse_conv_bn_layers(onexone_layer, onexone_bn)
    kernel_size = k_kxk.shape[0]
    k_1x1_padded = pad_kernel_to(k_1x1, kernel_size) if kernel_size != k_1x1.shape[0] else k_1x1

    branches = [(k_kxk, b_kxk), (k_1x1_padded, b_1x1)]

    identity_bn_name = f'{prefix}_identity_bn'
    if any(l.name == identity_bn_name for l in model.layers):
        identity_bn = model.get_layer(identity_bn_name)
        depthwise = isinstance(kxk_layer, DepthwiseConv2D)
        channels = k_kxk.shape[2] if depthwise else k_kxk.shape[3]
        id_kernel = identity_kernel(channels, kernel_size, depthwise=depthwise, dtype=k_kxk.dtype)
        gamma, beta, mean, var = identity_bn.get_weights()
        channel_axis = -2 if depthwise else -1
        k_id, b_id = fuse_conv_bn(id_kernel, None, gamma, beta, mean, var, epsilon=identity_bn.epsilon, channel_axis=channel_axis)
        branches.append((k_id, b_id))

    return sum_branches(branches)


def reparameterize_fastvit(model):
    '''Collapse every reparameterizable module in a trained FastViT model
    (stem convs, inter-stage downsample token mixers + projections, CPE, and
    RepMixer token mixers) into single fused convs, using `reparam_utils.py`.
    Returns a new `Model` built in deploy mode with identical inference-time
    behavior (within floating-point tolerance). Attention blocks (stage 4 of
    SA/MA variants) and every ConvFFN are never multi-branch, so their
    layers are copied verbatim.

    Input
    ----------
        model: a model returned by one of this module's constructors
               (`FastViT_T8`, etc.), used with `weights=None` and then
               trained - i.e. it must still have its `_fastvit_config`
               attribute (set at construction time).
    '''
    if not hasattr(model, '_fastvit_config'):
        raise ValueError(
            "`model` has no `_fastvit_config` - reparameterize_fastvit only supports models "
            "built by this module's own constructors (FastViT_T8, etc.)."
        )
    config = model._fastvit_config

    deploy_input = Input(shape=model.input_shape[1:])
    deploy_output = _fastvit_network(
        deploy_input, config['stage_channels'], config['stage_blocks'], config['stage_types'],
        config['mlp_ratio'], config['stem_width'], layer_scale_init=config['layer_scale_init'],
        deploy=True, name=config['model_name'],
    )
    deploy_model = Model(inputs=deploy_input, outputs=deploy_output, name=f"{config['model_name']}_fused")

    stem_fused_names = {f"{config['model_name']}_stem_{i}_fused" for i in (1, 2, 3)}
    downsample_dw_fused_names = set()
    downsample_proj_fused_names = set()
    token_fused_names = set()
    for stage_idx in range(1, len(config['stage_channels'])):
        stage_name = f"{config['model_name']}_stage{stage_idx + 1}"
        downsample_dw_fused_names.add(f'{stage_name}_downsample_dw_fused')
        downsample_proj_fused_names.add(f'{stage_name}_downsample_proj_fused')
    for stage_idx, (num_blocks, stage_type) in enumerate(zip(config['stage_blocks'], config['stage_types'])):
        if stage_type == 'conv':
            stage_name = f"{config['model_name']}_stage{stage_idx + 1}"
            for block_id in range(num_blocks):
                token_fused_names.add(f'{stage_name}_block{block_id}_token_fused')
    cpe_fused_names = {
        f"{config['model_name']}_stage{i + 1}_cpe_fused"
        for i, t in enumerate(config['stage_types']) if t == 'transformer'
    }

    fused_names = stem_fused_names | downsample_dw_fused_names | downsample_proj_fused_names | token_fused_names | cpe_fused_names

    for deploy_layer in deploy_model.layers:
        name = deploy_layer.name

        if name in stem_fused_names or name in downsample_proj_fused_names:
            prefix = name[: -len('_fused')]
            fused_kernel, fused_bias = _fuse_rep_conv_block(model, prefix)
            deploy_layer.set_weights([fused_kernel, fused_bias])

        elif name in downsample_dw_fused_names:
            prefix = name[: -len('_dw_fused')]
            dw7_layer = model.get_layer(f'{prefix}_dw7')
            dw7_bn = model.get_layer(f'{prefix}_dw7_bn')
            dw3_layer = model.get_layer(f'{prefix}_dw3')
            dw3_bn = model.get_layer(f'{prefix}_dw3_bn')
            k7, b7 = fuse_conv_bn_layers(dw7_layer, dw7_bn)
            k3, b3 = fuse_conv_bn_layers(dw3_layer, dw3_bn)
            k3_padded = pad_kernel_to(k3, 7)
            fused_kernel, fused_bias = sum_branches([(k7, b7), (k3_padded, b3)])
            deploy_layer.set_weights([fused_kernel, fused_bias])

        elif name in cpe_fused_names:
            prefix = name[: -len('_fused')]
            dw_layer = model.get_layer(f'{prefix}_dw')
            k, b = dw_layer.get_weights()
            channels = k.shape[2]
            id_kernel = identity_kernel(channels, 7, depthwise=True, dtype=k.dtype)
            id_bias = np.zeros((channels,), dtype=k.dtype)
            fused_kernel, fused_bias = sum_branches([(k, b), (id_kernel, id_bias)])
            deploy_layer.set_weights([fused_kernel, fused_bias])

        elif name in token_fused_names:
            prefix = name[: -len('_token_fused')]
            deep_kernel, deep_bias = _fuse_rep_conv_block(model, f'{prefix}_token_deep')

            norm_bn = model.get_layer(f'{prefix}_token_norm')
            channels = deep_kernel.shape[2]
            id_kernel = identity_kernel(channels, deep_kernel.shape[0], depthwise=True, dtype=deep_kernel.dtype)
            gamma, beta, mean, var = norm_bn.get_weights()
            norm_kernel, norm_bias = fuse_conv_bn(id_kernel, None, gamma, beta, mean, var, epsilon=norm_bn.epsilon, channel_axis=-2)

            # deep_mixer - norm_mixer: subtract via negation, then sum.
            mixer_kernel, mixer_bias = sum_branches([(deep_kernel, deep_bias), (-norm_kernel, -norm_bias)])

            scale_layer = model.get_layer(f'{prefix}_token_scale')
            (gamma_scale,) = scale_layer.get_weights()
            scaled_kernel, scaled_bias = scale_kernel_bias(mixer_kernel, mixer_bias, gamma_scale, channel_axis=-2)

            residual_kernel = identity_kernel(channels, deep_kernel.shape[0], depthwise=True, dtype=deep_kernel.dtype)
            residual_bias = np.zeros((channels,), dtype=deep_kernel.dtype)
            fused_kernel, fused_bias = sum_branches([(scaled_kernel, scaled_bias), (residual_kernel, residual_bias)])
            deploy_layer.set_weights([fused_kernel, fused_bias])

        elif deploy_layer.weights:
            deploy_layer.set_weights(model.get_layer(name).get_weights())

    deploy_model._fastvit_config = config
    return deploy_model
