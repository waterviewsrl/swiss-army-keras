'''RepViT (Meituan/THU-MIG, 2023) — "RepViT: Revisiting Mobile CNN From ViT
Perspective" (https://arxiv.org/abs/2307.09283).

Architecture verified against the reference Keras port in
`keras_cv_attention_models` (leondgarse/keras_cv_attention_models,
`repvit/repvit.py`) per plan.md §12 — per-stage channel/depth configs, block
structure, activation, and SE placement rule below are taken from that
source, not guessed (plan.md §7).

Structural reparameterization: every RepViT block's token mixer
("RepVGGDW") is a 3-way sum (3x3 depthwise+BN branch, 1x1 depthwise+bias
branch, raw identity) at train time, collapsible into a single 3x3
depthwise conv at inference — see `reparameterize_repvit` at the bottom of
this file, built on the shared primitives in `reparam_utils.py` (plan.md §2.1/
§2.3/§2.4's shared reparam utility). It works by building a second,
"deploy-mode" copy of the same network (`repvit_block(..., deploy=True)`
swaps the 3-way sum for one fused depthwise conv) and copying/fusing weights
into it layer-by-layer, rather than mutating the original model's graph in
place — simpler and far less fragile than live graph surgery on Keras
internals.

Native feature pyramid: RepViT's stem is a patchify stem (two stacked
stride-2 3x3 convs, no exposed intermediate output) that jumps straight to
4x downsampling, and stage 1 does not itself downsample — so the network
natively exposes 4 distinct spatial resolutions (4x/8x/16x/32x), not the
5-tap 2x/4x/8x/16x/32x pyramid some other backbones in this repo expose.
`BACKBONE_REGISTRY` entries for RepViT therefore register 4 layer names, not
5 — `backbone_zoo()`'s existing `depth_max = len(cadidate)` clamp already
handles a backbone with fewer than 5 taps gracefully (see `_backbone_zoo.py`),
so this needs no special-casing there.

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
    Multiply,
    ZeroPadding2D,
)
from tensorflow.keras.models import Model

from swiss_army_keras.activations import GELU
from swiss_army_keras.reparam_utils import fuse_conv_bn, fuse_conv_bn_layers, identity_kernel, sum_branches


# Plain tuples, not `tf.constant(...)`: a module-level eager tensor forces
# TF's eager context to initialize at import time (any op, even building a
# constant, calls `ctx.ensure_initialized()`) - which then makes a later
# `tf.config.experimental.set_memory_growth(gpu, True)` call in user code
# raise `RuntimeError: Physical devices cannot be modified after being
# initialized`, since device config must happen before the context exists.
# A plain tuple broadcasts into a tensor lazily, inside `x - MEAN`/`/ STD`
# below, at call time - not at import time.
_REPVIT_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_REPVIT_IMAGENET_STD = (0.229, 0.224, 0.225)


def repvit_preprocess_input(x):
    '''Standard ImageNet normalization (`x/255` then per-channel
    `(x-mean)/std`, `mean=(0.485,0.456,0.406)`, `std=(0.229,0.224,0.225)`) -
    confirmed as the actual normalization the real ported ImageNet weights
    (`scripts/port_repvit_weights.py`, plan.md §12.1) expect, via
    `timm.data.resolve_data_config`. Was a `[-1,1]`-rescaling placeholder
    before those weights existed (see the identical, still-current note on
    `mobileone_preprocess_input`/`fastvit_preprocess_input`, whose own
    pretrained weights aren't ported yet) - harmless either way for
    `weights=None` (random-init) usage, since there is no fixed
    normalization to match in that case, but now load-bearing for anyone
    using the real weights.
    '''
    x = x / 255.0
    return (x - _REPVIT_IMAGENET_MEAN) / _REPVIT_IMAGENET_STD


def _make_divisible(value, divisor=8, min_value=None):
    '''Round `value` to the nearest multiple of `divisor`, never dropping
    more than 10% below the original (the standard MobileNet-family rule,
    matched here since RepViT's SE reduction width uses it).
    '''
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _squeeze_excite(x, channels, se_ratio=0.25, name=''):
    reduction = _make_divisible(channels * se_ratio, 8)
    s = GlobalAveragePooling2D(keepdims=True, name=f'{name}_gap')(x)
    s = Conv2D(reduction, 1, padding='same', name=f'{name}_reduce')(s)
    s = Activation('relu', name=f'{name}_relu')(s)
    s = Conv2D(channels, 1, padding='same', name=f'{name}_expand')(s)
    s = Activation('sigmoid', name=f'{name}_sigmoid')(s)
    return Multiply(name=f'{name}_scale')([x, s])


def repvit_block(x, channels, se_ratio=0.0, epsilon=1e-5, deploy=False, name=''):
    '''One RepViT block: RepVGGDW token mixer -> optional squeeze-excite ->
    FFN channel mixer (its own, separate residual). stride=1, cin=cout=
    channels throughout - stage transitions go through `repvit_downsample`
    instead.

    `deploy=False` (default, training mode): the token mixer is a 3-way sum
    (3x3 depthwise+BN branch, 1x1 depthwise+bias branch, raw identity) with
    a trailing BN.
    `deploy=True`: the token mixer is a single fused 3x3 depthwise conv
    (`{name}_token_fused`) — the layer `reparameterize_repvit` writes fused
    weights into.
    '''
    if deploy:
        token_mixed = DepthwiseConv2D(3, padding='same', use_bias=True, name=f'{name}_token_fused')(x)
    else:
        dw3x3 = DepthwiseConv2D(3, padding='same', use_bias=False, name=f'{name}_token_dw3x3')(x)
        dw3x3 = BatchNormalization(epsilon=epsilon, name=f'{name}_token_dw3x3_bn')(dw3x3)

        dw1x1 = DepthwiseConv2D(1, padding='same', use_bias=True, name=f'{name}_token_dw1x1')(x)

        token_mixed = Add(name=f'{name}_token_add')([dw3x3, dw1x1, x])
        token_mixed = BatchNormalization(epsilon=epsilon, name=f'{name}_token_bn')(token_mixed)

    if se_ratio > 0:
        token_mixed = _squeeze_excite(token_mixed, channels, se_ratio, name=f'{name}_se')

    y = Conv2D(channels * 2, 1, padding='same', use_bias=False, name=f'{name}_ffn_pw1')(token_mixed)
    y = BatchNormalization(epsilon=epsilon, name=f'{name}_ffn_pw1_bn')(y)
    y = GELU(name=f'{name}_ffn_act')(y)
    y = Conv2D(channels, 1, padding='same', use_bias=False, name=f'{name}_ffn_pw2')(y)
    y = BatchNormalization(epsilon=epsilon, name=f'{name}_ffn_pw2_bn')(y)

    return Add(name=f'{name}_ffn_add')([y, token_mixed])


def repvit_downsample(x, channels_out, epsilon=1e-5, name=''):
    '''Stage-transition module: stride-2 depthwise + pointwise channel
    projection (no residual - spatial size and channel count both change),
    then one FFN block *with* a residual (valid once channels==channels_out
    on both sides). Precedes stages 2-4; stage 1 has no downsample module
    (the stem already brings the input to stage 1's resolution/4x). Has no
    multi-branch structure, so `reparameterize_repvit` leaves it untouched.
    '''
    # `padding='same'` + `strides=2` on an even feature-map size does NOT
    # match PyTorch's `padding=(kernel_size-1)//2` (symmetric) - Keras'
    # SAME rule pads *asymmetrically* here (0 top/left, 1 bottom/right for a
    # 3x3/stride-2 conv), a genuine, large numerical divergence confirmed by
    # a minimal isolated repro against the reference PyTorch conv (plan.md
    # §12.1) - explicit symmetric `ZeroPadding2D` + `padding='valid'`
    # matches PyTorch exactly (verified: max abs diff 0.0 on the same repro).
    d = ZeroPadding2D(1, name=f'{name}_dw3x3_pad')(x)
    d = DepthwiseConv2D(3, strides=2, padding='valid', use_bias=False, name=f'{name}_dw3x3')(d)
    d = BatchNormalization(epsilon=epsilon, name=f'{name}_dw3x3_bn')(d)
    d = Conv2D(channels_out, 1, padding='same', use_bias=False, name=f'{name}_pw')(d)
    d = BatchNormalization(epsilon=epsilon, name=f'{name}_pw_bn')(d)

    y = Conv2D(channels_out * 2, 1, padding='same', use_bias=False, name=f'{name}_ffn_pw1')(d)
    y = BatchNormalization(epsilon=epsilon, name=f'{name}_ffn_pw1_bn')(y)
    y = GELU(name=f'{name}_ffn_act')(y)
    y = Conv2D(channels_out, 1, padding='same', use_bias=False, name=f'{name}_ffn_pw2')(y)
    y = BatchNormalization(epsilon=epsilon, name=f'{name}_ffn_pw2_bn')(y)

    return Add(name=f'{name}_ffn_add')([y, d])


def repvit_stem(x, stem_width, epsilon=1e-5, name=''):
    '''Patchify stem: two stacked stride-2 3x3 convs -> 4x downsampling
    total, matching the reference implementation exactly (no exposed
    intermediate 2x tap - see module docstring).

    Explicit `ZeroPadding2D` + `padding='valid'`, not `padding='same'` - see
    the identical note in `repvit_downsample` for why (plan.md §12.1).
    '''
    x = ZeroPadding2D(1, name=f'{name}_conv1_pad')(x)
    x = Conv2D(stem_width // 2, 3, strides=2, padding='valid', use_bias=False, name=f'{name}_conv1')(x)
    x = BatchNormalization(epsilon=epsilon, name=f'{name}_conv1_bn')(x)
    x = GELU(name=f'{name}_conv1_act')(x)
    x = ZeroPadding2D(1, name=f'{name}_conv2_pad')(x)
    x = Conv2D(stem_width, 3, strides=2, padding='valid', use_bias=False, name=f'{name}_conv2')(x)
    x = BatchNormalization(epsilon=epsilon, name=f'{name}_conv2_bn')(x)
    return x


def repvit_stage_output_names(model_name, stage_blocks):
    '''The 4 pyramid-tap layer names for a given config - the last
    FFN-residual `Add` of each stage. Used both to register
    `BACKBONE_REGISTRY` entries in `_backbone_zoo.py` and internally by
    `reparameterize_repvit`.
    '''
    return tuple(
        f'{model_name}_stage{stage_idx + 1}_block{num_blocks - 1}_ffn_add'
        for stage_idx, num_blocks in enumerate(stage_blocks)
    )


def _repvit_network(input_tensor, stage_channels, stage_blocks, stem_width,
                    include_top=False, pooling=None, num_classes=1000, deploy=False, name='RepViT'):
    x = repvit_stem(input_tensor, stem_width, name=f'{name}_stem')

    stage_outputs = []
    for stage_idx, (channels, num_blocks) in enumerate(zip(stage_channels, stage_blocks)):
        stage_name = f'{name}_stage{stage_idx + 1}'

        if stage_idx > 0:
            x = repvit_downsample(x, channels, name=f'{stage_name}_downsample')

        for block_id in range(num_blocks):
            # SE fires on even-indexed blocks except the last block of the
            # stage, which always suppresses it - matches the reference
            # implementation's `cur_se_ratio` rule exactly (plan.md §12 research).
            se_ratio = 0.25 if (block_id % 2 == 0 and block_id != num_blocks - 1) else 0.0
            x = repvit_block(x, channels, se_ratio=se_ratio, deploy=deploy, name=f'{stage_name}_block{block_id}')

        stage_outputs.append(x)

    if include_top:
        x = GlobalAveragePooling2D(name=f'{name}_head_gap')(stage_outputs[-1])
        x = BatchNormalization(epsilon=1e-5, name=f'{name}_head_bn')(x)
        x = Dropout(0.0, name=f'{name}_head_dropout')(x)
        x = Dense(num_classes, activation='softmax', name=f'{name}_head_dense')(x)
        return x

    if pooling == 'avg':
        return GlobalAveragePooling2D(name=f'{name}_pool')(stage_outputs[-1])
    if pooling == 'max':
        return GlobalMaxPooling2D(name=f'{name}_pool')(stage_outputs[-1])

    return stage_outputs[-1]


def _build_repvit(stage_channels, stage_blocks, stem_width, model_name):
    def constructor(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs):
        if weights == 'imagenet':
            raise NotImplementedError(
                "weights='imagenet' auto-download isn't available yet for {} - that needs "
                "real, publicly-hosted weight files this package can fetch, which is a "
                "separate, not-yet-done piece of work (plan.md §12). Real, verified "
                "ImageNet-pretrained weights ARE available today, though: run "
                "`scripts/port_repvit_weights.py` (converts timm's pretrained checkpoint - "
                "verified against the PyTorch reference, max abs output difference "
                "<0.07 - plan.md §12.1) and pass the resulting `<model_name>.weights.h5` "
                "file path directly as `weights=`, e.g. "
                "weights='pretrained_weights/repvit/{}.weights.h5'.".format(model_name, model_name)
            )

        if input_tensor is None:
            if input_shape is None:
                raise ValueError('Either `input_tensor` or `input_shape` must be provided.')
            input_tensor = Input(shape=input_shape)

        output = _repvit_network(
            input_tensor, stage_channels, stage_blocks, stem_width,
            include_top=include_top, pooling=pooling, num_classes=classes, name=model_name,
        )
        model = Model(inputs=input_tensor, outputs=output, name=model_name)

        if weights not in (None, 'none'):
            # Any other string is treated as a local `.weights.h5` file path
            # (e.g. produced by `scripts/port_repvit_weights.py`) - built
            # with `include_top=False`, so `skip_mismatch=True` lets it load
            # cleanly into an `include_top=True` model too (the extra head
            # layers just keep their random init) without erroring.
            model.load_weights(weights, skip_mismatch=True)
        # Stashed for `reparameterize_repvit`, which needs to rebuild the
        # same network in deploy mode - avoids making the caller pass the
        # config a second time.
        model._repvit_config = dict(
            stage_channels=stage_channels, stage_blocks=stage_blocks, stem_width=stem_width, model_name=model_name,
        )
        return model

    constructor.__name__ = model_name
    return constructor


# Per-variant stage configs verified against keras_cv_attention_models'
# `repvit.py` (plan.md §12 research) - stem width, then (channels, num_blocks)
# per stage, stages 1-4 in order.
REPVIT_CONFIGS = {
    'RepViT-M0.9': dict(stage_channels=[48, 96, 192, 384], stage_blocks=[3, 3, 15, 2], stem_width=48, model_name='RepViT_M0_9'),
    'RepViT-M1.0': dict(stage_channels=[56, 112, 224, 448], stage_blocks=[3, 3, 15, 2], stem_width=56, model_name='RepViT_M1_0'),
    'RepViT-M1.1': dict(stage_channels=[64, 128, 256, 512], stage_blocks=[3, 3, 13, 2], stem_width=64, model_name='RepViT_M1_1'),
    'RepViT-M1.5': dict(stage_channels=[64, 128, 256, 512], stage_blocks=[5, 5, 25, 4], stem_width=64, model_name='RepViT_M1_5'),
    'RepViT-M2.3': dict(stage_channels=[80, 160, 320, 640], stage_blocks=[7, 7, 35, 2], stem_width=80, model_name='RepViT_M2_3'),
}

RepViT_M0_9 = _build_repvit(**REPVIT_CONFIGS['RepViT-M0.9'])
RepViT_M1_0 = _build_repvit(**REPVIT_CONFIGS['RepViT-M1.0'])
RepViT_M1_1 = _build_repvit(**REPVIT_CONFIGS['RepViT-M1.1'])
RepViT_M1_5 = _build_repvit(**REPVIT_CONFIGS['RepViT-M1.5'])
RepViT_M2_3 = _build_repvit(**REPVIT_CONFIGS['RepViT-M2.3'])


def reparameterize_repvit(model):
    '''Collapse every RepVGGDW token mixer in a trained RepViT model
    (3x3 depthwise+BN branch + 1x1 depthwise+bias branch + identity, see
    `repvit_block`) into a single fused 3x3 depthwise conv, using the shared
    primitives in `reparam_utils.py`. Returns a new `Model` (built in
    "deploy mode", `repvit_block(..., deploy=True)`) with identical
    inference-time behavior (within floating-point tolerance) and fewer ops
    per token mixer (one conv instead of two convs + a 3-way add + 2 BNs).

    Only the token-mixer is reparameterized - the FFN channel mixer has no
    multi-branch structure to fuse (a plain two-conv MLP with one residual),
    so its layers are copied verbatim.

    Input
    ----------
        model: a model returned by one of this module's constructors
               (`RepViT_M0_9`, etc.), used with `weights=None` and then
               trained - i.e. it must still have its `_repvit_config`
               attribute (set at construction time).
    '''
    if not hasattr(model, '_repvit_config'):
        raise ValueError(
            "`model` has no `_repvit_config` - reparameterize_repvit only supports models built "
            "by this module's own constructors (RepViT_M0_9, etc.)."
        )
    config = model._repvit_config

    deploy_input = Input(shape=model.input_shape[1:])
    deploy_output = _repvit_network(
        deploy_input, config['stage_channels'], config['stage_blocks'], config['stem_width'],
        deploy=True, name=config['model_name'],
    )
    deploy_model = Model(inputs=deploy_input, outputs=deploy_output, name=f"{config['model_name']}_fused")

    fused_layer_names = {
        layer.name for layer in deploy_model.layers if layer.name.endswith('_token_fused')
    }

    for deploy_layer in deploy_model.layers:
        if deploy_layer.name in fused_layer_names:
            prefix = deploy_layer.name[: -len('_token_fused')]
            dw3x3_layer = model.get_layer(f'{prefix}_token_dw3x3')
            dw3x3_bn_layer = model.get_layer(f'{prefix}_token_dw3x3_bn')
            dw1x1_layer = model.get_layer(f'{prefix}_token_dw1x1')
            # RepVGGDW has *two* BNs: one inside the 3x3 branch (fused into
            # that branch below), and a second, trailing one applied to the
            # branch-*sum* (`{prefix}_token_bn`, after the `Add`) - easy to
            # miss since it comes after the multi-branch structure, not
            # inside one of its branches.
            trailing_bn_layer = model.get_layer(f'{prefix}_token_bn')

            k3x3, b3x3 = fuse_conv_bn_layers(dw3x3_layer, dw3x3_bn_layer)
            k1x1, b1x1 = dw1x1_layer.get_weights()
            k1x1_padded = k1x1 if k1x1.shape[0] == 3 else np.pad(k1x1, ((1, 1), (1, 1), (0, 0), (0, 0)))

            channels = k3x3.shape[2]
            id_kernel = identity_kernel(channels, kernel_size=3, depthwise=True, dtype=k3x3.dtype)
            id_bias = np.zeros((channels,), dtype=k3x3.dtype)

            summed_kernel, summed_bias = sum_branches([
                (k3x3, b3x3),
                (k1x1_padded, b1x1),
                (id_kernel, id_bias),
            ])
            # The trailing BN is applied to the (already-linear) branch sum,
            # which is exactly "conv then BN" again - one more `fuse_conv_bn`
            # call folds it into the same fused kernel/bias.
            trailing_gamma, trailing_beta, trailing_mean, trailing_var = trailing_bn_layer.get_weights()
            fused_kernel, fused_bias = fuse_conv_bn(
                summed_kernel, summed_bias, trailing_gamma, trailing_beta, trailing_mean, trailing_var,
                epsilon=trailing_bn_layer.epsilon, channel_axis=-2,
            )
            deploy_layer.set_weights([fused_kernel, fused_bias])
        elif deploy_layer.weights:
            # Every other weighted layer (FFN convs/BNs, downsample module,
            # stem, SE, head) is architecturally identical between the
            # training-mode and deploy-mode graphs - same name, same shape -
            # so its weights carry over unchanged.
            deploy_layer.set_weights(model.get_layer(deploy_layer.name).get_weights())

    deploy_model._repvit_config = config
    return deploy_model
