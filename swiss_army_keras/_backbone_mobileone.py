'''MobileOne (Apple, 2022) — "An Improved One millisecond Mobile Backbone"
(https://arxiv.org/abs/2206.04040).

Unlike RepViT (§2.4) and FastViT (§2.3), no Keras port of MobileOne exists
anywhere (confirmed, plan.md §12) — architecture verified directly against
the official PyTorch source (github.com/apple/ml-mobileone, `mobileone.py`)
cross-checked against `timm`'s independent port (`timm/models/byobnet.py`),
per plan.md §7.

Stem + 4 stages, RepVGG-style. Each stage is `d` depthwise-separable
"pairs": a depthwise KxK conv (spatial mixing, stride-2 on each stage's
first pair only) followed by a pointwise 1x1 conv (channel mixing/
projection) — both reparameterizable `MobileOneBlock`s in their own right,
each with its own independent multi-branch structure. `num_blocks_per_stage
= [2, 8, 10, 1]` is identical across all 5 variants; what varies is a
4-element width-multiplier tuple (applied per stage, and to the stem with
an extra `min(64, ...)` cap the stages don't have — see `_stage_channels`),
`num_conv_branches` (only S0 uses 4; S1-S4 use 1), and squeeze-excite
(only S4, and only on the last 5-of-10 pairs of stage 3 plus all of
stage 4 — stages 1-2 never get SE, regardless of variant).

Native feature pyramid: unlike RepViT/FastViT's 4-tap limitation (their
stems reach 4x before any stage begins), MobileOne's stem is a *single*
stride-2 conv (2x), so a genuine 5-tap 2x/4x/8x/16x/32x pyramid exists —
`BACKBONE_REGISTRY`'s entries here register all 5, matching the
`tf.keras.applications`-style backbones' interface exactly, no `depth_max`
clamping needed.

Weights: no `weights='imagenet'` support in this pass — porting the
official PyTorch checkpoints (or `timm`'s copies) is separate follow-up
work (plan.md §12/§12.1: this is the family with no Keras reference at all,
so both the architecture *and* the weight conversion have no shortcut).
`weights=None` (random init) is fully supported.
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
    Multiply,
    ZeroPadding2D,
)
from tensorflow.keras.models import Model

from swiss_army_keras.reparam_utils import (
    fuse_conv_bn,
    fuse_conv_bn_layers,
    identity_kernel,
    pad_kernel_to,
    sum_branches,
)

BASE_CHANNELS = [64, 128, 256, 512]
NUM_BLOCKS_PER_STAGE = [2, 8, 10, 1]
SE_REDUCTION_RATIO = 0.0625


def _stage_channels(width_multipliers):
    '''`stem_channels = min(64, int(64 * wf[0]))` (capped - a detail the
    stages themselves don't share), `stage_channels[i] = int(BASE_CHANNELS[i]
    * wf[i])` (uncapped). Verified against the official per-variant tables
    (plan.md §7 research) - e.g. S1's `wf[0]=1.5` gives stem=64 (capped down
    from 96) but stage1=96 (not capped), a genuine, easy-to-miss asymmetry.
    '''
    stem_channels = min(64, int(64 * width_multipliers[0]))
    stage_channels = [int(base * wf) for base, wf in zip(BASE_CHANNELS, width_multipliers)]
    return stem_channels, stage_channels


def _make_conv(is_depthwise, out_channels, kernel_size, stride, use_bias, name):
    '''Returns a callable `f(x) -> tensor` for one conv branch, using
    **explicit, symmetric** zero-padding (`ZeroPadding2D(kernel_size // 2)`
    + `padding='valid'`) instead of Keras's `padding='same'`.

    This is not stylistic: at `stride > 1`, TF/Keras's `'same'` padding can
    be *asymmetric* (it computes `total_pad = max((out-1)*stride+k-in, 0)`
    and splits it as `pad_before = total_pad // 2`, dumping any odd
    remainder on the *after* side) — and critically, how much it falls short
    of pure symmetric centering depends on `k`. Fusing a KxK branch with a
    1x1 branch (`reparam_utils.pad_kernel_to`, which always centers
    symmetrically) silently assumes both branches sample the *same* input
    window - true at `stride=1` (where `'same'` padding for an odd kernel is
    always exactly symmetric, total_pad = k-1 is even) but **not
    guaranteed at `stride>1`**, where the two kernel sizes' asymmetry
    shortfalls can differ and desynchronize the two branches' sampling
    grids. Confirmed empirically while building this file (a stem-only
    isolation test showed the two branches sampling different windows,
    causing every output pixel to disagree - not a border effect) - the
    same class of bug was also found and fixed retroactively in FastViT's
    `rep_conv_block`/`rep_downsample` (§2.3) and audited (found safe,
    stride=1-only) in RepViT (§2.4). Explicit symmetric padding removes the
    dependence on `k` entirely, making every branch's sampling grid
    identical regardless of kernel size or stride.

    `Conv2D(filters, kernel_size, ...)` and `DepthwiseConv2D(kernel_size,
    ...)` take a *different* first positional argument - build each branch's
    call explicitly rather than sharing one call shape (this collided
    silently for FastViT §2.3 the first time; avoided here from the start).
    '''
    pad = kernel_size // 2

    def apply(x):
        if pad > 0:
            x = ZeroPadding2D(pad, name=f'{name}_pad')(x)
        if is_depthwise:
            return DepthwiseConv2D(kernel_size, strides=stride, padding='valid', use_bias=use_bias, name=name)(x)
        return Conv2D(out_channels, kernel_size, strides=stride, padding='valid', use_bias=use_bias, name=name)(x)

    return apply


def _squeeze_excite(x, channels, rd_ratio=SE_REDUCTION_RATIO, name=''):
    reduction = max(1, int(channels * rd_ratio))
    s = GlobalAveragePooling2D(keepdims=True, name=f'{name}_gap')(x)
    s = Conv2D(reduction, 1, padding='same', name=f'{name}_reduce')(s)
    s = Activation('relu', name=f'{name}_relu')(s)
    s = Conv2D(channels, 1, padding='same', name=f'{name}_expand')(s)
    s = Activation('sigmoid', name=f'{name}_sigmoid')(s)
    return Multiply(name=f'{name}_scale')([x, s])


def mobileone_block(x, out_channels, kernel_size, stride, depthwise, num_conv_branches=1,
                    use_se=False, deploy=False, epsilon=1e-3, name=''):
    '''One `MobileOneBlock`: `num_conv_branches` parallel KxK conv+BN
    branches (all identical shape, independently trained - the paper's
    over-parameterization trick), + one 1x1 conv+BN "scale" branch (only
    when `kernel_size > 1` - a 1x1 main branch would make this branch
    redundant with itself), + one identity+BN branch (only when
    `stride==1` and in_channels==out_channels), all summed, then optional
    squeeze-excite, then ReLU. At `deploy=True`, all of that collapses to a
    single fused conv (`{name}_fused`).
    '''
    in_channels = x.shape[-1]
    use_identity = (stride == 1 and in_channels == out_channels)
    use_scale = (kernel_size > 1)

    if deploy:
        mixed = _make_conv(depthwise, out_channels, kernel_size, stride, True, f'{name}_fused')(x)
    else:
        branches = [
            _make_conv(depthwise, out_channels, kernel_size, stride, False, f'{name}_conv{i}')(x)
            for i in range(num_conv_branches)
        ]
        branches = [BatchNormalization(epsilon=epsilon, name=f'{name}_conv{i}_bn')(b) for i, b in enumerate(branches)]

        if use_scale:
            scale = _make_conv(depthwise, out_channels, 1, stride, False, f'{name}_scale')(x)
            scale = BatchNormalization(epsilon=epsilon, name=f'{name}_scale_bn')(scale)
            branches.append(scale)

        if use_identity:
            branches.append(BatchNormalization(epsilon=epsilon, name=f'{name}_identity_bn')(x))

        mixed = Add(name=f'{name}_add')(branches) if len(branches) > 1 else branches[0]

    if use_se:
        mixed = _squeeze_excite(mixed, out_channels, name=f'{name}_se')

    return Activation('relu', name=f'{name}_act')(mixed)


def mobileone_stage_output_names(model_name):
    '''The 5 pyramid-tap layer names - stem, then each stage's last pair's
    pointwise block output.
    '''
    names = [f'{model_name}_stem_act']
    for stage_idx, num_blocks in enumerate(NUM_BLOCKS_PER_STAGE):
        names.append(f'{model_name}_stage{stage_idx + 1}_pair{num_blocks - 1}_pw_act')
    return tuple(names)


def _mobileone_network(input_tensor, width_multipliers, num_conv_branches, use_se,
                       include_top=False, pooling=None, num_classes=1000, deploy=False, name='MobileOne'):
    stem_channels, stage_channels = _stage_channels(width_multipliers)

    x = mobileone_block(input_tensor, stem_channels, kernel_size=3, stride=2, depthwise=False,
                        num_conv_branches=1, deploy=deploy, name=f'{name}_stem')

    for stage_idx, (channels, num_blocks) in enumerate(zip(stage_channels, NUM_BLOCKS_PER_STAGE)):
        stage_name = f'{name}_stage{stage_idx + 1}'
        # SE only ever applies to variants with use_se=True (only S4), and
        # only to stage 3's last 5-of-10 pairs / all of stage 4's 1 pair -
        # stages 1-2 never get it, regardless of variant (plan.md §7 research).
        if stage_idx == 2:
            se_start = num_blocks - (num_blocks // 2)
        elif stage_idx == 3:
            se_start = 0
        else:
            se_start = num_blocks  # i.e. never

        for pair_id in range(num_blocks):
            pair_se = use_se and pair_id >= se_start
            stride = 2 if pair_id == 0 else 1
            in_channels_before_pair = x.shape[-1]
            # A depthwise conv can never change channel count (output
            # channels = input channels, depth_multiplier=1) - its
            # `out_channels` must be its *actual input* channel count, not
            # the stage's target width. Only the pointwise conv right after
            # projects to the stage's target `channels`. Passing the target
            # width to the depthwise call here instead of the input width is
            # a real bug this file had until SE (S4 only) made it crash
            # visibly - see plan.md §13 Phase 2's MobileOne entry.
            x = mobileone_block(
                x, in_channels_before_pair, kernel_size=3, stride=stride, depthwise=True,
                num_conv_branches=num_conv_branches, use_se=pair_se, deploy=deploy,
                name=f'{stage_name}_pair{pair_id}_dw',
            )
            x = mobileone_block(
                x, channels, kernel_size=1, stride=1, depthwise=False,
                num_conv_branches=num_conv_branches, use_se=pair_se, deploy=deploy,
                name=f'{stage_name}_pair{pair_id}_pw',
            )

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


def _build_mobileone(width_multipliers, num_conv_branches, use_se, model_name):
    def constructor(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs):
        if weights not in (None, 'none'):
            raise NotImplementedError(
                "weights='{}' is not supported for {} yet - only weights=None (random "
                "initialization) is implemented so far. Porting the official PyTorch/timm "
                "checkpoints is tracked separately (plan.md §12/§12.1).".format(weights, model_name)
            )

        if input_tensor is None:
            if input_shape is None:
                raise ValueError('Either `input_tensor` or `input_shape` must be provided.')
            input_tensor = Input(shape=input_shape)

        output = _mobileone_network(
            input_tensor, width_multipliers, num_conv_branches, use_se,
            include_top=include_top, pooling=pooling, num_classes=classes, name=model_name,
        )
        model = Model(inputs=input_tensor, outputs=output, name=model_name)
        model._mobileone_config = dict(
            width_multipliers=width_multipliers, num_conv_branches=num_conv_branches,
            use_se=use_se, model_name=model_name,
        )
        return model

    constructor.__name__ = model_name
    return constructor


# Per-variant configs verified against apple/ml-mobileone + timm/byobnet.py
# (plan.md §12 research, cross-checked against two independent sources).
MOBILEONE_CONFIGS = {
    'MobileOne-S0': dict(width_multipliers=(0.75, 1.0, 1.0, 2.0), num_conv_branches=4, use_se=False, model_name='MobileOne_S0'),
    'MobileOne-S1': dict(width_multipliers=(1.5, 1.5, 2.0, 2.5), num_conv_branches=1, use_se=False, model_name='MobileOne_S1'),
    'MobileOne-S2': dict(width_multipliers=(1.5, 2.0, 2.5, 4.0), num_conv_branches=1, use_se=False, model_name='MobileOne_S2'),
    'MobileOne-S3': dict(width_multipliers=(2.0, 2.5, 3.0, 4.0), num_conv_branches=1, use_se=False, model_name='MobileOne_S3'),
    'MobileOne-S4': dict(width_multipliers=(3.0, 3.5, 3.5, 4.0), num_conv_branches=1, use_se=True, model_name='MobileOne_S4'),
}

MobileOne_S0 = _build_mobileone(**MOBILEONE_CONFIGS['MobileOne-S0'])
MobileOne_S1 = _build_mobileone(**MOBILEONE_CONFIGS['MobileOne-S1'])
MobileOne_S2 = _build_mobileone(**MOBILEONE_CONFIGS['MobileOne-S2'])
MobileOne_S3 = _build_mobileone(**MOBILEONE_CONFIGS['MobileOne-S3'])
MobileOne_S4 = _build_mobileone(**MOBILEONE_CONFIGS['MobileOne-S4'])


def mobileone_preprocess_input(x):
    '''[-1, 1] rescaling (MobileNet-family convention) - a placeholder until
    real pretrained weights are ported (plan.md §12/§12.1); see the
    identical note on `repvit_preprocess_input`/`fastvit_preprocess_input`.
    '''
    return (x / 127.5) - 1.0


def _fuse_mobileone_block(model, prefix, num_conv_branches, has_scale, has_identity, is_depthwise):
    '''Fuse one `MobileOneBlock`'s branches: `num_conv_branches` KxK+BN
    branches (summed together, being identical shape), one optional 1x1+BN
    "scale" branch (padded to KxK), one optional identity+BN branch.
    '''
    branch_pairs = []
    for i in range(num_conv_branches):
        conv_layer = model.get_layer(f'{prefix}_conv{i}')
        bn_layer = model.get_layer(f'{prefix}_conv{i}_bn')
        branch_pairs.append(fuse_conv_bn_layers(conv_layer, bn_layer))
    kernel_size = branch_pairs[0][0].shape[0]

    if has_scale:
        scale_conv = model.get_layer(f'{prefix}_scale')
        scale_bn = model.get_layer(f'{prefix}_scale_bn')
        k_scale, b_scale = fuse_conv_bn_layers(scale_conv, scale_bn)
        k_scale = pad_kernel_to(k_scale, kernel_size)
        branch_pairs.append((k_scale, b_scale))

    if has_identity:
        channels = branch_pairs[0][0].shape[2] if is_depthwise else branch_pairs[0][0].shape[3]
        id_kernel = identity_kernel(channels, kernel_size, depthwise=is_depthwise, dtype=branch_pairs[0][0].dtype)
        identity_bn = model.get_layer(f'{prefix}_identity_bn')
        gamma, beta, mean, var = identity_bn.get_weights()
        channel_axis = -2 if is_depthwise else -1
        k_id, b_id = fuse_conv_bn(id_kernel, None, gamma, beta, mean, var, epsilon=identity_bn.epsilon, channel_axis=channel_axis)
        branch_pairs.append((k_id, b_id))

    return sum_branches(branch_pairs)


def reparameterize_mobileone(model):
    '''Collapse every `MobileOneBlock` (stem, and every depthwise/pointwise
    pair in every stage) in a trained MobileOne model into single fused
    convs, using `reparam_utils.py`. Returns a new `Model` built in deploy
    mode with identical inference-time behavior (within floating-point
    tolerance). SE blocks and the trailing ReLU have no multi-branch
    structure and are copied verbatim.

    Input
    ----------
        model: a model returned by one of this module's constructors
               (`MobileOne_S0`, etc.), used with `weights=None` and then
               trained - i.e. it must still have its `_mobileone_config`
               attribute (set at construction time).
    '''
    if not hasattr(model, '_mobileone_config'):
        raise ValueError(
            "`model` has no `_mobileone_config` - reparameterize_mobileone only supports models "
            "built by this module's own constructors (MobileOne_S0, etc.)."
        )
    config = model._mobileone_config

    deploy_input = Input(shape=model.input_shape[1:])
    deploy_output = _mobileone_network(
        deploy_input, config['width_multipliers'], config['num_conv_branches'], config['use_se'],
        deploy=True, name=config['model_name'],
    )
    deploy_model = Model(inputs=deploy_input, outputs=deploy_output, name=f"{config['model_name']}_fused")

    stem_channels, stage_channels = _stage_channels(config['width_multipliers'])
    model_name = config['model_name']

    # Every block's fusion metadata: prefix -> (num_conv_branches, has_scale,
    # has_identity, is_depthwise). `has_scale` mirrors `mobileone_block`'s own
    # `use_scale = kernel_size > 1` rule: the stem and every depthwise block
    # use kernel_size=3 (has_scale=True); every pointwise block uses
    # kernel_size=1 (has_scale=False, matching `mobileone_block` exactly -
    # a 1x1 main branch makes a separate 1x1 "scale" branch redundant).
    block_specs = {}
    # stem: num_conv_branches=1 always (even for S0, per plan.md §7 research),
    # kernel_size=3 (has_scale=True), stride=2 (has_identity=False), regular conv.
    block_specs[f'{model_name}_stem'] = (1, True, False, False)

    prev_channels = stem_channels
    for stage_idx, (channels, num_blocks) in enumerate(zip(stage_channels, NUM_BLOCKS_PER_STAGE)):
        stage_name = f'{model_name}_stage{stage_idx + 1}'
        for pair_id in range(num_blocks):
            stride = 2 if pair_id == 0 else 1
            dw_prefix = f'{stage_name}_pair{pair_id}_dw'
            pw_prefix = f'{stage_name}_pair{pair_id}_pw'
            # Depthwise never changes channel count by construction, so its
            # identity condition reduces to just `stride == 1`.
            block_specs[dw_prefix] = (config['num_conv_branches'], True, stride == 1, True)
            # Pointwise is always stride=1; identity iff it doesn't project
            # to a different channel count (true for every pair after the
            # stage's first, false for the first pair of any stage whose
            # width differs from the previous stage's).
            block_specs[pw_prefix] = (config['num_conv_branches'], False, prev_channels == channels, False)
            prev_channels = channels

    for deploy_layer in deploy_model.layers:
        name = deploy_layer.name
        if not name.endswith('_fused'):
            if deploy_layer.weights:
                deploy_layer.set_weights(model.get_layer(name).get_weights())
            continue

        prefix = name[: -len('_fused')]
        num_branches, has_scale, has_identity, is_depthwise = block_specs[prefix]
        fused_kernel, fused_bias = _fuse_mobileone_block(model, prefix, num_branches, has_scale, has_identity, is_depthwise)
        deploy_layer.set_weights([fused_kernel, fused_bias])

    deploy_model._mobileone_config = config
    return deploy_model
