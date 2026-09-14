
'''DDRNet with DAPPM (Hong et al., 2021, "Deep Dual-resolution Networks for
Real-time and Accurate Semantic Segmentation of Road Scenes",
arXiv:2101.06085).

Architecture verified directly against the official reference source
(`ydhongHIT/DDRNet`, `segmentation/DDRNet_23_slim.py`, cross-checked against
`DDRNet_23.py` in the same repo — same rigor as the other decoders/backbones
in this plan, per plan.md §7). **Unlike `lr_aspp`/`pp_liteseg`, DDRNet is not
a decoder head consumed via `backbone_zoo`** — it's a complete, self-contained
dual-resolution network from stem to head (no `backbone=` parameter), the one
genuinely different shape among the three decoders added under plan.md §3.

- **Stem**: two stride-2 3x3 Conv+BN+ReLU in sequence (stride 4 total, no
  max-pool), output channels `planes` (32 for the "slim" config, 64 for the
  full-width one — both configs share the same block-count schedule, they
  only differ in width, confirmed from the reference).
- **Main (low-resolution) branch**: `layer1` (stride4, stays) -> `layer2`
  (stride8) -> `layer3` (stride16) -> `layer4` (stride32) -> `layer5`
  (Bottleneck, stride64) — `layer1`-`layer4` built from `BasicBlock`
  (expansion 1), `layer5` from `Bottleneck` (**expansion 2**, not the usual
  ResNet 4 — confirmed from the reference, not assumed by analogy).
- **High-resolution branch**: splits off *after* `layer2` (i.e. operates on
  the stride-8 feature) and never downsamples again — `layer3_`/`layer4_`
  (2 `BasicBlock`s each, both stay at stride8, channel width
  `highres_planes = planes * 2`) then `layer5_` (one `Bottleneck`, stays
  stride8, output `highres_planes * 2 = planes * 4`).
- **Bilateral fusion** happens at exactly two points (after stage 3 and
  stage 4 of both branches): the main branch gets a downsampled contribution
  from the high-res branch (`down3`: one stride-2 3x3 conv, stride8->16, one
  conv suffices for a factor-2 gap; `down4`: a *cascade of two* stride-2 3x3
  convs, stride8->32, since that gap is factor 4) added in; the high-res
  branch gets an upsampled, channel-compressed contribution from the main
  branch (`compression3`/`compression4`: 1x1 conv + BN, then bilinear
  upsample) added in. Both `down*`/`compression*` paths are BN-terminated
  with **no trailing ReLU** — the ReLU is applied to each path's *input*
  (`relu(main_branch_output)`/`relu(highres_branch_output)`) at the point of
  consumption instead, an asymmetric pre/post-activation detail confirmed
  directly from the source, not assumed.
- **DAPPM** (Deep Aggregation PPM) sits only once, on the main branch's
  final (`layer5`, stride64) output. Its 4 pooling branches use *specific*
  kernel/stride/padding triples (`(5,2,2)`, `(9,4,4)`, `(17,8,8)`, replicated
  here via explicit `ZeroPadding2D` + `AveragePooling2D(padding='valid')`,
  since Keras's `AveragePooling2D` has no numeric-padding argument the way
  PyTorch's `AvgPool2d` does) plus a global-average-pool branch and a
  passthrough (no-pool) branch — 5 branches total, every one
  pre-activation-style (`BN -> ReLU -> Conv`). **Combined by a cascade, not
  a single concat/sum**: each pooled+conv branch is bilinear-upsampled back
  to the module's native resolution and *added to the previous scale's
  already-processed output*, then run through its own 3x3 conv, before the
  next scale folds in — confirmed directly from the source's
  `x_list.append(processN(upsample(scaleN(x)) + x_list[N-1]))` accumulation,
  not a simplifying guess. The 5 cascaded outputs are concatenated and
  projected down by one final 1x1 conv, then added to a plain 1x1-conv
  shortcut of DAPPM's raw input.
- DAPPM's output (stride64) is bilinear-upsampled (x8) back to stride8 and
  added to the high-resolution branch's `layer5_` output before the final
  head.
- **Auxiliary/deep-supervision head** (optional, `augment=True` by default,
  per plan.md §3.3's requirement for this specific decoder — the other two
  decoders in this plan don't need one): takes the high-res branch's
  intermediate feature *right after the first bilateral fusion* (stride8,
  `highres_planes` channels) — i.e. before `layer4`/`layer4_` even run, not
  any later stage — through the same `segmenthead` structure as the main
  head (`BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1`).
- **One confirmed gap in the reference worth flagging explicitly**: neither
  the main head nor the aux head is upsampled to the *input's* full
  resolution inside the official `DualResNet`/`segmenthead` classes as
  published — both return stride-8 logits, with the final upsample
  apparently handled by that repo's separate training/eval scripts, not
  shown in the model file itself. This implementation adds that upsample
  explicitly (bilinear, to `input_tensor`'s spatial size) for both heads,
  matching the same "decoder head upsamples internally" convention already
  established by `lr_aspp`/`pp_liteseg`/`deeplab_v3_plus` in this codebase —
  an intentional addition on top of the reference, not something silently
  assumed to already be there.

Uses `tfmot.quantization.keras.quantize_annotate_layer` wrapping for QAT
parity, matching every other decoder in this plan; `Add` (bilateral fusion,
DAPPM's cascade and final shortcut-add) is left unannotated, same reasoning
and precedent as `lr_aspp`'s `Multiply`/`pp_liteseg`'s `Add` finding (§3.1/§3.2)
— `tfmot`'s default scheme has no quantization config for Keras merge layers.
'''

from __future__ import absolute_import

from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Input,
    UpSampling2D,
    ZeroPadding2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal

from swiss_army_keras.layer_utils import dummy_preprocessing, qat_annotate_layer

import tensorflow_model_optimization as tfmot


def _bn_relu(x, name=''):
    '''A "pre-activation" `BatchNormalization -> ReLU` pair whose input is
    *not* a `Conv2D`/`DepthwiseConv2D` (DDRNet's DAPPM and its per-branch
    `BN -> ReLU -> Conv` ordering are the only pre-activation-style blocks
    in this codebase — every other decoder here uses the opposite,
    post-activation `Conv -> BN -> Act` order). `tfmot`'s default QAT
    scheme's `BatchNormalization` quantize config only supports a BN that
    directly follows a `Conv2D`/`DepthwiseConv2D` (for BN-folding) — wrapping
    *this* standalone usage in `quantize_annotate_layer` made `quantize_apply`
    raise `RuntimeError: ...BatchNormalization is not supported`, the same
    class of "not supported, leave it unannotated" finding already made for
    `Multiply`/`Concatenate`/`Add` in the other two decoders (§3.1/§3.2).
    Left unannotated here for the same reason.
    '''
    x = BatchNormalization(name=f'{name}_bn')(x)
    return Activation('relu', name=f'{name}_relu')(x)


def _conv_bn(x, out_channels, kernel_size, stride=1, use_bias=False, name=''):
    x = qat_annotate_layer(Conv2D(
        out_channels, kernel_size, strides=stride, padding='same', use_bias=use_bias,
        kernel_initializer=HeNormal(), name=f'{name}_conv',
    ))(x)
    return qat_annotate_layer(BatchNormalization(name=f'{name}_bn'))(x)


def _conv_bn_relu(x, out_channels, kernel_size, stride=1, name=''):
    x = _conv_bn(x, out_channels, kernel_size, stride=stride, name=name)
    return qat_annotate_layer(Activation('relu', name=f'{name}_relu'))(x)


def basic_block(x, out_channels, stride=1, no_relu=False, name=''):
    '''Standard ResNet `BasicBlock` (expansion 1): two 3x3 conv+BN, a
    projection shortcut whenever the stride or channel count changes,
    residual add, then a *conditional* trailing ReLU — DDRNet's own
    `_make_layer` always builds this with `no_relu=True` on a stage's last
    block (see module docstring); the caller applies ReLU explicitly at the
    point that output is next consumed instead.
    '''
    in_channels = x.shape[-1]
    shortcut = x
    if stride != 1 or in_channels != out_channels:
        shortcut = _conv_bn(x, out_channels, 1, stride=stride, name=f'{name}_shortcut')

    h = _conv_bn_relu(x, out_channels, 3, stride=stride, name=f'{name}_conv1')
    h = _conv_bn(h, out_channels, 3, stride=1, name=f'{name}_conv2')
    h = Add(name=f'{name}_add')([h, shortcut])
    if not no_relu:
        h = qat_annotate_layer(Activation('relu', name=f'{name}_relu'))(h)
    return h


def bottleneck_block(x, mid_channels, stride=1, no_relu=True, expansion=2, name=''):
    '''DDRNet's `Bottleneck` — **expansion 2**, not ResNet's usual 4
    (confirmed from the reference, used for `layer5`/`layer5_` only).
    '''
    out_channels = mid_channels * expansion
    in_channels = x.shape[-1]
    shortcut = x
    if stride != 1 or in_channels != out_channels:
        shortcut = _conv_bn(x, out_channels, 1, stride=stride, name=f'{name}_shortcut')

    h = _conv_bn_relu(x, mid_channels, 1, stride=1, name=f'{name}_conv1')
    h = _conv_bn_relu(h, mid_channels, 3, stride=stride, name=f'{name}_conv2')
    h = _conv_bn(h, out_channels, 1, stride=1, name=f'{name}_conv3')
    h = Add(name=f'{name}_add')([h, shortcut])
    if not no_relu:
        h = qat_annotate_layer(Activation('relu', name=f'{name}_relu'))(h)
    return h


def _make_stage(x, block_fn, out_channels, num_blocks, stride=1, name=''):
    '''`num_blocks` residual blocks; the first carries `stride`/the channel
    change, the rest are stride-1. The *last* block in the stage is built
    with `no_relu=True` (see `basic_block`/`bottleneck_block`) — the
    returned tensor has no trailing activation, by design.
    '''
    x = block_fn(x, out_channels, stride=stride, no_relu=(num_blocks == 1), name=f'{name}_block0')
    for i in range(1, num_blocks):
        x = block_fn(x, out_channels, stride=1, no_relu=(i == num_blocks - 1), name=f'{name}_block{i}')
    return x


def _avg_pool_branch(x, pool_size, stride, pad, name=''):
    if pad > 0:
        x = ZeroPadding2D(padding=pad, name=f'{name}_pad')(x)
    return qat_annotate_layer(AveragePooling2D(
        pool_size=pool_size, strides=stride, padding='valid', name=f'{name}_pool',
    ))(x)


def dappm(x, branch_channels, out_channels, name='dappm'):
    '''Deep Aggregation PPM: 5 branches (passthrough + 3 average-pool scales
    `(kernel, stride, pad)` = `(5, 2, 2)`, `(9, 4, 4)`, `(17, 8, 8)`, plus a
    global-average-pool branch), each pre-activation (`BN -> ReLU -> Conv1x1`),
    combined by a **cascade** (each upsampled branch added to the *previous*
    cascade output, then a 3x3 conv), not a plain concat/sum — see module
    docstring for why this distinction matters and where it was verified.
    '''
    height, width = int(x.shape[1]), int(x.shape[2])

    scale0 = _bn_relu(x, name=f'{name}_scale0')
    scale0 = _conv_bn(scale0, branch_channels, 1, name=f'{name}_scale0_conv')

    stage_specs = [(5, 2, 2), (9, 4, 4), (17, 8, 8)]
    cascade = [scale0]
    for i, (kernel, stride, pad) in enumerate(stage_specs, start=1):
        pooled = _avg_pool_branch(x, kernel, stride, pad, name=f'{name}_scale{i}')
        pooled = _bn_relu(pooled, name=f'{name}_scale{i}')
        pooled = _conv_bn(pooled, branch_channels, 1, name=f'{name}_scale{i}_conv')
        pooled = qat_annotate_layer(UpSampling2D(
            size=(height // pooled.shape[1], width // pooled.shape[2]),
            interpolation='bilinear', name=f'{name}_scale{i}_up',
        ))(pooled)
        merged = Add(name=f'{name}_scale{i}_merge')([pooled, cascade[-1]])
        merged = _bn_relu(merged, name=f'{name}_process{i}')
        merged = _conv_bn(merged, branch_channels, 3, name=f'{name}_process{i}_conv')
        cascade.append(merged)

    global_pooled = qat_annotate_layer(AveragePooling2D(
        pool_size=(height, width), name=f'{name}_scale4_pool',
    ))(x)
    global_pooled = _bn_relu(global_pooled, name=f'{name}_scale4')
    global_pooled = _conv_bn(global_pooled, branch_channels, 1, name=f'{name}_scale4_conv')
    global_pooled = qat_annotate_layer(UpSampling2D(
        size=(height, width), interpolation='bilinear', name=f'{name}_scale4_up',
    ))(global_pooled)
    merged = Add(name=f'{name}_scale4_merge')([global_pooled, cascade[-1]])
    merged = _bn_relu(merged, name=f'{name}_process4')
    merged = _conv_bn(merged, branch_channels, 3, name=f'{name}_process4_conv')
    cascade.append(merged)

    concatenated = Concatenate(axis=-1, name=f'{name}_concat')(cascade)
    compressed = _bn_relu(concatenated, name=f'{name}_compression')
    compressed = _conv_bn(compressed, out_channels, 1, name=f'{name}_compression_conv')

    shortcut = _bn_relu(x, name=f'{name}_shortcut')
    shortcut = _conv_bn(shortcut, out_channels, 1, name=f'{name}_shortcut_conv')

    return Add(name=f'{name}_out')([compressed, shortcut])


def segmenthead(x, mid_channels, n_labels, name='segmenthead'):
    x = _bn_relu(x, name=f'{name}_pre')
    x = _conv_bn_relu(x, mid_channels, 3, name=f'{name}_conv1')
    # A second pre-activation `BN -> ReLU` (preceding the classifier conv,
    # not following one) - same unannotated treatment as `_bn_relu` and for
    # the same reason (see its docstring).
    x = _bn_relu(x, name=f'{name}_pre2')
    return qat_annotate_layer(
        Conv2D(n_labels, 1, padding='same', use_bias=True, name=f'{name}_classifier'))(x)


def ddrnet(input_tensor, n_labels, planes=32, layers=(2, 2, 2, 2), spp_planes=128,
           head_planes=64, augment=True, name='ddrnet'):
    '''DDRNet with DAPPM (Hong et al., 2021) — a complete, self-contained
    dual-resolution segmentation network (no `backbone=` parameter; see
    module docstring for why this decoder doesn't follow the
    `backbone_zoo`-based pattern of `lr_aspp`/`pp_liteseg`).

    Input
    ----------
        input_tensor: the input tensor, e.g., `keras.layers.Input((None, None, 3))`.
        n_labels: number of segmentation classes.
        planes: base channel width — 32 reproduces the official "DDRNet-23-slim"
                config, 64 reproduces the full-width "DDRNet-23" (both share
                the same `layers` block-count schedule per the reference).
        layers: 4-tuple, `BasicBlock` count for `layer1`-`layer4` respectively.
        spp_planes: DAPPM's per-branch and compressed-output channel count.
        head_planes: `segmenthead`'s intermediate channel count (shared by
                     both the main and, if `augment`, the auxiliary head).
        augment: if `True` (default, matching the paper's training-time
                 config), also returns the auxiliary deep-supervision head
                 as a second model output.
        name: prefix of the created Keras model and its layers.

    Output
    ----------
        A `tf.keras.Model` with a `.preprocessing` attribute (`dummy_preprocessing`
        — this network has no separate registered backbone to inherit one
        from). Output is `[main_output]`, or `[main_output, aux_output]` if
        `augment=True`, each an `(H, W, n_labels)` logit map at the input's
        spatial resolution.
    '''

    highres_planes = planes * 2

    x = _conv_bn_relu(input_tensor, planes, 3, stride=2, name=f'{name}_stem1')
    x = _conv_bn_relu(x, planes, 3, stride=2, name=f'{name}_stem2')

    x = _make_stage(x, basic_block, planes, layers[0], stride=1, name=f'{name}_layer1')
    x = qat_annotate_layer(Activation('relu', name=f'{name}_layer1_relu'))(x)

    x = _make_stage(x, basic_block, planes * 2, layers[1], stride=2, name=f'{name}_layer2')

    x_ = _make_stage(
        qat_annotate_layer(Activation('relu', name=f'{name}_layer2_relu_hb'))(x),
        basic_block, highres_planes, 2, stride=1, name=f'{name}_layer3_',
    )
    x = _make_stage(
        qat_annotate_layer(Activation('relu', name=f'{name}_layer2_relu_mb'))(x),
        basic_block, planes * 4, layers[2], stride=2, name=f'{name}_layer3',
    )

    down3 = _conv_bn(
        qat_annotate_layer(Activation('relu', name=f'{name}_hb3_relu'))(x_),
        planes * 4, 3, stride=2, name=f'{name}_down3',
    )
    compression3 = _conv_bn(
        qat_annotate_layer(Activation('relu', name=f'{name}_mb3_relu'))(x),
        highres_planes, 1, name=f'{name}_compression3',
    )
    compression3 = qat_annotate_layer(UpSampling2D(
        size=(x_.shape[1] // compression3.shape[1], x_.shape[2] // compression3.shape[2]),
        interpolation='bilinear', name=f'{name}_compression3_up',
    ))(compression3)

    x_next = Add(name=f'{name}_fuse3_main')([x, down3])
    x_next_ = Add(name=f'{name}_fuse3_hb')([x_, compression3])
    x, x_ = x_next, x_next_

    x_ = _make_stage(
        qat_annotate_layer(Activation('relu', name=f'{name}_layer3_relu_hb'))(x_),
        basic_block, highres_planes, 2, stride=1, name=f'{name}_layer4_',
    )
    aux_feature = x_

    x = _make_stage(
        qat_annotate_layer(Activation('relu', name=f'{name}_layer3_relu_mb'))(x),
        basic_block, planes * 8, layers[3], stride=2, name=f'{name}_layer4',
    )

    down4 = _conv_bn_relu(
        qat_annotate_layer(Activation('relu', name=f'{name}_hb4_relu'))(x_),
        planes * 4, 3, stride=2, name=f'{name}_down4_1',
    )
    down4 = _conv_bn(down4, planes * 8, 3, stride=2, name=f'{name}_down4_2')
    compression4 = _conv_bn(
        qat_annotate_layer(Activation('relu', name=f'{name}_mb4_relu'))(x),
        highres_planes, 1, name=f'{name}_compression4',
    )
    compression4 = qat_annotate_layer(UpSampling2D(
        size=(x_.shape[1] // compression4.shape[1], x_.shape[2] // compression4.shape[2]),
        interpolation='bilinear', name=f'{name}_compression4_up',
    ))(compression4)

    x_next = Add(name=f'{name}_fuse4_main')([x, down4])
    x_next_ = Add(name=f'{name}_fuse4_hb')([x_, compression4])
    x, x_ = x_next, x_next_

    x_ = bottleneck_block(
        qat_annotate_layer(Activation('relu', name=f'{name}_layer4_relu_hb'))(x_),
        highres_planes, stride=1, no_relu=True, name=f'{name}_layer5_',
    )
    x = _make_stage(
        qat_annotate_layer(Activation('relu', name=f'{name}_layer4_relu_mb'))(x),
        lambda t, out_ch, stride, no_relu, name: bottleneck_block(
            t, out_ch // 2, stride=stride, no_relu=no_relu, name=name),
        planes * 16, 1, stride=2, name=f'{name}_layer5',
    )

    x = dappm(
        qat_annotate_layer(Activation('relu', name=f'{name}_layer5_relu'))(x),
        spp_planes, planes * 4, name=f'{name}_dappm',
    )
    x = qat_annotate_layer(UpSampling2D(
        size=(x_.shape[1] // x.shape[1], x_.shape[2] // x.shape[2]),
        interpolation='bilinear', name=f'{name}_dappm_up',
    ))(x)

    x_final = Add(name=f'{name}_fuse5_hb')([x_, x])

    main_out = segmenthead(x_final, head_planes, n_labels, name=f'{name}_final_head')
    main_out = qat_annotate_layer(UpSampling2D(
        size=(input_tensor.shape[1] // main_out.shape[1], input_tensor.shape[2] // main_out.shape[2]),
        interpolation='bilinear', name=f'{name}_final_upsample',
    ))(main_out)

    outputs = [main_out]
    if augment:
        aux_out = segmenthead(aux_feature, head_planes, n_labels, name=f'{name}_aux_head')
        aux_out = qat_annotate_layer(UpSampling2D(
            size=(input_tensor.shape[1] // aux_out.shape[1], input_tensor.shape[2] // aux_out.shape[2]),
            interpolation='bilinear', name=f'{name}_aux_upsample',
        ))(aux_out)
        outputs.append(aux_out)

    m = Model([input_tensor, ], outputs)

    with tfmot.quantization.keras.quantize_scope():
        m = tfmot.quantization.keras.quantize_apply(m)

    m.preprocessing = dummy_preprocessing

    return m
