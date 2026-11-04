'''PIDNet (Xu et al., 2023, "PIDNet: A Real-time Semantic Segmentation
Network Inspired by PID Controllers", CVPR 2023, arXiv:2206.02066).

Architecture verified directly against the official reference source
(`XuJiacong/PIDNet`, `models/pidnet.py` + `models/model_utils.py`, fetched
and read verbatim, not summarized/guessed) — same rigor as every other
backbone/decoder in this plan (plan.md §7). **Like `ddrnet` (§3.3), PIDNet
is not a decoder head consumed via `backbone_zoo`** — it's a complete,
self-contained three-branch network from stem to head (no `backbone=`
parameter). Every `BasicBlock`/`Bottleneck` (**expansion 2**, confirmed
from the reference, same non-standard convention DDRNet already uses) and
the small conv+BN helper functions are the *same* components DDRNet already
implements and this file imports directly (`basic_block`, `bottleneck_block`,
`_make_stage`, `_conv_bn`, `_conv_bn_relu`, `_bn_relu`, `_avg_pool_branch`,
`dappm`) rather than duplicating them — genuinely the same building blocks,
not just superficially similar, given PIDNet's own P/I branches are built
from literally the same ResNet `BasicBlock`/`Bottleneck` shapes.

- **Stem**: identical to DDRNet's — two stride-2 3x3 Conv+BN+ReLU (stride 4
  total), `planes` channels.
- **Shared trunk**: `layer1` (`m` `BasicBlock`s, stride4, stays) -> `layer2`
  (`m` `BasicBlock`s, stride8) — both stages end without a trailing ReLU
  (`_make_stage`'s own convention), with an explicit `Activation('relu')`
  applied by the caller at each point that output is next consumed, exactly
  matching the reference's own `x = self.relu(self.layer2(self.relu(x)))`
  and DDRNet's identical pattern.
- **Three branches split after `layer2`** (all read the same post-`layer2`-
  ReLU tensor):
  - **P (Proportional, detail)**: `layer3_`/`layer4_` (`m` `BasicBlock`s
    each, `highres_planes = planes*2`, never downsamples past stride8),
    `layer5_` (a single `Bottleneck`, mid-channels `planes*2`, output
    `planes*4` via expansion=2).
  - **I (Integral, context)**: `layer3` (`n` `BasicBlock`s, stride16,
    `planes*4`) -> `layer4` (`n` `BasicBlock`s, stride32, `planes*8`) ->
    `layer5` (**two** stacked `Bottleneck`s, mid-channels `planes*8`,
    stride64, output `planes*16`) -> a pyramid-pooling module (PAPPM or
    DAPPM, see below) -> upsampled back to stride8.
  - **D (Derivative, boundary)**: `layer3_d`/`layer4_d`/`layer5_d` — channel
    widths and block counts/types **depend on `m`** (`m == 2`: `layer3_d` a
    single `BasicBlock` `planes*2 -> planes`, `layer4_d` a single
    `Bottleneck` `planes -> planes*2`; any other `m`: both `layer3_d` and
    `layer4_d` are single `BasicBlock`s staying at `planes*2`; `layer5_d`
    is always a single `Bottleneck` `planes*2 -> planes*4`, regardless of
    `m`) — fed not from the stem/trunk directly but via `diff3`/`diff4`
    taps pulling I-branch features in (plain conv+BN, no activation,
    upsampled to a *fixed* `input_height//8, input_width//8` target
    computed once from the original input, then **added** to the D
    branch's own running tensor — not fused via any attention mechanism,
    unlike P's PagFM).
  - **`m` genuinely does double duty**, confirmed from the reference, not
    a simplification made here: it's both the shared trunk's/P branch's
    per-stage `BasicBlock` repeat count *and* the discriminator for which
    pooling module (`PAPPM` when `m == 2`, else `DAPPM`) and which final
    fusion module (`Light_Bag` when `m == 2`, else `Bag`) the whole network
    uses — `n` only ever controls `layer3`/`layer4`'s (I-branch-only)
    block counts.
- **PagFM** (Pixel-attention-Guided Fusion, injects I into P at two points,
  after stage 3 and stage 4): projects both P (`x`) and I (`y`, already
  channel-compressed via `compression3`/`compression4`) to `mid_channels`
  (`planes`) via 1x1 conv+BN each, `sim_map = sigmoid(sum(f_x(x) *
  upsample(f_y(y)), axis=channel, keepdims=True))` — a **single-channel**
  spatial attention map (`with_channel=False` is the reference's actual
  configuration here, confirmed from the constructor call site
  `PagFM(planes*2, planes)`, no `with_channel=True` kwarg), then
  `output = x * (1 - sim_map) + upsample(y) * sim_map` — reuses
  `pp_liteseg`'s own `AttentionBlend` layer for this last step (`§3.2`'s
  `x*atten + y*(1-atten)` is the same blend formula, arguments reordered).
  The channel-sum-then-sigmoid step is a small dedicated `ChannelSumSigmoid`
  Layer subclass (not raw functional `tf.reduce_sum`/`Lambda`), for the same
  QAT-model-cloning-safety reason `AttentionBlend`/`ChannelMeanMaxConcat`
  (`_model_pp_liteseg.py`) and `MultiQueryAttention`
  (`_backbone_mobilenetv4.py`) already are proper `Layer` subclasses instead
  of raw ops.
- **Bag / Light_Bag** (the final 3-way P/I/D merge — **not** the "bag of
  tricks" kind; "Boundary-Attention-**Guided**"): `edge_att = sigmoid(D)`
  gates the *full-width* (`planes*4`-channel) P/I tensors directly — **not**
  a 1-channel spatial map here (that's only true of `seghead_d`'s separate,
  auxiliary-loss-only 1-channel projection of D, an easy point to conflate
  with this gate, which uses D's raw `planes*4`-channel tensor). `Bag`:
  `conv3x3(bn_relu(edge_att*P + (1-edge_att)*I))` — one pre-activation
  conv, no trailing BN (confirmed from source: `nn.Sequential(BatchNorm,
  ReLU, Conv2d(bias=False))`, nothing after). `Light_Bag` (the `m == 2`
  variant): two parallel 1x1 conv+BN paths,
  `conv_p(P + (1-edge_att)*I) + conv_i(I + edge_att*P)` — asymmetric
  (`P`/`I` each appear un-gated in one path and gated in the other), not
  expressible via `AttentionBlend`'s single blend formula, so built
  directly from `Multiply`/`Add`/a small `OneMinus` layer.
- **PAPPM vs DAPPM**: `dappm` is imported and reused as-is from
  `_model_ddrnet.py` for the `m != 2` ("L") case — genuinely the same
  5-branch cascade-pooling module PIDNet's own `DAPPM` class is (confirmed
  identical branch kernel/stride/pad triples, cascade-accumulation pattern,
  and final concat+compression+shortcut, from reading both source files
  directly). **Known, deliberately-not-fixed discrepancy inherited from
  the existing `dappm`**: DDRNet's `dappm` appends one extra
  `BatchNormalization` after each of its 5 branches' `Conv1x1` (via
  `_conv_bn`) that the verified PyTorch reference does not have (each
  branch there ends on a bare `Conv2d(bias=False)`, no trailing BN) — found
  while cross-referencing PIDNet's byte-identical `DAPPM` class against
  this file's own already-shipped, already-tested `dappm`, out of scope to
  silently "fix" here since it's pre-existing, shipped behavior this
  session wasn't asked to touch; flagged in plan.md instead. `PAPPM`
  (`m == 2`, "S"/"M") is genuinely different, not just a renamed DAPPM —
  a new `_pappm` in this file: same 5 pooling branches, but every branch is
  added to `scale0`'s output *independently* (not cascaded onto the
  previous branch's result), concatenated, passed through one 3x3
  **grouped** conv (`groups=4`) rather than 4 separate per-scale 3x3 convs,
  then compressed+shortcut the same way as DAPPM.
- **`segmenthead`**: **not** reused from `_model_ddrnet.py` — checked
  against the verified reference and it's a different, simpler structure
  here (`BN -> ReLU -> Conv3x3(bias=False) -> BN -> ReLU -> Conv1x1(bias=True)`,
  a *single* BN before each of the two convs) than DDRNet's own
  `segmenthead` (which has an extra `BN -> ReLU` pair between the two
  convs) — a local `_pidnet_segmenthead` matching PIDNet's own verified
  structure exactly.
- **Auxiliary taps** (`augment=True`, training-time only, 3 outputs
  total): P's auxiliary head reads `x_` **after** `pag3`'s fusion (not
  before); D's auxiliary head reads `x_d` **after** the `diff4` addition
  but **before** `layer5_d` runs — both confirmed exact tap points from the
  reference's `forward()`, not assumed by analogy to where DDRNet taps its
  own single auxiliary head. `seghead_d` outputs a single channel (a raw
  boundary logit — see `losses.py`'s `pidnet_boundary_loss`/
  `pidnet_boundary_attentive_loss` for how this gets supervised), not
  `n_labels` channels like `seghead_p`/`final_layer`.
- **No final-resolution upsample inside the reference `forward()`** — same
  confirmed gap DDRNet's own module docstring already flagged for that
  network: every output there is left at stride8. This implementation adds
  the same explicit bilinear upsample-to-input-resolution DDRNet already
  adds, for the same reason (matching every other decoder in this
  codebase's "heads upsample internally" convention).

Uses `qat_annotate_layer` wrapping for QAT parity, the same policy as
`ddrnet`: `Add`/`Multiply`/`Concatenate` and any standalone pre-activation
BN not directly following a `Conv2D` are left unannotated (`tfmot`'s
default scheme has no quantization config for Keras merge layers, and no
support for a BN that doesn't directly follow a `Conv2D`/`DepthwiseConv2D`
— both already-confirmed findings from `_model_ddrnet.py`/`_model_lr_aspp.py`
/`_model_pp_liteseg.py`, not re-derived here).

Weights: `weights=None` (random init) only, no pretrained-weight support in
this pass — matching DDRNet's own current status, and consistent with how
every backbone in this plan had its architecture (Phase 2) built well
before its weights were ported (Phase 8). PIDNet's own separately-released
ImageNet backbone checkpoints are a real, open follow-up (plan.md §3.4/
Phase 9), not addressed here.
'''

from __future__ import absolute_import

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Concatenate,
    Input,
    Layer,
    Multiply,
    UpSampling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.utils import register_keras_serializable

from swiss_army_keras.layer_utils import dummy_preprocessing, qat_annotate_layer
from swiss_army_keras._model_ddrnet import (
    basic_block,
    bottleneck_block,
    dappm,
    _avg_pool_branch,
    _bn_relu,
    _conv_bn,
    _conv_bn_relu,
    _make_stage,
)
from swiss_army_keras._model_pp_liteseg import AttentionBlend

import tensorflow_model_optimization as tfmot


@register_keras_serializable(package='swiss_army_keras')
class ChannelSumSigmoid(Layer):
    '''`sigmoid(sum(x * y, axis=channel, keepdims=True))` — PagFM's
    pixel-wise similarity/attention map (the `with_channel=False` path,
    this file's only usage — see module docstring). A proper `Layer`
    subclass rather than raw functional `tf.reduce_sum`/`Lambda` calls, for
    the same QAT-model-cloning-safety reason `AttentionBlend`/
    `ChannelMeanMaxConcat` (`_model_pp_liteseg.py`) and `MultiQueryAttention`
    (`_backbone_mobilenetv4.py`) already are proper `Layer` subclasses.
    '''

    def call(self, inputs):
        x, y = inputs
        return tf.sigmoid(tf.reduce_sum(x * y, axis=-1, keepdims=True))


@register_keras_serializable(package='swiss_army_keras')
class OneMinus(Layer):
    '''`1 - x` — `Light_Bag`'s only use of the complement of its sigmoid
    gate that isn't already covered by `AttentionBlend`'s own internal
    `1 - atten` (Light_Bag's asymmetric combination isn't expressible via
    `AttentionBlend`'s single blend formula, see module docstring). Same
    proper-`Layer`-subclass rationale as `ChannelSumSigmoid` above.
    '''

    def call(self, x):
        return 1.0 - x


def _pagfm(x, y, mid_channels, name=''):
    '''`x`: P branch (target resolution). `y`: I branch (needs upsampling
    to `x`'s resolution, both before and after its own `mid_channels`
    projection — confirmed from source: `f_y(y)` is projected+upsampled to
    compute the attention map, but the final blend uses a *separately*
    upsampled *raw* `y`, not `f_y(y)`).
    '''
    target_h, target_w = int(x.shape[1]), int(x.shape[2])

    y_q = _conv_bn(y, mid_channels, 1, name=f'{name}_f_y')
    y_q = qat_annotate_layer(UpSampling2D(
        size=(target_h // y_q.shape[1], target_w // y_q.shape[2]),
        interpolation='bilinear', name=f'{name}_yq_up',
    ))(y_q)
    x_k = _conv_bn(x, mid_channels, 1, name=f'{name}_f_x')

    sim_map = ChannelSumSigmoid(name=f'{name}_sim')([x_k, y_q])

    y_up = qat_annotate_layer(UpSampling2D(
        size=(target_h // y.shape[1], target_w // y.shape[2]),
        interpolation='bilinear', name=f'{name}_y_up',
    ))(y)

    # x*(1-sim_map) + y_up*sim_map == AttentionBlend(y_up, x, sim_map)
    # (AttentionBlend(a, b, atten) = a*atten + b*(1-atten)).
    return AttentionBlend(name=f'{name}_blend')([y_up, x, sim_map])


def _bag(p, i, d, out_channels, name=''):
    '''`conv3x3(bn_relu(edge_att*P + (1-edge_att)*I))` — the `m != 2`
    ("L") fusion variant. One pre-activation conv, no trailing BN
    (confirmed from source).
    '''
    edge_att = qat_annotate_layer(Activation('sigmoid', name=f'{name}_edge'))(d)
    blended = AttentionBlend(name=f'{name}_blend')([p, i, edge_att])
    blended = _bn_relu(blended, name=f'{name}_pre')
    return qat_annotate_layer(Conv2D(
        out_channels, 3, padding='same', use_bias=False,
        kernel_initializer=HeNormal(), name=f'{name}_conv',
    ))(blended)


def _light_bag(p, i, d, out_channels, name=''):
    '''`conv_p(P + (1-edge_att)*I) + conv_i(I + edge_att*P)` — the `m == 2`
    ("S"/"M") fusion variant. Asymmetric (P/I each appear once gated, once
    not), not expressible via `AttentionBlend`'s single blend formula.
    '''
    edge_att = qat_annotate_layer(Activation('sigmoid', name=f'{name}_edge'))(d)
    inv_edge_att = OneMinus(name=f'{name}_inv_edge')(edge_att)

    p_input = Add(name=f'{name}_p_input')([
        p, Multiply(name=f'{name}_inv_edge_i')([inv_edge_att, i]),
    ])
    i_input = Add(name=f'{name}_i_input')([
        i, Multiply(name=f'{name}_edge_p')([edge_att, p]),
    ])

    p_add = _conv_bn(p_input, out_channels, 1, name=f'{name}_conv_p')
    i_add = _conv_bn(i_input, out_channels, 1, name=f'{name}_conv_i')
    return Add(name=f'{name}_out')([p_add, i_add])


def _pappm(x, branch_channels, out_channels, name='pappm'):
    '''Parallelized DAPPM: the same 5 pooling branches as `dappm`
    (passthrough + 3 average-pool scales `(kernel,stride,pad)` =
    `(5,2,2)`, `(9,4,4)`, `(17,8,8)`, plus a global-average-pool branch),
    but each is added to `scale0`'s output *independently* (not cascaded
    onto the previous branch's own result, unlike `dappm`), concatenated,
    and passed through a single 3x3 **grouped** conv (`groups=4`) rather
    than 4 separate per-scale 3x3 convs — confirmed from source, this is
    the actual mechanism behind "parallel" in the name, not a simplifying
    guess. Every branch ends on a bare conv (no trailing BN), matching the
    reference exactly (unlike this file's reused `dappm`, which has a
    known, documented, deliberately-not-fixed extra-BN discrepancy — see
    module docstring).
    '''
    height, width = int(x.shape[1]), int(x.shape[2])

    x0 = _bn_relu(x, name=f'{name}_scale0')
    x0 = qat_annotate_layer(Conv2D(
        branch_channels, 1, padding='same', use_bias=False,
        kernel_initializer=HeNormal(), name=f'{name}_scale0_conv',
    ))(x0)

    stage_specs = [(5, 2, 2), (9, 4, 4), (17, 8, 8)]
    scale_list = []
    for i, (kernel, stride, pad) in enumerate(stage_specs, start=1):
        pooled = _avg_pool_branch(x, kernel, stride, pad, name=f'{name}_scale{i}')
        pooled = _bn_relu(pooled, name=f'{name}_scale{i}')
        pooled = qat_annotate_layer(Conv2D(
            branch_channels, 1, padding='same', use_bias=False,
            kernel_initializer=HeNormal(), name=f'{name}_scale{i}_conv',
        ))(pooled)
        pooled = qat_annotate_layer(UpSampling2D(
            size=(height // pooled.shape[1], width // pooled.shape[2]),
            interpolation='bilinear', name=f'{name}_scale{i}_up',
        ))(pooled)
        scale_list.append(Add(name=f'{name}_scale{i}_merge')([pooled, x0]))

    global_pooled = qat_annotate_layer(tf.keras.layers.AveragePooling2D(
        pool_size=(height, width), name=f'{name}_scale4_pool',
    ))(x)
    global_pooled = _bn_relu(global_pooled, name=f'{name}_scale4')
    global_pooled = qat_annotate_layer(Conv2D(
        branch_channels, 1, padding='same', use_bias=False,
        kernel_initializer=HeNormal(), name=f'{name}_scale4_conv',
    ))(global_pooled)
    global_pooled = qat_annotate_layer(UpSampling2D(
        size=(height, width), interpolation='bilinear', name=f'{name}_scale4_up',
    ))(global_pooled)
    scale_list.append(Add(name=f'{name}_scale4_merge')([global_pooled, x0]))

    concatenated = Concatenate(axis=-1, name=f'{name}_concat')(scale_list)
    scale_out = _bn_relu(concatenated, name=f'{name}_scale_process')
    scale_out = qat_annotate_layer(Conv2D(
        branch_channels * 4, 3, padding='same', use_bias=False, groups=4,
        kernel_initializer=HeNormal(), name=f'{name}_scale_process_conv',
    ))(scale_out)

    compressed_in = Concatenate(axis=-1, name=f'{name}_compress_concat')([x0, scale_out])
    compressed = _bn_relu(compressed_in, name=f'{name}_compression')
    compressed = qat_annotate_layer(Conv2D(
        out_channels, 1, padding='same', use_bias=False,
        kernel_initializer=HeNormal(), name=f'{name}_compression_conv',
    ))(compressed)

    shortcut = _bn_relu(x, name=f'{name}_shortcut')
    shortcut = qat_annotate_layer(Conv2D(
        out_channels, 1, padding='same', use_bias=False,
        kernel_initializer=HeNormal(), name=f'{name}_shortcut_conv',
    ))(shortcut)

    return Add(name=f'{name}_out')([compressed, shortcut])


def _pidnet_segmenthead(x, mid_channels, out_channels, name='segmenthead'):
    '''`BN -> ReLU -> Conv3x3(bias=False) -> BN -> ReLU -> Conv1x1(bias=True)`
    — a *single* BN before each conv (not reused from `_model_ddrnet.segmenthead`,
    which has an extra `BN -> ReLU` pair between the two convs — see module
    docstring for why that's a real, confirmed difference, not an
    oversight).
    '''
    x = _bn_relu(x, name=f'{name}_pre1')
    x = qat_annotate_layer(Conv2D(
        mid_channels, 3, padding='same', use_bias=False,
        kernel_initializer=HeNormal(), name=f'{name}_conv1',
    ))(x)
    x = _bn_relu(x, name=f'{name}_pre2')
    return qat_annotate_layer(Conv2D(
        out_channels, 1, padding='same', use_bias=True, name=f'{name}_conv2',
    ))(x)


def _bottleneck_stage(x, out_channels, num_blocks, stride=1, name=''):
    '''`_make_stage` wired for `bottleneck_block` (expansion 2): `out_channels`
    is the stage's *final* output width, `bottleneck_block` itself wants
    the pre-expansion mid-channel count (`out_channels // 2`) — the same
    lambda-wrapping trick `_model_ddrnet.ddrnet`'s own `layer5` already
    uses for exactly this reason.
    '''
    return _make_stage(
        x,
        lambda t, out_ch, stride, no_relu, name: bottleneck_block(
            t, out_ch // 2, stride=stride, no_relu=no_relu, name=name),
        out_channels, num_blocks, stride=stride, name=name,
    )


def pidnet(input_tensor, n_labels, m=2, n=3, planes=32, ppm_planes=96,
           head_planes=128, augment=True, name='pidnet'):
    '''PIDNet (Xu et al., 2023) — a complete, self-contained three-branch
    (detail/context/boundary) segmentation network (no `backbone=`
    parameter; see module docstring for why this decoder doesn't follow
    the `backbone_zoo`-based pattern of `lr_aspp`/`pp_liteseg`).

    Input
    ----------
        input_tensor: the input tensor, e.g., `keras.layers.Input((None, None, 3))`.
        n_labels: number of segmentation classes.
        m: **both** the shared trunk's/P branch's per-stage `BasicBlock`
           repeat count *and* the selector for which pooling module
           (`PAPPM` when `m == 2`, else `DAPPM`) and final fusion module
           (`Light_Bag` when `m == 2`, else `Bag`) the network uses —
           confirmed genuine double duty in the reference, not simplified
           here. `m=2` reproduces the official "S"/"M" configs, `m=3`
           (with `planes=64`) reproduces "L".
        n: `layer3`/`layer4`'s (I-branch-only, deeper) `BasicBlock` repeat
           count.
        planes: base channel width — 32 reproduces "S", 64 reproduces
                "M"/"L" (both share `m=2`/`n=3`; only "L" differs, via
                `m=3`/`n=4`/`ppm_planes=112`/`head_planes=256`).
        ppm_planes: the pooling module's (PAPPM/DAPPM) per-branch channel
                    count.
        head_planes: `_pidnet_segmenthead`'s intermediate channel count for
                     the main/P-auxiliary heads (the D-auxiliary head
                     always uses `planes` instead, per the reference).
        augment: if `True` (default, matching the paper's training-time
                 config), also returns the P-branch and D-branch auxiliary
                 heads as 2nd/3rd model outputs (D's is a single-channel
                 raw boundary logit, not `n_labels` channels).
        name: prefix of the created Keras model and its layers.

    Output
    ----------
        A `tf.keras.Model` with a `.preprocessing` attribute
        (`dummy_preprocessing` — no separate registered backbone to
        inherit one from). Output is `[main_output]`, or `[p_aux_output,
        main_output, d_aux_output]` if `augment=True` (this exact order,
        matching the reference's own `[x_extra_p, x_, x_extra_d]`), each an
        `(H, W, C)` logit map at the input's spatial resolution (`C =
        n_labels` for the main/P-aux outputs, `C = 1` for the D-aux one).
    '''
    highres_planes = planes * 2
    input_h, input_w = input_tensor.shape[1], input_tensor.shape[2]
    target_h, target_w = input_h // 8, input_w // 8
    use_light = (m == 2)

    x = _conv_bn_relu(input_tensor, planes, 3, stride=2, name=f'{name}_stem1')
    x = _conv_bn_relu(x, planes, 3, stride=2, name=f'{name}_stem2')

    x = _make_stage(x, basic_block, planes, m, stride=1, name=f'{name}_layer1')
    x = qat_annotate_layer(Activation('relu', name=f'{name}_layer1_relu'))(x)

    x = _make_stage(x, basic_block, planes * 2, m, stride=2, name=f'{name}_layer2')
    x = qat_annotate_layer(Activation('relu', name=f'{name}_layer2_relu'))(x)

    # Branch point: P and D both read this same post-layer2-ReLU tensor.
    x_ = _make_stage(x, basic_block, highres_planes, m, stride=1, name=f'{name}_layer3_')
    d_width_3 = planes if use_light else highres_planes
    x_d = _make_stage(x, basic_block, d_width_3, 1, stride=1, name=f'{name}_layer3_d')

    x = _make_stage(x, basic_block, planes * 4, n, stride=2, name=f'{name}_layer3')
    x = qat_annotate_layer(Activation('relu', name=f'{name}_layer3_relu'))(x)

    compression3 = _conv_bn(x, highres_planes, 1, name=f'{name}_compression3')
    x_ = _pagfm(x_, compression3, planes, name=f'{name}_pag3')

    diff3_width = planes if use_light else highres_planes
    diff3 = _conv_bn(x, diff3_width, 3, name=f'{name}_diff3')
    diff3 = qat_annotate_layer(UpSampling2D(
        size=(target_h // diff3.shape[1], target_w // diff3.shape[2]),
        interpolation='bilinear', name=f'{name}_diff3_up',
    ))(diff3)
    x_d = Add(name=f'{name}_fuse3_d')([x_d, diff3])

    if augment:
        temp_p = x_

    x = _make_stage(x, basic_block, planes * 8, n, stride=2, name=f'{name}_layer4')

    x_ = _make_stage(
        qat_annotate_layer(Activation('relu', name=f'{name}_p3_relu'))(x_),
        basic_block, highres_planes, m, stride=1, name=f'{name}_layer4_',
    )
    if use_light:
        x_d = _bottleneck_stage(
            qat_annotate_layer(Activation('relu', name=f'{name}_d3_relu'))(x_d),
            planes * 2, 1, stride=1, name=f'{name}_layer4_d',
        )
    else:
        x_d = _make_stage(
            qat_annotate_layer(Activation('relu', name=f'{name}_d3_relu'))(x_d),
            basic_block, highres_planes, 1, stride=1, name=f'{name}_layer4_d',
        )

    x = qat_annotate_layer(Activation('relu', name=f'{name}_layer4_relu'))(x)
    compression4 = _conv_bn(x, highres_planes, 1, name=f'{name}_compression4')
    x_ = _pagfm(x_, compression4, planes, name=f'{name}_pag4')

    diff4 = _conv_bn(x, highres_planes, 3, name=f'{name}_diff4')
    diff4 = qat_annotate_layer(UpSampling2D(
        size=(target_h // diff4.shape[1], target_w // diff4.shape[2]),
        interpolation='bilinear', name=f'{name}_diff4_up',
    ))(diff4)
    x_d = Add(name=f'{name}_fuse4_d')([x_d, diff4])

    if augment:
        temp_d = x_d

    x_ = _bottleneck_stage(
        qat_annotate_layer(Activation('relu', name=f'{name}_p4_relu'))(x_),
        planes * 4, 1, stride=1, name=f'{name}_layer5_',
    )
    x_d = _bottleneck_stage(
        qat_annotate_layer(Activation('relu', name=f'{name}_d4_relu'))(x_d),
        planes * 4, 1, stride=1, name=f'{name}_layer5_d',
    )
    x = _bottleneck_stage(
        qat_annotate_layer(Activation('relu', name=f'{name}_i4_relu'))(x),
        planes * 16, 2, stride=2, name=f'{name}_layer5',
    )

    if use_light:
        x = _pappm(x, ppm_planes, planes * 4, name=f'{name}_pappm')
    else:
        x = dappm(x, ppm_planes, planes * 4, name=f'{name}_dappm')
    x = qat_annotate_layer(UpSampling2D(
        size=(target_h // x.shape[1], target_w // x.shape[2]),
        interpolation='bilinear', name=f'{name}_spp_up',
    ))(x)

    if use_light:
        fused = _light_bag(x_, x, x_d, planes * 4, name=f'{name}_dfm')
    else:
        fused = _bag(x_, x, x_d, planes * 4, name=f'{name}_dfm')

    main_out = _pidnet_segmenthead(fused, head_planes, n_labels, name=f'{name}_final_head')
    main_out = qat_annotate_layer(UpSampling2D(
        size=(input_h // main_out.shape[1], input_w // main_out.shape[2]),
        interpolation='bilinear', name=f'{name}_final_upsample',
    ))(main_out)

    if not augment:
        outputs = [main_out]
    else:
        p_aux = _pidnet_segmenthead(temp_p, head_planes, n_labels, name=f'{name}_p_head')
        p_aux = qat_annotate_layer(UpSampling2D(
            size=(input_h // p_aux.shape[1], input_w // p_aux.shape[2]),
            interpolation='bilinear', name=f'{name}_p_upsample',
        ))(p_aux)

        d_aux = _pidnet_segmenthead(temp_d, planes, 1, name=f'{name}_d_head')
        d_aux = qat_annotate_layer(UpSampling2D(
            size=(input_h // d_aux.shape[1], input_w // d_aux.shape[2]),
            interpolation='bilinear', name=f'{name}_d_upsample',
        ))(d_aux)

        outputs = [p_aux, main_out, d_aux]

    model = Model([input_tensor, ], outputs)

    with tfmot.quantization.keras.quantize_scope():
        model = tfmot.quantization.keras.quantize_apply(model)

    model.preprocessing = dummy_preprocessing

    return model
