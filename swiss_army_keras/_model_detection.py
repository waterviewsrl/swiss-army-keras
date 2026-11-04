
'''Lightweight anchor-free object detection (plan.md §9.3).

**Scope decision, resolving §7/§9.3's own open fork**: plan.md presented two
options — a primary NanoDet-Plus-style GFL head (joint classification+
IoU-quality target via Quality Focal Loss, box sides regressed as small
discrete distributions via Distribution Focal Loss) or a "fallback: FCOS-lite
head... if GFL's distribution-regression is judged too complex for a first
pass". **This implements FCOS-lite**, not GFL — judged too complex for a
first pass here specifically because GFL's own quality assignment couples
tightly to the *label assignment* strategy (the joint target needs each
positive anchor's actual regressed IoU with its matched GT, computed
end-to-end, not just a static one-hot label), compounding two genuinely new,
hard-to-independently-verify pieces (a novel loss target *and* a novel label
assignment) into one implementation with nothing in this repo to check either
against. FCOS-lite decouples them cleanly (a standard hard classification
target + a separate centerness branch, assigned independently) — both
`quality_focal_loss`/`distribution_focal_loss` (plan.md §9.4) remain
implemented and available in `losses.py` for a future GFL head, they're just
not wired into this file's head.

Similarly, of the two label-assignment options (**ATSS** or **SimOTA**),
this implements **ATSS** (Zhang et al. 2020) — a fixed statistics-based
threshold per ground-truth box, no learned/iterative solver — over SimOTA
(an optimal-transport problem solved per-image via Sinkhorn-like iteration),
for the same reason: fewer moving, independently-unverifiable parts for a
first pass, and ATSS is explicitly named in plan.md as an equally acceptable
option ("either is a meaningful quality upgrade").

Architecture
----------
- **Neck**: a thin PAFPN-lite (top-down FPN pass + bottom-up PAN pass) over
  3 backbone feature levels (P3/P4/P5 — `feature_levels=(3, 4, 5)` by
  default, the same 1-indexed convention `pp_liteseg` uses, with the same
  `(2, 3, 4)` override for 4-tap backbones like RepViT/FastViT). Plan.md
  suggested reusing SPPM (§3.2)/DAPPM (§3.3) as detection necks too — not
  done here: those modules operate on a *single* deepest feature map,
  while a 3-level FPN/PAN fusion is the part actually needed for a
  multi-scale detection head, and grafting a single-input pooling module
  into that role would have required design choices plan.md doesn't
  specify. Left as a genuine follow-up, not silently skipped.
- **Head**: FCOS-style — depthwise-separable conv stems (their own
  `Conv2D`/`BatchNormalization` *layer objects* built once, then reused —
  true Keras weight sharing — across all 3 pyramid levels, not 3
  independent copies), split into a classification branch (per-class
  sigmoid, plain hard 0/1 target — not QFL's joint quality target, since
  this is FCOS-lite not GFL), a regression branch (4 values, `(l, t, r, b)`
  distances to the box edges, `exp`-activated and scaled by a per-level
  learnable scalar — the standard FCOS trick letting each pyramid level
  cover its own natural box-size range), and a centerness branch (a single
  sigmoid tapped off the regression stem's features, FCOS's own
  low-quality/off-center suppression signal, multiplied into the
  classification score at inference time, not training time).
- **Label assignment (training-time only, not part of the Keras graph)**:
  `atss_assign_targets` — pure numpy, since per-image ground-truth boxes
  are naturally variable-length, which is awkward inside a static TF graph
  and is not how label assignment is typically done in reference detector
  implementations either (it's a data-pipeline step, not a graph op) — same
  reasoning `boundary_loss` (§4.1) already used for its own
  per-batch-numpy signed-distance-map computation.
- **Inference**: raw per-level predictions (`<name>_base`, exposed via
  `base.py`, matching every decoder's own `_base`/full split convention)
  are decoded into absolute boxes and filtered via `_nms_dispatch` in the
  full `<name>` model (`models.py`) — **not** `tf.image.
  combined_non_max_suppression` (tried first, and shipped that way through
  plan.md §9.3/§13.6's own original passes): found, while verifying an
  unrelated fix, to never actually convert under strict
  `TFLITE_BUILTINS_INT8` export at all (`ERROR_NEEDS_FLEX_OPS`) — see
  `_nms_dispatch`'s own docstring for the full account, including the real
  batching-capability tradeoff its replacement brings (batch-size-1 by
  default, real fixed-size multi-image batching as an opt-in via a static-
  batch `input_tensor`).

GFL-style upgrade (plan.md §9.5/§9.6, Phase 9)
----------
`gfl_lite_detection`/`gfl_lite_detection_base` are a second, independent
detector built in this same file — the concrete resolution of the head-fork
this module's own docstring deferred above (`quality_focal_loss`/
`distribution_focal_loss` "remain implemented... for a future GFL head").
Resolves plan.md §9.5's "two real designs share one name" fork as **v1**
(the arXiv paper: VarifocalLoss/QFL + SimOTA), not v2 (`PicoHeadV2`'s
TaskAlignedAssigner + GFLv2 alignment head) — v1 reuses `quality_focal_loss`/
`distribution_focal_loss`/`varifocal_loss`/`iou_box(mode='giou')`, all
already implemented (§9.4) but never wired into a head. §9.5 and §9.6
(NanoDet-Plus) are built together, not as two separate detectors — they
overlap substantially (near-identical head, assigners that are refinements
of each other): `gfl_lite_head` pairs with a choice of **three** selectable
assigners (`atss_assign_targets`, above; `simota_assign_targets`, PicoDet
v1; `dsla_assign_targets`, NanoDet-Plus) and a new `ghost_pan_lite_neck`
(NanoDet-Plus's own neck choice — CSP-PAN, PP-PicoDet's neck upgrade, was
not built this pass).

Verified directly against raw source (`nanodet/model/head/
nanodet_plus_head.py`, `.../gfl_head.py`, `.../head/assigner/dsl_assigner.py`,
`.../fpn/ghost_pan.py`, `.../backbone/ghostnet.py`, PaddleDetection's
`ppdet/modeling/assigners/simota_assigner.py`) — this found two things this
module's earlier research summary (plan.md §9.5/§9.6) had wrong:
**Ghost-PAN does not share `pafpn_lite_neck`'s topology** (concat + a Ghost
block projecting back down to `out_channels`, not add-then-smoothing-conv —
`ghost_pan_lite_neck` is its own function, not a `pafpn_lite_neck`
variant), and **the optional 4th pyramid level is an elementwise add of two
independent stride-2 convs** (one off the channel-equalized deepest input,
one off the neck's own final bottom-up output) — not "two stride-2 convs
in sequence" as originally described.

- **No centerness branch** — `gfl_lite_head`'s defining structural
  difference from `fcos_lite_head`: QFL/VFL's own joint classification+
  IoU-quality target (`y_true` = 0 for negatives, the matched box's IoU for
  positives) replaces centerness entirely, rather than a separate branch
  multiplied in at inference time.
- **One shared tower per level, not separate cls/reg branches** — what
  §9.5 flagged as PicoDet's optional `share_cls_reg`: here it's simply the
  design (GFL's head *is* one tower whose final conv splits into
  classification and distribution-regression channels), not an opt-in flag.
  Still gets its own independently-initialized stem per level (reusing
  `_independent_depthwise_sep_stem`) for the same `tfmot` QAT-cloning
  reason `fcos_lite_head` already documents above.
- **Box sides are regressed as small discrete distributions** (`reg_max=7`,
  8 bins per side, the reference default), not a direct scalar — trained
  via `distribution_focal_loss` (§9.4), decoded via `gfl_integral_decode`'s
  softmax-expectation projection (`Integral`'s own formula:
  `sum(softmax(logits) * arange(reg_max + 1)) * stride`).
- **SimOTA and DSLA are dynamic assigners, genuinely different in kind
  from `atss_assign_targets`**: ATSS depends only on box geometry
  (precomputable ahead of training). SimOTA/DSLA's cost matrix needs the
  model's own *current* predicted scores/boxes, recomputed every training
  step (confirmed from source — the reference calls its own assigner
  per-batch inside the loss, fed `.detach()`'d live predictions) — this
  repo's first precedent for a step-dependent assigner. Exposed with an
  explicit `pred_scores_per_level`/`pred_boxes_per_level` signature rather
  than hiding that dependency; intended to be called from a custom
  `GradientTape` training step (fed `tf.stop_gradient`'d current
  predictions), the same way `atss_assign_targets` is intended to be
  called from a data pipeline — neither is wired into a `model.fit`-ready
  loss or `dataset_utils.py` (which has no detection loader at all), for
  the same "one loss/utility per concern, combination documented not
  forced" reasoning `pidnet_boundary_loss`/`pidnet_boundary_attentive_loss`
  already established.
- **Not ported this pass**: AGM (NanoDet-Plus's training-only auxiliary
  head — DSLA computes its cost from the real head's own predictions
  instead, per §9.6's own recommendation), CSP-PAN, and PicoHeadV2/
  TaskAlignedAssigner ("v2"'s design).
'''

from __future__ import absolute_import

import math

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    DepthwiseConv2D,
    Layer,
    UpSampling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable

from swiss_army_keras._backbone_zoo import backbone_zoo
from swiss_army_keras.layer_utils import dummy_preprocessing, qat_annotate_layer

import tensorflow_model_optimization as tfmot


# ========================= #
# PAFPN-lite neck


def _conv_bn_act(x, out_channels, kernel_size=1, stride=1, use_bias=False, activation='relu', name=''):
    x = qat_annotate_layer(Conv2D(
        out_channels, kernel_size, strides=stride, padding='same', use_bias=use_bias, name=f'{name}_conv',
    ))(x)
    x = qat_annotate_layer(BatchNormalization(name=f'{name}_bn'))(x)
    if activation is not None:
        x = qat_annotate_layer(Activation(activation, name=f'{name}_act'))(x)
    return x


def pafpn_lite_neck(features, out_channels=96, name='pafpn'):
    '''A thin PAFPN (Path-Aggregation FPN): a top-down FPN pass (deepest
    feature upsampled and added into each shallower one, lateral 1x1 convs
    projecting every level to a common width first) followed by a
    bottom-up PAN pass (each level re-downsampled and added back into the
    next-deeper one) — letting shallow, high-resolution features benefit
    from deep semantic context (top-down) *and* deep features benefit from
    shallow, precisely-localized detail (bottom-up), rather than only one
    direction of fusion.

    Input
    ----------
        features: a 3-element list `[P3, P4, P5]`, shallowest first (e.g.
                  strides 8/16/32), as returned by `backbone_zoo`.
        out_channels: common channel width every output level is projected to.

    Output
    ----------
        A 3-element list `[P3_out, P4_out, P5_out]`, same order/spatial
        sizes as the input, all at `out_channels` width.
    '''
    p3, p4, p5 = features

    lat3 = _conv_bn_act(p3, out_channels, kernel_size=1, name=f'{name}_lat3')
    lat4 = _conv_bn_act(p4, out_channels, kernel_size=1, name=f'{name}_lat4')
    lat5 = _conv_bn_act(p5, out_channels, kernel_size=1, name=f'{name}_lat5')

    # Top-down (FPN).
    td5 = lat5
    up5 = qat_annotate_layer(UpSampling2D(
        size=(lat4.shape[1] // td5.shape[1], lat4.shape[2] // td5.shape[2]),
        interpolation='nearest', name=f'{name}_up5',
    ))(td5)
    td4 = Add(name=f'{name}_td4_add')([lat4, up5])
    td4 = _conv_bn_act(td4, out_channels, kernel_size=3, name=f'{name}_td4_smooth')

    up4 = qat_annotate_layer(UpSampling2D(
        size=(lat3.shape[1] // td4.shape[1], lat3.shape[2] // td4.shape[2]),
        interpolation='nearest', name=f'{name}_up4',
    ))(td4)
    td3 = Add(name=f'{name}_td3_add')([lat3, up4])
    td3 = _conv_bn_act(td3, out_channels, kernel_size=3, name=f'{name}_td3_smooth')

    # Bottom-up (PAN).
    out3 = td3
    down3 = _conv_bn_act(out3, out_channels, kernel_size=3, stride=2, name=f'{name}_down3')
    out4 = Add(name=f'{name}_bu4_add')([td4, down3])
    out4 = _conv_bn_act(out4, out_channels, kernel_size=3, name=f'{name}_bu4_smooth')

    down4 = _conv_bn_act(out4, out_channels, kernel_size=3, stride=2, name=f'{name}_down4')
    out5 = Add(name=f'{name}_bu5_add')([td5, down4])
    out5 = _conv_bn_act(out5, out_channels, kernel_size=3, name=f'{name}_bu5_smooth')

    return [out3, out4, out5]


# ========================= #
# Ghost-PAN neck (plan.md §9.6, Phase 9) - a genuinely different fusion
# mechanism from `pafpn_lite_neck` above, not a "swap the conv type" variant
# of it (see module docstring's "GFL-style upgrade" section for the
# correction this found relative to plan.md's own earlier description).


def _ghost_module(x, out_channels, kernel_size=1, ratio=2, dw_size=3, activation='relu', name=''):
    '''GhostNet's "cheap operations" module (Han et al. 2020,
    arXiv:1911.11907) - a normal conv produces a handful of "intrinsic"
    channels (`init_channels = ceil(out_channels / ratio)`), then a cheap
    depthwise op cheaply expands those to the rest of the target channel
    count (`new_channels = init_channels * (ratio - 1)`), concatenated
    together. **Real output width is `init_channels * ratio`, which can
    exceed `out_channels` by a few channels when `out_channels` isn't
    divisible by `ratio`** - matches the reference `GhostModule` exactly
    (confirmed from `ghostnet.py`: it doesn't trim the extra channels
    either), not a bug introduced by this port.
    '''
    init_channels = math.ceil(out_channels / ratio)
    new_channels = init_channels * (ratio - 1)

    primary = qat_annotate_layer(Conv2D(
        init_channels, kernel_size, padding='same', use_bias=False, name=f'{name}_primary_conv'))(x)
    primary = qat_annotate_layer(BatchNormalization(name=f'{name}_primary_bn'))(primary)
    if activation is not None:
        primary = qat_annotate_layer(Activation(activation, name=f'{name}_primary_act'))(primary)

    if new_channels <= 0:
        return primary

    cheap = qat_annotate_layer(DepthwiseConv2D(
        dw_size, padding='same', use_bias=False, depth_multiplier=ratio - 1, name=f'{name}_cheap_conv'))(primary)
    cheap = qat_annotate_layer(BatchNormalization(name=f'{name}_cheap_bn'))(cheap)
    if activation is not None:
        cheap = qat_annotate_layer(Activation(activation, name=f'{name}_cheap_act'))(cheap)

    return Concatenate(name=f'{name}_concat')([primary, cheap])


def _ghost_bottleneck(x, mid_channels, out_channels, kernel_size=5, activation='relu', name=''):
    '''`GhostBottleneck` (`ghostnet.py`), scoped to how `GhostPAN`'s own
    fusion blocks actually call it: `stride=1`, no squeeze-excite (both are
    real options in the reference, used by GhostNet's own *backbone* stages,
    but never by the neck) - an expansion `_ghost_module` (1x1, `activation`),
    a projection `_ghost_module` (1x1, no activation), and a residual
    connection (identity if the channel count doesn't change, otherwise a
    depthwise+pointwise conv projecting the raw input to `out_channels`).
    '''
    in_channels = int(x.shape[-1])

    y = _ghost_module(x, mid_channels, kernel_size=1, activation=activation, name=f'{name}_expand')
    y = _ghost_module(y, out_channels, kernel_size=1, activation=None, name=f'{name}_project')

    if in_channels == out_channels:
        shortcut = x
    else:
        shortcut = qat_annotate_layer(DepthwiseConv2D(
            kernel_size, padding='same', use_bias=False, name=f'{name}_shortcut_dw'))(x)
        shortcut = qat_annotate_layer(BatchNormalization(name=f'{name}_shortcut_dw_bn'))(shortcut)
        shortcut = qat_annotate_layer(Conv2D(
            out_channels, 1, padding='same', use_bias=False, name=f'{name}_shortcut_pw'))(shortcut)
        shortcut = qat_annotate_layer(BatchNormalization(name=f'{name}_shortcut_pw_bn'))(shortcut)

    return Add(name=f'{name}_add')([y, shortcut])


def _ghost_blocks(x, out_channels, expand=1, kernel_size=5, num_blocks=1, use_res=False, activation='relu', name=''):
    '''`GhostBlocks` (`ghost_pan.py`) - a short stack of `_ghost_bottleneck`s.
    **`num_blocks > 1` reproduces a real reference quirk**: every block's
    expansion step reads the *original* input's channel count, not the
    previous block's own output width - only shape-safe when the input and
    output channel counts match, true for `GhostPAN`'s own only usage
    (`num_blocks=1`). Kept faithful rather than silently "fixed", since
    `num_blocks=1` (this function's own default, matching every call site
    in the reference) never triggers it.
    '''
    in_channels = int(x.shape[-1])
    mid_channels = int(out_channels * expand)

    y = x
    for i in range(num_blocks):
        y = _ghost_bottleneck(y, mid_channels, out_channels, kernel_size=kernel_size,
                              activation=activation, name=f'{name}_block{i}')

    if use_res:
        shortcut = _conv_bn_act(x, out_channels, kernel_size=1, activation=activation, name=f'{name}_res')
        y = Add(name=f'{name}_res_add')([y, shortcut])

    return y


def ghost_pan_lite_neck(features, out_channels=96, kernel_size=5, expand=1, num_blocks=1,
                        num_extra_levels=0, activation='relu', name='ghost_pan'):
    '''NanoDet-Plus's own neck (`GhostPAN`, `ghost_pan.py`) - same
    top-down-then-bottom-up shape as `pafpn_lite_neck` above, but each
    fusion step is a channel **concat** followed by a `_ghost_blocks`
    projection back down to `out_channels`, not an `Add` of same-width
    tensors followed by a smoothing conv - a genuinely different fusion
    mechanism, confirmed from source (see module docstring). Also uses
    **bilinear** upsampling in the top-down pass (the reference's own
    default), unlike `pafpn_lite_neck`'s nearest-neighbor.

    Input
    ----------
        features: shallowest-first list of feature maps (e.g. `[P3, P4, P5]`).
        out_channels: common channel width every fused level lands on.
        kernel_size: kernel size used by the bottom-up downsample convs and
                     every `_ghost_bottleneck`'s depthwise steps (5, the
                     reference's own default - larger than `pafpn_lite_neck`'s
                     3, part of how Ghost-PAN affords cheaper-per-parameter
                     receptive field growth).
        expand, num_blocks: `_ghost_blocks`' own parameters, per fusion node.
        num_extra_levels: if > 0, appends that many extra, coarser pyramid
                          levels (plan.md's own "optional 4th level") - each
                          `Add([stride2_conv(<the previous extra level's
                          input>), stride2_conv(<the previous extra level's
                          own predecessor output>)])`, matching the
                          reference's `extra_in_layer(inputs[-1]) +
                          extra_out_layer(outs[-1])` exactly (an elementwise
                          add of two independent stride-2 convs, not two
                          convs applied in sequence - a correction to this
                          plan's own earlier description, see module
                          docstring).

    Output
    ----------
        A list of `len(features) + num_extra_levels` feature maps,
        shallowest first, all at `out_channels` width.
    '''
    reduced = [
        _conv_bn_act(f, out_channels, kernel_size=1, activation=activation, name=f'{name}_reduce{i}')
        for i, f in enumerate(features)
    ]
    n_levels = len(reduced)

    # Top-down (concat + Ghost-block fusion, not Add + smoothing conv).
    inner = [reduced[-1]]
    for idx in range(n_levels - 1, 0, -1):
        feat_high = inner[0]
        feat_low = reduced[idx - 1]

        up = qat_annotate_layer(UpSampling2D(
            size=(feat_low.shape[1] // feat_high.shape[1], feat_low.shape[2] // feat_high.shape[2]),
            interpolation='bilinear', name=f'{name}_up{idx}',
        ))(feat_high)
        cat = Concatenate(name=f'{name}_td{idx}_cat')([up, feat_low])
        inner_out = _ghost_blocks(cat, out_channels, expand=expand, kernel_size=kernel_size,
                                  num_blocks=num_blocks, activation=activation, name=f'{name}_td{idx}')
        inner.insert(0, inner_out)

    # Bottom-up (concat + Ghost-block fusion).
    outs = [inner[0]]
    for idx in range(n_levels - 1):
        feat_low = outs[-1]
        feat_high = inner[idx + 1]

        down = _conv_bn_act(feat_low, out_channels, kernel_size=kernel_size, stride=2,
                            activation=activation, name=f'{name}_down{idx}')
        cat = Concatenate(name=f'{name}_bu{idx}_cat')([down, feat_high])
        out = _ghost_blocks(cat, out_channels, expand=expand, kernel_size=kernel_size,
                            num_blocks=num_blocks, activation=activation, name=f'{name}_bu{idx}')
        outs.append(out)

    # Extra, coarser level(s) - opt-in, off by default.
    for level_idx in range(num_extra_levels):
        extra_in = _conv_bn_act(reduced[-1], out_channels, kernel_size=kernel_size, stride=2,
                                activation=activation, name=f'{name}_extra{level_idx}_in')
        extra_out = _conv_bn_act(outs[-1], out_channels, kernel_size=kernel_size, stride=2,
                                 activation=activation, name=f'{name}_extra{level_idx}_out')
        outs.append(Add(name=f'{name}_extra{level_idx}_add')([extra_in, extra_out]))

    return outs


# ========================= #
# FCOS-lite head


@register_keras_serializable(package='swiss_army_keras')
class _AssertBatchSizeOne(Layer):
    '''Runtime batch-size-1 guard for `fcos_lite_instance_segmentation`'s
    own single-image NMS/gather/mask-assembly path, and for
    `_indexed_nms_gather_batch1`'s own dynamic-batch NMS path (see either
    docstring for why that path is batch-size-1 only). `tf.debugging.
    assert_equal` can't be used directly inline in a Keras functional
    graph - Keras's own automatic raw-TF-op wrapping (`TFOpLambda`) can't
    build a `TypeSpec` for the control-only `Assert` op the check produces,
    confirmed empirically (`TypeError: Could not build a TypeSpec for ...
    op: "Assert"` when tried without this wrapper). Wrapping it in a real
    `Layer.call` sidesteps that - Keras leaves a layer's own internal op
    graph alone.

    **`dtype` forced to `float32` unconditionally** - a real, confirmed
    regression found via `tests/test_mixed_precision_qat.py`: under a
    global mixed-precision policy (e.g. `mixed_float16`), a plain custom
    `Layer` with no explicit `dtype` picks up the ambient policy and
    silently casts its own output down to float16 - the exact same bug
    class `fcos_lite_head`'s old `ScaleLayer` had (§10), for the same
    underlying reason (a non-QAT-annotated layer downstream of QAT-forced-
    float32 boxes, feeding into `_indexed_nms_gather`'s own plain
    functional ops, which have no ambient policy of their own and just
    inherit whatever dtype they're handed).
    '''

    def __init__(self, **kwargs):
        kwargs['dtype'] = 'float32'
        super().__init__(**kwargs)

    def call(self, x):
        with tf.control_dependencies([
            tf.debugging.assert_equal(
                tf.shape(x)[0], 1,
                message="fcos_lite_instance_segmentation's full model only supports batch_size=1 "
                        "(see its own docstring) - use fcos_lite_instance_segmentation_base for "
                        "batched training."),
        ]):
            return tf.identity(x)


def _qat_channelwise_scale(x, name):
    '''A learnable per-channel multiplicative scale — FCOS's own trick for
    letting each pyramid level's regression branch cover its own natural
    box-size range despite sharing the same regression-branch *weights*
    across every level (the branch itself has no way to know which level
    it's being applied to, otherwise). Used by both `fcos_lite_head` and
    `fcos_lite_instance_head` (§13.6).

    **Replaces a former custom `ScaleLayer` (a plain `add_weight()`-based
    `Layer`), removed entirely — not a design choice made for its own
    sake, a fix for a real, confirmed `tfmot` incompatibility.**
    `ScaleLayer` placed directly downstream of a `qat_annotate_layer`-
    wrapped `Conv2D` corrupted `tfmot.quantization.keras.quantize_apply`'s
    internal `Model.from_config` node reconstruction in this repo's pinned
    environment (`tensorflow==2.21.0`/`tf_keras==2.21.0`/
    `tensorflow_model_optimization==0.8.1`) —
    `AttributeError: 'str' object has no attribute 'name'` (or a sibling
    `TypeError: 'str' object is not callable`) — confirmed via a minimal,
    from-scratch reproduction completely independent of this file's own
    complexity (a single `Conv2D` -> `ScaleLayer` -> `Model`, no backbone,
    no sharing, no multi-level anything, already failed). Root cause
    isolated to *any* custom `Layer` subclass with its own trainable
    weight (`add_weight()`/`build()`) sitting immediately after a QAT
    layer — a weightless custom `Layer` in the same position is fine, so
    is a raw `tf.exp(...)`/multiply.

    Implemented as a `DepthwiseConv2D(1, use_bias=False)` (kernel
    initialized to `1.0`, i.e. starts as a true identity/no-op, exactly
    matching `ScaleLayer`'s own `initial_value=1.0` starting point) — a
    stock, QAT-registry-known layer type, so `qat_annotate_layer` applies
    to it the normal way, sidestepping the "unannotated custom weighted
    Layer downstream of a QAT layer" trigger entirely (nothing custom is
    left un-annotated here). Being QAT-annotated also means it's already
    covered by `qat_annotate_layer`'s own `QuantizeWrapper`/`QuantizeLayer`
    float32-forcing patch (§10) — under a global mixed-precision policy,
    this layer's output stays float32 without needing `ScaleLayer`'s old
    explicit `dtype='float32'` override (confirmed via
    `tests/test_mixed_precision_qat.py`'s own regression test for exactly
    this).

    **One real, documented behavior difference from `ScaleLayer`**:
    `ScaleLayer` was a single scalar shared across all channels;
    `DepthwiseConv2D(1)` gives each channel (4, for `reg_ltrb`'s own
    `l/t/r/b`) its own independent learnable scale — a strict
    generalization (4 degrees of freedom instead of 1), not a loss of
    capability, at the cost of 3 extra floats per level.
    '''
    return qat_annotate_layer(DepthwiseConv2D(
        1, use_bias=False, depthwise_initializer=tf.keras.initializers.Constant(1.0),
        name=f'{name}_scale',
    ))(x)


def _independent_depthwise_sep_stem(x, prefix, n_layers, channels):
    '''Applies `n_layers` depthwise-separable conv blocks
    (`DepthwiseConv2D -> BN -> ReLU -> Conv2D(1x1) -> BN -> ReLU`), each
    with its own freshly-initialized weights — used once per pyramid level
    (see `fcos_lite_head`'s docstring for why this is *not* a literal
    weight-shared-across-levels stem the way "true" FCOS uses, despite the
    architecture otherwise matching it: `tfmot`'s default QAT scheme cannot
    clone a model where a `Conv2D`/`DepthwiseConv2D` layer object is called
    at more than one call site (e.g. once per pyramid level) if a
    `BatchNormalization` follows it *anywhere* in that usage — confirmed via
    a minimal, isolated reproduction independent of this file's own
    complexity (`quantize_apply` raises `RuntimeError:
    ...BatchNormalization is not supported`, regardless of whether the BN
    instance itself is shared or fresh per call, and regardless of how many
    times the preceding conv gets wrapped in `quantize_annotate_layer`).
    '''
    for i in range(n_layers):
        x = qat_annotate_layer(
            DepthwiseConv2D(3, padding='same', use_bias=False, name=f'{prefix}_dw{i}'))(x)
        x = qat_annotate_layer(
            BatchNormalization(name=f'{prefix}_dw{i}_bn'))(x)
        x = qat_annotate_layer(
            Activation('relu', name=f'{prefix}_dw{i}_act'))(x)
        x = qat_annotate_layer(
            Conv2D(channels, 1, padding='same', use_bias=False, name=f'{prefix}_pw{i}'))(x)
        x = qat_annotate_layer(
            BatchNormalization(name=f'{prefix}_pw{i}_bn'))(x)
        x = qat_annotate_layer(
            Activation('relu', name=f'{prefix}_pw{i}_act'))(x)
    return x


def fcos_lite_head(fpn_features, n_classes, strides=(8, 16, 32), stem_channels=96, stem_layers=2, name='fcos_head'):
    '''FCOS-lite detection head applied independently to each of
    `fpn_features`'s 3 levels, each level's regression output scaled by its
    own learnable per-channel scale (`_qat_channelwise_scale`).

    **Not weight-tied across levels, unlike "true" FCOS's own shared head**
    — each level gets its own independently-initialized classification and
    regression/centerness stems (`_independent_depthwise_sep_stem`), a
    deliberate simplification forced by a `tfmot` QAT-cloning limitation
    with multi-call-site `Conv2D`/`BatchNormalization` layers (see that
    function's docstring for the full account) rather than a design choice
    made for its own sake. The final 1x1 output convs (classification,
    regression, centerness) *are* still shared across levels — those aren't
    followed by a BN, so they aren't affected by the same limitation, and
    keeping them shared costs nothing.

    Input
    ----------
        fpn_features: 3-element list `[P3, P4, P5]` (e.g. from `pafpn_lite_neck`).
        n_classes: number of object classes (no background class — FCOS-lite
                   uses independent per-class sigmoids, not a softmax over
                   `n_classes + 1`).
        strides: each level's total downsampling factor relative to the
                 input, in the same order as `fpn_features` — needed to
                 scale the regression branch's raw output into input-pixel
                 units (`exp(raw) * stride * per_level_scale`).
        stem_channels: each level's independent stems' channel width.
        stem_layers: number of depthwise-separable conv blocks per stem.

    Output
    ----------
        A list of 3 `(cls_logits, reg_ltrb, centerness)` tuples, one per
        input level — `cls_logits` shape `(B, H, W, n_classes)` (sigmoid
        *logits*, not yet activated — apply `sigmoid` before use, or feed
        directly to a loss that expects logits), `reg_ltrb` shape
        `(B, H, W, 4)` (already positive, in input-pixel units), `centerness`
        shape `(B, H, W, 1)` (sigmoid logits).
    '''
    cls_head_conv = qat_annotate_layer(
        Conv2D(n_classes, 1, padding='same', name=f'{name}_cls_out'))
    reg_head_conv = qat_annotate_layer(
        Conv2D(4, 1, padding='same', name=f'{name}_reg_out'))
    centerness_head_conv = qat_annotate_layer(
        Conv2D(1, 1, padding='same', name=f'{name}_centerness_out'))

    outputs = []
    for level_idx, (feature, stride) in enumerate(zip(fpn_features, strides)):
        level_name = f'p{level_idx}'

        cls_feat = _independent_depthwise_sep_stem(feature, f'{name}_cls_{level_name}', stem_layers, stem_channels)
        cls_logits = cls_head_conv(cls_feat)

        reg_feat = _independent_depthwise_sep_stem(feature, f'{name}_reg_{level_name}', stem_layers, stem_channels)
        reg_raw = reg_head_conv(reg_feat)
        # QAT-annotated (see _qat_channelwise_scale's own docstring), so
        # already covered by qat_annotate_layer's float32-forcing patch
        # under a global mixed-precision policy - no separate dtype
        # override needed here, unlike the old ScaleLayer-based version
        # this replaced (a plain, un-annotated custom Layer, which used to
        # silently pick up the ambient float16 policy and crash
        # `tf.image.combined_non_max_suppression` downstream, which
        # hard-requires float32 boxes).
        reg_raw = _qat_channelwise_scale(reg_raw, name=f'{name}_{level_name}')
        reg_ltrb = tf.exp(reg_raw) * float(stride)

        centerness = centerness_head_conv(reg_feat)

        outputs.append((cls_logits, reg_ltrb, centerness))

    return outputs


# ========================= #
# GFL-style head (plan.md §9.5/§9.6, Phase 9) - see module docstring's
# "GFL-style upgrade" section for the full account.


def gfl_lite_head(fpn_features, n_classes, reg_max=7, stem_channels=96, stem_layers=2, name='gfl_head'):
    '''GFL-style detection head (`NanoDetPlusHead`, `nanodet_plus_head.py`) -
    one shared conv tower per level (not `fcos_lite_head`'s separate cls/reg
    branches - what plan.md §9.5 flagged as PicoDet's optional
    `share_cls_reg` is simply this head's actual design), whose final 1x1
    conv jointly produces classification logits *and* box-side distribution
    logits. No centerness branch - QFL/VFL's own joint quality target
    (see below) replaces it.

    Same `tfmot` QAT-cloning limitation `fcos_lite_head`/
    `_independent_depthwise_sep_stem` already documents applies identically
    here (a `Conv2D`/`BatchNormalization` called at more than one pyramid
    level fails `quantize_apply` if followed anywhere by a `BatchNormalization`)
    - each level gets its own independently-initialized stem, reusing that
    same helper.

    Input
    ----------
        fpn_features: shallowest-first list of feature maps (e.g. from
                      `ghost_pan_lite_neck`).
        n_classes: number of object classes (independent per-class sigmoids,
                   like `fcos_lite_head` - not a softmax over `n_classes + 1`).
        reg_max: each box side is regressed as a discrete distribution over
                 `reg_max + 1` bins (0..reg_max, in units of that level's
                 stride) - 7, the reference's own default.
        stem_channels, stem_layers: per-level independent stem config, same
                                    meaning as `fcos_lite_head`'s.

    Output
    ----------
        A list of `(cls_logits, reg_dist_logits)` tuples, one per input
        level - `cls_logits` shape `(B, H, W, n_classes)` (raw logits,
        `sigmoid` before use - see training-combination note below),
        `reg_dist_logits` shape `(B, H, W, 4, reg_max + 1)` (raw per-bin
        logits per side, `l, t, r, b` order matching `decode_boxes`,
        `softmax` over the trailing axis before use, e.g. via
        `gfl_integral_decode`).

    Training-time loss combination (documented, not wired into a single
    forced loss function - see module docstring's "not wired into a
    `model.fit`-ready loss" note):
        `quality_focal_loss()` or `varifocal_loss()` on
        `(joint_quality_target, sigmoid(cls_logits))` (`joint_quality_target`
        is 0 for negatives, the matched ground-truth box's IoU for
        positives - exactly what `simota_assign_targets`/`dsla_assign_targets`/
        `atss_assign_targets`'s own `cls_targets` return already is);
        `distribution_focal_loss()` on `(dist_target_in_bin_units,
        reg_dist_logits)` for positive locations only; `iou_box(mode='giou')`
        on decoded boxes for positive locations only. **Box-format warning**:
        this module's boxes are `(x_min, y_min, x_max, y_max)` throughout
        (`decode_boxes`, every assigner) while `losses.iou_box_coef`/
        `iou_box` expect `(y_min, x_min, y_max, x_max)` - swap axes before
        calling `iou_box`, a real mismatch found while cross-checking the
        two files, not present anywhere the box math stays internal to this
        file (already self-consistently `x, y`-ordered throughout).
    '''
    n_bins = reg_max + 1
    out_conv = qat_annotate_layer(Conv2D(
        n_classes + 4 * n_bins, 1, padding='same', name=f'{name}_out'))

    outputs = []
    for level_idx, feature in enumerate(fpn_features):
        level_name = f'p{level_idx}'

        feat = _independent_depthwise_sep_stem(feature, f'{name}_{level_name}', stem_layers, stem_channels)
        raw = out_conv(feat)

        cls_logits = raw[..., :n_classes]
        reg_dist_logits = tf.reshape(raw[..., n_classes:], (-1, feat.shape[1], feat.shape[2], 4, n_bins))

        outputs.append((cls_logits, reg_dist_logits))

    return outputs


# ========================= #
# Anchor points + box decoding


def generate_anchor_points(feature_shapes, strides):
    '''Builds the `(x, y)` pixel-space coordinates of every FPN location's
    "anchor point" (its receptive-field center, in FCOS's anchor-free
    framing) for every level — needed both for training-time `atss_assign_targets`
    and for inference-time `decode_boxes`.

    Input
    ----------
        feature_shapes: list of `(height, width)` per level (spatial sizes
                        of `fcos_lite_head`'s per-level outputs).
        strides: matching per-level stride (input pixels per feature-map cell).

    Output
    ----------
        A list (one entry per level) of `(H*W, 2)` numpy arrays, `(x, y)`
        pixel coordinates, row-major (matching a flattened `(H, W)` grid).
    '''
    anchor_points = []
    for (height, width), stride in zip(feature_shapes, strides):
        shifts_x = (np.arange(width, dtype=np.float32) + 0.5) * stride
        shifts_y = (np.arange(height, dtype=np.float32) + 0.5) * stride
        grid_x, grid_y = np.meshgrid(shifts_x, shifts_y)
        anchor_points.append(np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1))
    return anchor_points


def decode_boxes(anchor_points, reg_ltrb):
    '''Converts per-location `(l, t, r, b)` distances-to-box-edges back into
    absolute `(x_min, y_min, x_max, y_max)` boxes, given each location's
    anchor point.

    Input
    ----------
        anchor_points: `(N, 2)` array/tensor of `(x, y)` anchor point
                       coordinates (e.g. one level's entry from
                       `generate_anchor_points`, or all levels concatenated).
        reg_ltrb: `(..., N, 4)` tensor of `(l, t, r, b)` distances, same `N`
                  as `anchor_points`, broadcastable over any leading (e.g.
                  batch) dimensions.

    Output
    ----------
        `(..., N, 4)` tensor of `(x_min, y_min, x_max, y_max)` boxes.
    '''
    anchor_points = tf.cast(anchor_points, reg_ltrb.dtype)
    l, t, r, b = tf.unstack(reg_ltrb, 4, axis=-1)
    x, y = anchor_points[..., 0], anchor_points[..., 1]

    x_min = x - l
    y_min = y - t
    x_max = x + r
    y_max = y + b
    return tf.stack([x_min, y_min, x_max, y_max], axis=-1)


def gfl_integral_decode(reg_dist_logits, stride):
    '''Projects `gfl_lite_head`'s per-bin distribution logits down to a
    scalar per-side pixel distance - `Integral`'s own formula
    (`gfl_head.py`): `sum(softmax(logits, axis=-1) * arange(n_bins)) * stride`
    (a plain softmax-weighted expectation over bin index, not a learned
    step - no weights, hence no QAT-annotation question at all). Plain
    `tf.*` ops, safe to call directly in the functional graph (matching
    `fcos_lite_head`'s own `tf.exp(reg_raw) * float(stride)` precedent) -
    only ever used *outside* `quantize_apply`'s boundary, at inference time
    in `gfl_lite_detection` (mirroring `decode_boxes`/NMS's own placement
    in `fcos_lite_detection`). See `gfl_integral_decode_np` for the
    training-time (assigner-feeding) numpy equivalent.

    Input
    ----------
        reg_dist_logits: `(..., 4, n_bins)` tensor (e.g. `gfl_lite_head`'s
                         own per-level output).
        stride: that level's input-pixel stride.

    Output
    ----------
        `(..., 4)` tensor of `(l, t, r, b)` pixel distances - feed directly
        to `decode_boxes` alongside that level's anchor points.
    '''
    n_bins = reg_dist_logits.shape[-1]
    probs = tf.nn.softmax(reg_dist_logits, axis=-1)
    bin_values = tf.range(n_bins, dtype=probs.dtype)
    return tf.reduce_sum(probs * bin_values, axis=-1) * float(stride)


def gfl_integral_decode_np(reg_dist_logits, stride):
    '''Pure-numpy equivalent of `gfl_integral_decode`, for building the
    `pred_boxes_per_level` input `simota_assign_targets`/`dsla_assign_targets`
    need (their own cost matrices require the model's *current* decoded
    predictions - see module docstring) - intended to be called on
    `tf.stop_gradient`'d predictions inside a `tf.numpy_function`, the same
    pattern `pidnet_boundary_loss` already established for numpy-in-the-
    loss-graph computation.
    '''
    n_bins = reg_dist_logits.shape[-1]
    shifted = reg_dist_logits - np.max(reg_dist_logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    bin_values = np.arange(n_bins, dtype=probs.dtype)
    return np.sum(probs * bin_values, axis=-1) * float(stride)


# ========================= #
# ATSS label assignment (training-time, pure numpy — see module docstring)


def _box_iou_np(boxes_a, boxes_b):
    '''`(N, 4)` x `(M, 4)` -> `(N, M)` pairwise IoU, `(x_min, y_min, x_max, y_max)` boxes.'''
    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])

    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[..., 0] * wh[..., 1]

    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def atss_assign_targets(gt_boxes, gt_labels, anchor_points_per_level, strides, n_classes, topk=9,
                        anchor_scale=8.0, return_gt_indices=False):
    '''ATSS (Zhang et al. 2020) label assignment for one image — pure numpy,
    intended to be called once per training example while building a
    dataset pipeline, not as part of the Keras graph (see module docstring
    for why).

    Input
    ----------
        gt_boxes: `(n_gt, 4)` array, `(x_min, y_min, x_max, y_max)`.
        gt_labels: `(n_gt,)` array of integer class indices (`0..n_classes-1`).
        anchor_points_per_level: list of per-level `(N_l, 2)` `(x, y)`
                                 arrays, as returned by `generate_anchor_points`.
        strides: matching per-level stride, same order/length as
                 `anchor_points_per_level`.
        n_classes: total number of classes (for building the one-hot
                   classification target).
        topk: number of closest-by-center-distance candidate anchors
              selected *per level* per ground-truth box before the
              IoU-statistics threshold is applied (9, the paper's own default).
        anchor_scale: side length of each anchor's canonical square
                      "anchor box" (used only for the IoU-threshold step),
                      as a multiple of that anchor's level stride — 8x
                      stride, a common anchor-free-ATSS adaptation (the
                      paper itself uses real multi-aspect-ratio anchor
                      boxes; this single-square-box simplification is a
                      documented approximation, not the literal paper).
        return_gt_indices: `False` (default) — unchanged behavior/return
                           shape, exactly as before. `True` — appends a 5th
                           per-level list, `gt_indices` (`(N_l,)` int,
                           `-1` for negatives, the assigned ground-truth
                           box's own index for positives) — needed by
                           `assign_instance_targets` (plan.md §13.6) to know
                           *which* GT mask a positive anchor's own mask
                           target should come from, information the
                           original 4-tuple return doesn't carry (a
                           one-hot class alone is ambiguous whenever two GT
                           boxes share a class).

    Output
    ----------
        `(cls_targets, reg_targets, centerness_targets, positive_mask)` (or
        a 5-tuple with `gt_indices` appended if `return_gt_indices=True`),
        each a list of per-level numpy arrays (matching
        `anchor_points_per_level`'s per-level lengths):
            cls_targets: `(N_l, n_classes)` one-hot (all-zero for negatives).
            reg_targets: `(N_l, 4)` `(l, t, r, b)` distances (garbage/zero
                         for negatives — masked out by `positive_mask`, not
                         meant to be used un-masked).
            centerness_targets: `(N_l,)` FCOS centerness target (0 for negatives).
            positive_mask: `(N_l,)` boolean, `True` for assigned positives.
    '''
    n_levels = len(anchor_points_per_level)
    level_sizes = [len(p) for p in anchor_points_per_level]
    all_points = np.concatenate(anchor_points_per_level, axis=0)
    n_total = len(all_points)

    cls_targets = np.zeros((n_total, n_classes), dtype=np.float32)
    reg_targets = np.zeros((n_total, 4), dtype=np.float32)
    centerness_targets = np.zeros((n_total,), dtype=np.float32)
    positive_mask = np.zeros((n_total,), dtype=bool)
    gt_index_targets = np.full((n_total,), -1, dtype=np.int64)
    assigned_iou = np.full((n_total,), -1.0, dtype=np.float32)

    if len(gt_boxes) == 0:
        results = (
            _split_by_level(cls_targets, level_sizes), _split_by_level(reg_targets, level_sizes),
            _split_by_level(centerness_targets, level_sizes), _split_by_level(positive_mask, level_sizes),
        )
        if return_gt_indices:
            results = results + (_split_by_level(gt_index_targets, level_sizes),)
        return results

    level_start = np.cumsum([0] + level_sizes)

    for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
        gt_cx = (gt_box[0] + gt_box[2]) / 2.0
        gt_cy = (gt_box[1] + gt_box[3]) / 2.0

        candidate_indices = []
        for level_idx in range(n_levels):
            points = anchor_points_per_level[level_idx]
            dists = np.sqrt((points[:, 0] - gt_cx) ** 2 + (points[:, 1] - gt_cy) ** 2)
            k = min(topk, len(points))
            local_topk = np.argsort(dists)[:k]
            candidate_indices.append(local_topk + level_start[level_idx])
        candidate_indices = np.concatenate(candidate_indices, axis=0)

        candidate_points = all_points[candidate_indices]
        anchor_sizes = np.repeat(strides, level_sizes)[candidate_indices] * anchor_scale
        candidate_anchor_boxes = np.stack([
            candidate_points[:, 0] - anchor_sizes / 2.0, candidate_points[:, 1] - anchor_sizes / 2.0,
            candidate_points[:, 0] + anchor_sizes / 2.0, candidate_points[:, 1] + anchor_sizes / 2.0,
        ], axis=-1)

        ious = _box_iou_np(candidate_anchor_boxes, gt_box[None, :])[:, 0]
        iou_threshold = ious.mean() + ious.std()

        inside_box = (
            (candidate_points[:, 0] > gt_box[0]) & (candidate_points[:, 0] < gt_box[2]) &
            (candidate_points[:, 1] > gt_box[1]) & (candidate_points[:, 1] < gt_box[3])
        )
        is_positive = (ious >= iou_threshold) & inside_box

        positive_global_indices = candidate_indices[is_positive]
        positive_ious = ious[is_positive]

        for idx, iou in zip(positive_global_indices, positive_ious):
            # Conflict resolution: if another GT already claimed this anchor
            # with a higher IoU, keep that assignment instead.
            if iou <= assigned_iou[idx]:
                continue
            assigned_iou[idx] = iou
            positive_mask[idx] = True
            gt_index_targets[idx] = gt_idx

            point = all_points[idx]
            l = point[0] - gt_box[0]
            t = point[1] - gt_box[1]
            r = gt_box[2] - point[0]
            b = gt_box[3] - point[1]
            reg_targets[idx] = [l, t, r, b]

            cls_targets[idx] = 0.0
            cls_targets[idx, gt_label] = 1.0

            centerness_targets[idx] = np.sqrt(
                (min(l, r) / max(l, r, 1e-12)) * (min(t, b) / max(t, b, 1e-12))
            )

    results = (
        _split_by_level(cls_targets, level_sizes), _split_by_level(reg_targets, level_sizes),
        _split_by_level(centerness_targets, level_sizes), _split_by_level(positive_mask, level_sizes),
    )
    if return_gt_indices:
        results = results + (_split_by_level(gt_index_targets, level_sizes),)
    return results


def _split_by_level(flat_array, level_sizes):
    splits = np.cumsum(level_sizes)[:-1]
    return np.split(flat_array, splits, axis=0)


# ========================= #
# SimOTA / DSLA dynamic label assignment (plan.md §9.5/§9.6, Phase 9) -
# training-time, pure numpy, like ATSS above - but genuinely different in
# kind: both need the model's own *current* predictions (see module
# docstring's "GFL-style upgrade" section), not just box geometry.


def _box_giou_np(boxes_a, boxes_b):
    '''`(N, 4)` x `(M, 4)` -> `(N, M)` pairwise GIoU, `(x_min, y_min,
    x_max, y_max)` boxes - the same format `_box_iou_np` uses, extended the
    same way `losses.iou_box_coef`'s `'giou'` mode extends its own `'iou'`
    mode. A separate, self-contained numpy function rather than a call into
    `losses.iou_box_coef` - that function expects `(y_min, x_min, y_max,
    x_max)` boxes (see `gfl_lite_head`'s docstring's box-format warning);
    keeping this file's own box math entirely `(x, y)`-ordered avoids the
    mismatch outright rather than converting at every call site.
    '''
    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])

    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    enclose_lt = np.minimum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    enclose_rb = np.maximum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    enclose_wh = np.maximum(0.0, enclose_rb - enclose_lt)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

    return iou - np.divide(enclose_area - union, enclose_area, out=np.zeros_like(iou), where=enclose_area > 0)


def _in_box_mask_np(points, boxes):
    '''`(N, 2)` `(x, y)` points x `(M, 4)` boxes -> `(N, M)` boolean, `True`
    where a point falls strictly inside a box - shared by DSLA's candidate
    gate (`boxes = gt_boxes`) and SimOTA's own (`boxes = gt_boxes` and,
    separately, each GT's `center_radius`-scaled square).
    '''
    l = points[:, 0:1] - boxes[None, :, 0]
    t = points[:, 1:2] - boxes[None, :, 1]
    r = boxes[None, :, 2] - points[:, 0:1]
    b = boxes[None, :, 3] - points[:, 1:2]
    return np.stack([l, t, r, b], axis=-1).min(axis=-1) > 0


def _dynamic_k_matching(cost_matrix, pairwise_ious, topk):
    '''Shared by `simota_assign_targets`/`dsla_assign_targets` - dynamic-k
    matching (each GT's positive count is the clamped-`>=1` sum of its
    `topk` largest candidate IoUs, not a fixed number) followed by min-cost
    conflict resolution for any candidate multiple GTs claim - verified
    identical between `simota_assigner.py`'s and `dsl_assigner.py`'s own
    `dynamic_k_matching` methods, down to the conflict-resolution step.

    Input
    ----------
        cost_matrix, pairwise_ious: `(n_candidates, n_gt)` arrays (already
                                    restricted to whichever candidate set
                                    the caller's own gate selected).
        topk: candidate pool size for the per-GT dynamic-k sum (`candidate_topk`
              for SimOTA, `topk` for DSLA - same role, different reference
              default).

    Output
    ----------
        `(matched_gt_inds, fg_mask)` - `fg_mask` is `(n_candidates,)`
        boolean (`True` for matched/foreground candidates), `matched_gt_inds`
        is `(n_fg,)` (the GT index each foreground candidate resolves to,
        same order as `cost_matrix[fg_mask]`).
    '''
    n_candidates, n_gt = cost_matrix.shape
    matching_matrix = np.zeros_like(cost_matrix)

    k = min(topk, n_candidates)
    topk_ious = np.sort(pairwise_ious, axis=0)[::-1][:k]
    dynamic_ks = np.clip(topk_ious.sum(axis=0).astype(np.int64), 1, None)

    for gt_idx in range(n_gt):
        pos_idx = np.argsort(cost_matrix[:, gt_idx])[:dynamic_ks[gt_idx]]
        matching_matrix[pos_idx, gt_idx] = 1.0

    conflict_mask = matching_matrix.sum(axis=1) > 1
    if conflict_mask.any():
        cost_argmin = np.argmin(cost_matrix[conflict_mask], axis=1)
        matching_matrix[conflict_mask] = 0.0
        matching_matrix[conflict_mask, cost_argmin] = 1.0

    fg_mask = matching_matrix.sum(axis=1) > 0
    matched_gt_inds = matching_matrix[fg_mask].argmax(axis=1)
    return matched_gt_inds, fg_mask


def _flatten_predictions(pred_scores_per_level, pred_boxes_per_level, n_classes):
    all_scores = np.concatenate([np.reshape(s, (-1, n_classes)) for s in pred_scores_per_level], axis=0)
    all_boxes = np.concatenate([np.reshape(b, (-1, 4)) for b in pred_boxes_per_level], axis=0)
    return all_scores, all_boxes


def _empty_targets(n_total, n_classes, level_sizes):
    cls_targets = np.zeros((n_total, n_classes), dtype=np.float32)
    reg_targets = np.zeros((n_total, 4), dtype=np.float32)
    positive_mask = np.zeros((n_total,), dtype=bool)
    return (
        _split_by_level(cls_targets, level_sizes), _split_by_level(reg_targets, level_sizes),
        _split_by_level(positive_mask, level_sizes),
    )


def _write_dynamic_targets(cls_targets, reg_targets, positive_mask, all_points, valid_indices,
                           matched_gt_inds, fg_mask, pairwise_ious, gt_boxes, gt_labels):
    fg_valid_indices = valid_indices[fg_mask]
    matched_ious = pairwise_ious[fg_mask, matched_gt_inds]
    positive_mask[fg_valid_indices] = True

    for local_i, global_idx in enumerate(fg_valid_indices):
        gt_idx = matched_gt_inds[local_i]
        gt_box = gt_boxes[gt_idx]
        point = all_points[global_idx]

        reg_targets[global_idx] = [
            point[0] - gt_box[0], point[1] - gt_box[1], gt_box[2] - point[0], gt_box[3] - point[1],
        ]
        cls_targets[global_idx, gt_labels[gt_idx]] = matched_ious[local_i]


def simota_assign_targets(gt_boxes, gt_labels, anchor_points_per_level, strides,
                          pred_scores_per_level, pred_boxes_per_level, n_classes,
                          center_radius=2.5, candidate_topk=10, iou_weight=3.0, cls_weight=1.0,
                          vfl_alpha=0.75, vfl_gamma=2.0):
    '''SimOTA (PP-PicoDet v1's own assigner, `use_vfl=True` branch -
    PicoDet's own released configs use this one, not the plain-BCE
    alternative also present in the reference) label assignment for one
    image - pure numpy, but (unlike `atss_assign_targets`) needs the
    model's *current* predictions, since its cost matrix is prediction-
    dependent - see module docstring. Intended to be called from a custom
    training step, fed `tf.stop_gradient`'d current-step predictions (via
    `tf.numpy_function`), not a data pipeline.

    Input
    ----------
        gt_boxes, gt_labels: as `atss_assign_targets`.
        anchor_points_per_level, strides: as `atss_assign_targets`.
        pred_scores_per_level: list of per-level `(H, W, n_classes)` (or
                               already-flat `(N_l, n_classes)`) arrays -
                               **already sigmoid-activated probabilities**,
                               not logits (matching `quality_focal_loss`/
                               `varifocal_loss`'s own already-sigmoid
                               convention - the reference itself takes raw
                               logits and calls a `..._with_logits` BCE
                               variant; this port standardizes on
                               probabilities for consistency with the rest
                               of this repo's loss functions, a deliberate,
                               documented simplification, not an oversight).
        pred_boxes_per_level: list of per-level `(H, W, 4)`/`(N_l, 4)`
                              decoded `(x_min, y_min, x_max, y_max)` boxes -
                              e.g. `gfl_integral_decode_np` + `decode_boxes`
                              applied to the current step's raw predictions.
        n_classes: total number of classes.
        center_radius: candidate-gate square half-width, as a multiple of
                       that anchor's level stride (2.5, the reference default).
        candidate_topk: `_dynamic_k_matching`'s candidate pool size (10,
                        the reference default - `SimOTAAssigner`'s own
                        `candidate_topk`).
        iou_weight, cls_weight: cost-matrix term weights (3.0/1.0, the
                                reference defaults).
        vfl_alpha, vfl_gamma: `varifocal_loss`'s own negative-term
                              parameters, reused here for the cost's
                              classification term (0.75/2.0, matching
                              `varifocal_loss`'s own defaults).

    Output
    ----------
        `(cls_targets, reg_targets, positive_mask)`, the same per-level-list
        shape `atss_assign_targets` returns minus `centerness_targets` (GFL
        heads have none) - **`cls_targets` here is IoU-quality-weighted,
        not one-hot** (the correct target for `quality_focal_loss`/
        `varifocal_loss`).
    '''
    level_sizes = [len(p) for p in anchor_points_per_level]
    all_points = np.concatenate(anchor_points_per_level, axis=0)
    all_strides = np.repeat(strides, level_sizes).astype(np.float64)
    n_total = len(all_points)

    if len(gt_boxes) == 0:
        return _empty_targets(n_total, n_classes, level_sizes)

    all_scores, all_boxes = _flatten_predictions(pred_scores_per_level, pred_boxes_per_level, n_classes)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    gt_labels = np.asarray(gt_labels)

    is_in_gts = _in_box_mask_np(all_points, gt_boxes)

    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0

    n_gt = len(gt_boxes)
    cl = gt_cx[None, :] - center_radius * all_strides[:, None]
    ct = gt_cy[None, :] - center_radius * all_strides[:, None]
    cr = gt_cx[None, :] + center_radius * all_strides[:, None]
    cb = gt_cy[None, :] + center_radius * all_strides[:, None]
    l = all_points[:, 0:1] - cl
    t = all_points[:, 1:2] - ct
    r = cr - all_points[:, 0:1]
    b = cb - all_points[:, 1:2]
    is_in_centers = np.stack([l, t, r, b], axis=-1).min(axis=-1) > 0

    is_in_gts_or_centers = is_in_gts | is_in_centers
    valid_mask = is_in_gts_or_centers.any(axis=1)
    valid_indices = np.nonzero(valid_mask)[0]

    if len(valid_indices) == 0:
        return _empty_targets(n_total, n_classes, level_sizes)

    is_in_both = (is_in_gts & is_in_centers)[valid_indices]
    valid_boxes = all_boxes[valid_indices]
    valid_scores = all_scores[valid_indices]

    pairwise_ious = _box_iou_np(valid_boxes, gt_boxes)
    pairwise_giou = _box_giou_np(valid_boxes, gt_boxes)

    eps = 1e-7
    gt_onehot = np.zeros((n_gt, n_classes), dtype=np.float64)
    gt_onehot[np.arange(n_gt), gt_labels] = 1.0
    target = gt_onehot[None, :, :] * pairwise_ious[:, :, None]  # (n_valid, n_gt, n_classes)

    pred_b = np.clip(valid_scores, eps, 1.0 - eps)[:, None, :]
    is_positive = target > 0
    weight = np.where(is_positive, target, vfl_alpha * pred_b ** vfl_gamma)
    ce = -(target * np.log(pred_b) + (1.0 - target) * np.log(1.0 - pred_b))
    cls_cost = (weight * ce).sum(axis=-1)  # (n_valid, n_gt)

    giou_cost = 1.0 - pairwise_giou
    cost_matrix = cls_cost * cls_weight + giou_cost * iou_weight + (~is_in_both).astype(np.float64) * 1e8

    matched_gt_inds, fg_mask = _dynamic_k_matching(cost_matrix, pairwise_ious, candidate_topk)

    cls_targets, reg_targets, positive_mask = (
        np.zeros((n_total, n_classes), dtype=np.float32), np.zeros((n_total, 4), dtype=np.float32),
        np.zeros((n_total,), dtype=bool),
    )
    _write_dynamic_targets(cls_targets, reg_targets, positive_mask, all_points, valid_indices,
                           matched_gt_inds, fg_mask, pairwise_ious, gt_boxes, gt_labels)

    return (
        _split_by_level(cls_targets, level_sizes), _split_by_level(reg_targets, level_sizes),
        _split_by_level(positive_mask, level_sizes),
    )


def dsla_assign_targets(gt_boxes, gt_labels, anchor_points_per_level, strides,
                        pred_scores_per_level, pred_boxes_per_level, n_classes,
                        topk=13, iou_factor=3.0):
    '''DSLA (Dynamic Soft Label Assigner, NanoDet-Plus's own assigner)
    label assignment for one image - pure numpy, same "needs the model's
    current predictions" caveat as `simota_assign_targets` (see its
    docstring and the module docstring's "GFL-style upgrade" section).

    The two real differences from `simota_assign_targets` (verified from
    source, not assumed symmetric - both assigners share the same
    `_dynamic_k_matching` conflict-resolution step, but differ here):
    **candidate gate** - strictly inside the GT box only, no center-region
    step at all (simpler than SimOTA's box-or-center-square union); **cost
    formula** - `BCE(pred_score, soft_label) * |soft_label - pred_score|^2`
    (soft label = `one_hot(gt_label) * iou`, applied across *every* class
    channel, not just the true one - DSLA's actual "soft label" mechanism)
    `+ (-log(iou + eps)) * iou_factor`, with no `+1e8` term (no center-region
    gate to penalize against).

    Input
    ----------
        (see `simota_assign_targets` for shared arguments)
        topk: `_dynamic_k_matching`'s candidate pool size (13, the
              reference's own default - `DynamicSoftLabelAssigner`'s `topk`).
        iou_factor: regression/IoU cost term weight (3.0, the reference default).

    Output
    ----------
        Same `(cls_targets, reg_targets, positive_mask)` shape as
        `simota_assign_targets`.
    '''
    level_sizes = [len(p) for p in anchor_points_per_level]
    all_points = np.concatenate(anchor_points_per_level, axis=0)
    n_total = len(all_points)

    if len(gt_boxes) == 0:
        return _empty_targets(n_total, n_classes, level_sizes)

    all_scores, all_boxes = _flatten_predictions(pred_scores_per_level, pred_boxes_per_level, n_classes)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    gt_labels = np.asarray(gt_labels)
    n_gt = len(gt_boxes)

    is_in_gts = _in_box_mask_np(all_points, gt_boxes)
    valid_mask = is_in_gts.any(axis=1)
    valid_indices = np.nonzero(valid_mask)[0]

    if len(valid_indices) == 0:
        return _empty_targets(n_total, n_classes, level_sizes)

    valid_boxes = all_boxes[valid_indices]
    valid_scores = all_scores[valid_indices]

    pairwise_ious = _box_iou_np(valid_boxes, gt_boxes)
    eps = 1e-7
    iou_cost = -np.log(pairwise_ious + eps)

    gt_onehot = np.zeros((n_gt, n_classes), dtype=np.float64)
    gt_onehot[np.arange(n_gt), gt_labels] = 1.0
    soft_label = gt_onehot[None, :, :] * pairwise_ious[:, :, None]  # (n_valid, n_gt, n_classes)

    pred_b = np.clip(valid_scores, eps, 1.0 - eps)[:, None, :]
    scale_factor = np.abs(soft_label - pred_b)
    bce = -(soft_label * np.log(pred_b) + (1.0 - soft_label) * np.log(1.0 - pred_b))
    cls_cost = (bce * scale_factor ** 2).sum(axis=-1)  # (n_valid, n_gt)

    cost_matrix = cls_cost + iou_cost * iou_factor

    matched_gt_inds, fg_mask = _dynamic_k_matching(cost_matrix, pairwise_ious, topk)

    cls_targets, reg_targets, positive_mask = (
        np.zeros((n_total, n_classes), dtype=np.float32), np.zeros((n_total, 4), dtype=np.float32),
        np.zeros((n_total,), dtype=bool),
    )
    _write_dynamic_targets(cls_targets, reg_targets, positive_mask, all_points, valid_indices,
                           matched_gt_inds, fg_mask, pairwise_ious, gt_boxes, gt_labels)

    return (
        _split_by_level(cls_targets, level_sizes), _split_by_level(reg_targets, level_sizes),
        _split_by_level(positive_mask, level_sizes),
    )


# ========================= #
# Lightweight instance segmentation (plan.md §13.6, Phase 11) - YOLACT-style
# (Bolya, Zhou, Xiao, Lee, "YOLACT: Real-time Instance Segmentation", ICCV
# 2019, arXiv:1904.02689) prototypes + per-box mask coefficients, attached
# to the existing `fcos_lite_head`/`pafpn_lite_neck` infrastructure -
# FCOS-lite family only this pass (GFL-lite instance segmentation is
# structurally similar but a separate, later addition, matching this
# repo's own "one detector family at a time" precedent). ATSS-only
# assignment this pass (SimOTA/DSLA-based instance assignment is a real,
# separate follow-up). Chosen over CondInst/SOLOv2-style dynamic
# per-instance conv-filter generation specifically for TFLite int8/
# Axis-ACAP-ARTPEC-8 export reasons - see plan.md §13.6 for the full
# comparison.
#
# **De-risked via a standalone spike before this feature was built**: the
# NMS-survivor-gather chain below (`_indexed_nms_gather`) was verified,
# built on a real `fcos_lite_detection_base`, to convert cleanly to int8
# TFLite under the strict `TFLITE_BUILTINS_INT8` op set with zero
# `SELECT_TF_OPS` fallback - `NON_MAX_SUPPRESSION_V4`/`GATHER`/`TOPK_V2`
# all present as standard builtin ops in the converted model, which then
# ran correctly through a real `tf.lite.Interpreter`. What that spike does
# NOT establish: whether Axis's own ARTPEC-8 DLPU delegate specifically
# accelerates those ops versus falling back to CPU for them - a narrower,
# still-open follow-up needing Axis's own hardware docs or on-device
# testing, not a blocker to building the feature's software side.


def _prototype_branch(finest_feature, k=32, proto_channels=64, name='proto'):
    '''YOLACT's own prototype branch - a small FCN producing `k` shared,
    dense "prototype" masks over the whole image (not per-instance - every
    detection's own mask is later assembled as a linear combination of
    these same `k` prototypes, weighted by that detection's own mask
    coefficients). Tapped off the neck's shallowest/finest level (stride 8
    for this repo's own `pafpn_lite_neck`) and upsampled once more, landing
    at stride 4 - YOLACT's own "1/4 input resolution" prototype convention,
    verified from source, not assumed. `k=32` is the paper's own default.

    Input
    ----------
        finest_feature: the neck's shallowest-level output (e.g.
                        `pafpn_lite_neck(...)`'s own `out3`).
        k: number of prototype masks.
        proto_channels: hidden channel width of this branch's own convs.
        name: prefix of the created Keras layers.

    Output
    ----------
        A `(B, H/4, W/4, k)` tensor (assuming a stride-8 `finest_feature`) -
        raw, pre-sigmoid prototype logits (sigmoid applied at mask-assembly
        time, in the full model, alongside the mask-coefficient combination
        - see `fcos_lite_instance_segmentation`).
    '''
    x = _conv_bn_act(finest_feature, proto_channels, kernel_size=3, name=f'{name}_conv1')
    x = qat_annotate_layer(UpSampling2D(size=2, interpolation='bilinear', name=f'{name}_upsample'))(x)
    x = _conv_bn_act(x, proto_channels, kernel_size=3, name=f'{name}_conv2')
    return qat_annotate_layer(Conv2D(k, 1, padding='same', name=f'{name}_out'))(x)


def fcos_lite_instance_head(fpn_features, n_classes, k=32, strides=(8, 16, 32), stem_channels=96,
                            stem_layers=2, name='fcos_instance_head'):
    '''`fcos_lite_head` plus one added branch (`k`-channel, `tanh`-activated
    mask coefficients, YOLACT's own choice - "to enable subtraction"
    between prototypes, confirmed from source) - **a new function, not a
    modification of `fcos_lite_head` itself**, the same discipline PIDNet's
    own `_pidnet_segmenthead` already established (a new function once
    verified to genuinely differ from DDRNet's `segmenthead`, not a
    retrofit) - `fcos_lite_head`/`fcos_lite_detection_base`/
    `fcos_lite_detection` stay untouched, already-shipped and
    already-tested. Otherwise identical: same per-level independent stems,
    same `tfmot` QAT-cloning reasoning (see `fcos_lite_head`'s own
    docstring for the full account), same shared final 1x1 output convs,
    same `_qat_channelwise_scale`-based per-level regression scale.

    Input
    ----------
        (see `fcos_lite_head` for `fpn_features`/`n_classes`/`strides`/
        `stem_channels`/`stem_layers`)
        k: number of mask coefficients per box - must match
           `_prototype_branch`'s own `k`.

    Output
    ----------
        A list of 3 `(cls_logits, reg_ltrb, centerness, mask_coefs)`
        tuples, one per input level - the first three exactly as
        `fcos_lite_head`'s own output; `mask_coefs` shape `(B, H, W, k)`,
        `tanh`-activated (already bounded to `[-1, 1]`, no further
        activation needed before mask assembly).
    '''
    cls_head_conv = qat_annotate_layer(
        Conv2D(n_classes, 1, padding='same', name=f'{name}_cls_out'))
    reg_head_conv = qat_annotate_layer(
        Conv2D(4, 1, padding='same', name=f'{name}_reg_out'))
    centerness_head_conv = qat_annotate_layer(
        Conv2D(1, 1, padding='same', name=f'{name}_centerness_out'))
    mask_coef_conv = qat_annotate_layer(
        Conv2D(k, 1, padding='same', activation='tanh', name=f'{name}_mask_coef_out'))

    outputs = []
    for level_idx, (feature, stride) in enumerate(zip(fpn_features, strides)):
        level_name = f'p{level_idx}'

        cls_feat = _independent_depthwise_sep_stem(feature, f'{name}_cls_{level_name}', stem_layers, stem_channels)
        cls_logits = cls_head_conv(cls_feat)

        reg_feat = _independent_depthwise_sep_stem(feature, f'{name}_reg_{level_name}', stem_layers, stem_channels)
        reg_raw = reg_head_conv(reg_feat)
        reg_raw = _qat_channelwise_scale(reg_raw, name=f'{name}_{level_name}')
        reg_ltrb = tf.exp(reg_raw) * float(stride)

        centerness = centerness_head_conv(reg_feat)
        mask_coefs = mask_coef_conv(reg_feat)

        outputs.append((cls_logits, reg_ltrb, centerness, mask_coefs))

    return outputs


def _resize_mask_np(mask, height, width):
    '''A single `(H, W)` binary/float mask -> `(height, width)`, via
    `cv2.resize` (already an established dependency of this repo, used
    identically by `losses.py`'s own Canny-based boundary-map helpers) -
    linear interpolation, giving a soft/anti-aliased mask target rather
    than a hard-thresholded one.
    '''
    return cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def assign_instance_targets(gt_boxes, gt_labels, gt_masks, anchor_points_per_level, strides, n_classes,
                            input_height, input_width, prototype_height, prototype_width,
                            topk=9, anchor_scale=8.0):
    '''Extends `atss_assign_targets` with per-instance mask targets - calls
    it unchanged (via its own new `return_gt_indices=True` opt-in, added
    this pass) for the box/class assignment, then for every positive
    anchor, attaches that anchor's own assigned ground-truth box's mask
    (resized to prototype resolution) and that box's own area in
    prototype-resolution units (for `losses.yolact_mask_loss`'s own area
    normalization) - both precomputed once per GT box, not once per
    positive anchor.

    Input
    ----------
        gt_boxes, gt_labels, anchor_points_per_level, strides, n_classes,
        topk, anchor_scale: as `atss_assign_targets`.
        gt_masks: `(n_gt, H, W)` array, one binary mask per ground-truth
                  box (any `H, W` - resized internally).
        input_height, input_width: the model's own input spatial size -
            needed to convert each GT box's pixel-space area (`gt_boxes`
            is in the same input-pixel coordinate space
            `anchor_points_per_level` already is) into prototype-resolution
            units.
        prototype_height, prototype_width: `_prototype_branch`'s own output
            spatial size.

    Output
    ----------
        `atss_assign_targets`'s own 4-tuple (`cls_targets`, `reg_targets`,
        `centerness_targets`, `positive_mask`) plus two more per-level
        lists: `mask_targets` (`(N_l, prototype_height, prototype_width)`,
        zero for negatives) and `mask_areas` (`(N_l,)`, each positive
        anchor's assigned GT box's own area in prototype-resolution units,
        floored at `1.0` to keep `yolact_mask_loss`'s division well-defined;
        zero/unused for negatives).
    '''
    cls_targets, reg_targets, centerness_targets, positive_mask, gt_indices = atss_assign_targets(
        gt_boxes, gt_labels, anchor_points_per_level, strides, n_classes,
        topk=topk, anchor_scale=anchor_scale, return_gt_indices=True,
    )

    level_sizes = [len(p) for p in anchor_points_per_level]
    mask_targets = [np.zeros((n, prototype_height, prototype_width), dtype=np.float32) for n in level_sizes]
    mask_areas = [np.zeros((n,), dtype=np.float32) for n in level_sizes]

    if len(gt_boxes) == 0:
        return cls_targets, reg_targets, centerness_targets, positive_mask, mask_targets, mask_areas

    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    resized_gt_masks = np.stack([
        _resize_mask_np(m, prototype_height, prototype_width) for m in gt_masks
    ])

    scale_x = prototype_width / float(input_width)
    scale_y = prototype_height / float(input_height)
    gt_areas = np.array([
        max(0.0, (gt_boxes[i, 2] - gt_boxes[i, 0]) * scale_x) *
        max(0.0, (gt_boxes[i, 3] - gt_boxes[i, 1]) * scale_y)
        for i in range(len(gt_boxes))
    ], dtype=np.float32)
    gt_areas = np.maximum(gt_areas, 1.0)

    for level_idx in range(len(level_sizes)):
        level_gt_indices = gt_indices[level_idx]
        positive_local = np.where(level_gt_indices >= 0)[0]
        for local_idx in positive_local:
            gt_idx = level_gt_indices[local_idx]
            mask_targets[level_idx][local_idx] = resized_gt_masks[gt_idx]
            mask_areas[level_idx][local_idx] = gt_areas[gt_idx]

    return cls_targets, reg_targets, centerness_targets, positive_mask, mask_targets, mask_areas


def _indexed_nms_gather(boxes, scores, extra_per_box_tensors, n_classes, max_output_size_per_class,
                        max_detections, iou_threshold, score_threshold):
    '''Per-class indexed NMS + gather, replacing
    `tf.image.combined_non_max_suppression` wherever per-box data (e.g.
    mask coefficients) needs to survive suppression alongside its own box -
    `combined_non_max_suppression` returns no selection indices, so nothing
    else can be gathered onto its own survivors (the same gap this repo's
    own `layer_utils.add_energy_score_output` already ran into and stopped
    short of solving for a per-box Energy Score). Verified against
    YOLACT's own reference `fast_nms` - it solves this identical problem
    the same way: its own custom NMS computes an index/keep tensor and
    applies identical indexing to boxes, scores, and mask coefficients in
    lockstep, rather than using a black-box combined-NMS op.

    Uses `tf.image.non_max_suppression_padded(...,
    pad_to_max_output_size=True)` (unlike `combined_non_max_suppression`,
    this *does* return selection indices) per class, `tf.gather`s every
    extra per-box tensor through those same indices, then a final
    cross-class `tf.nn.top_k` (gathering the extras through *that* index
    tensor too).

    **De-risked via a standalone spike** (plan.md §13.6): this exact chain,
    built on a real `fcos_lite_detection_base` output, converts cleanly to
    int8 TFLite under the strict `TFLITE_BUILTINS_INT8` op set with zero
    `SELECT_TF_OPS` fallback (`NON_MAX_SUPPRESSION_V4`/`GATHER`/`TOPK_V2`
    all present as standard builtin ops in the converted model).

    **Single image only (no batch dimension) - the caller's
    responsibility, not handled here.** `non_max_suppression_padded` has
    no native batched form; batching this (e.g. via `tf.map_fn`) is a
    real, separate, unverified risk (a `WHILE` op in TFLite) the spike did
    not test - `fcos_lite_instance_segmentation`'s own docstring documents
    this as an explicit, deliberate batch-size-1 limitation on its full
    (inference) model.

    Input
    ----------
        boxes: `(N, 4)` `(x_min, y_min, x_max, y_max)`.
        scores: `(N, n_classes)`.
        extra_per_box_tensors: dict of `{name: (N, ...)}` tensors to gather
            in lockstep with `boxes`/`scores` (e.g. `{'mask_coefs': (N, k)}`).
        n_classes, max_output_size_per_class, max_detections, iou_threshold,
        score_threshold: as their `fcos_lite_detection`-side namesakes.

    Output
    ----------
        `(boxes, scores, labels, extras)` - `boxes` `(max_detections, 4)`,
        `scores`/`labels` `(max_detections,)`, `extras` a dict matching
        `extra_per_box_tensors`'s own keys, each `(max_detections, ...)`.
        Slots beyond however many real detections were found are
        zero-padded (matching `non_max_suppression_padded`'s own
        `pad_to_max_output_size` convention), with score `0.0` marking an
        unused slot - no separate `valid_detections` count is returned
        (derive it as `tf.reduce_sum(tf.cast(scores > 0, tf.int32))` if
        needed, the same information, not duplicated here).
    '''
    per_class_boxes, per_class_scores, per_class_labels = [], [], []
    per_class_extras = {name: [] for name in extra_per_box_tensors}

    for c in range(n_classes):
        class_scores = scores[:, c]
        selected_indices, num_valid = tf.image.non_max_suppression_padded(
            boxes, class_scores, max_output_size=max_output_size_per_class,
            iou_threshold=iou_threshold, score_threshold=score_threshold,
            pad_to_max_output_size=True,
        )
        gathered_scores = tf.gather(class_scores, selected_indices)
        valid_mask = tf.cast(tf.range(max_output_size_per_class) < num_valid, gathered_scores.dtype)
        gathered_scores = gathered_scores * valid_mask

        per_class_boxes.append(tf.gather(boxes, selected_indices))
        per_class_scores.append(gathered_scores)
        per_class_labels.append(tf.fill([max_output_size_per_class], c))
        for extra_name, tensor in extra_per_box_tensors.items():
            per_class_extras[extra_name].append(tf.gather(tensor, selected_indices))

    all_boxes = tf.concat(per_class_boxes, axis=0)
    all_scores = tf.concat(per_class_scores, axis=0)
    all_labels = tf.concat(per_class_labels, axis=0)
    all_extras = {name: tf.concat(tensors, axis=0) for name, tensors in per_class_extras.items()}

    top_scores, top_indices = tf.nn.top_k(all_scores, k=max_detections)
    top_boxes = tf.gather(all_boxes, top_indices)
    top_labels = tf.gather(all_labels, top_indices)
    top_extras = {name: tf.gather(tensor, top_indices) for name, tensor in all_extras.items()}

    return top_boxes, top_scores, top_labels, top_extras


def _indexed_nms_gather_batch1(boxes, scores, n_classes, max_output_size_per_class,
                               max_detections, iou_threshold, score_threshold, name=None):
    '''Batch-size-1-only replacement for `tf.image.combined_non_max_suppression`,
    for a *dynamic*-batch `input_tensor` (the normal `Input((H, W, 3))`
    every builder in this repo otherwise uses) - the default NMS path for
    `fcos_lite_detection`/`gfl_lite_detection`'s own full models (see
    `_nms_dispatch`). Same `_AssertBatchSizeOne` runtime guard
    `fcos_lite_instance_segmentation` already established (raises a clear
    `InvalidArgumentError` on a batch>1 call, rather than silently only
    processing the first image).

    Output matches `combined_non_max_suppression`'s own convention exactly
    (`classes` as `float32`, not the `int32` `_indexed_nms_gather` itself
    uses) - see `_nms_dispatch`'s own docstring for the full "why this
    exists" account (a real, confirmed `tf.CombinedNonMaxSuppression`
    TFLite-export bug, not a design change).
    '''
    boxes = _AssertBatchSizeOne(name=name)(boxes)
    top_boxes, top_scores, top_labels, _ = _indexed_nms_gather(
        boxes[0], scores[0], {}, n_classes, max_output_size_per_class,
        max_detections, iou_threshold, score_threshold,
    )
    top_boxes = top_boxes[tf.newaxis, ...]
    top_scores = top_scores[tf.newaxis, ...]
    top_labels = tf.cast(top_labels, tf.float32)[tf.newaxis, ...]
    valid_detections = tf.reduce_sum(tf.cast(top_scores > 0, tf.int32), axis=-1)
    return top_boxes, top_scores, top_labels, valid_detections


def _indexed_nms_gather_unrolled(boxes, scores, batch_size, n_classes, max_output_size_per_class,
                                 max_detections, iou_threshold, score_threshold):
    '''Fixed-batch-size replacement for `tf.image.combined_non_max_suppression`,
    for a *static*-batch `input_tensor` (built via `Input(batch_shape=
    (N, H, W, 3))` instead of the usual dynamic-batch `Input((H, W, 3))`) -
    the opt-in, real-multi-image-batching NMS path for `fcos_lite_detection`/
    `gfl_lite_detection`'s own full models (see `_nms_dispatch`), for
    deployment scenarios where batching genuinely helps (e.g. serving many
    images per inference call through Triton or similar), at the cost of
    the exported model being locked to that one batch size.

    **A `tf.map_fn`-based dynamic-batch version was tried first and found
    to crash the TFLite converter outright** (`SIGSEGV`, not a catchable
    Python exception) when converting a real, full-depth QAT-quantized
    backbone under strict `TFLITE_BUILTINS_INT8` - confirmed via a
    from-scratch isolation, independent of this file's own complexity:
    reproduces identically on two different backbones (`MobileOne-S0`,
    `MobileNetV2`) and both the MLIR and legacy TFLite converter paths;
    does *not* reproduce on a small synthetic QAT model (a handful of
    layers) with the exact same `tf.map_fn` wrapping, isolating the
    trigger to "large/deep QAT-quantized graph feeding a `WHILE`-loop-based
    op under strict int8" specifically, not `tf.map_fn`/`_indexed_nms_gather`
    in general. That code (`_BatchedIndexedNms`/`_indexed_nms_gather_batched`,
    both since removed, not merely deprecated - too dangerous to leave in
    a shipped codebase where a future caller could reach for it) is a real,
    reportable finding about this repo's pinned `tensorflow`/`tf_keras`/
    `tensorflow_model_optimization` versions, not something fixable from
    application code.

    This function instead unrolls the batch dimension as a **plain Python
    `for` loop at graph-construction time** - `batch_size` independent
    `_indexed_nms_gather` calls (one per image, no loop *construct* in the
    resulting graph at all, so no `WHILE` op, so nothing for the crash
    above to trigger on), stacked back together via `tf.stack`. Verified
    on a real, full-depth `MobileOne-S0`-backed detector: converts cleanly
    under strict `TFLITE_BUILTINS_INT8` with zero Flex/custom ops,
    `NON_MAX_SUPPRESSION_V4`/`GATHER`/`TOPK_V2` all present as standard
    builtins.

    Input
    ----------
        boxes: `(batch_size, N, 4)`, scores: `(batch_size, N, n_classes)` -
            both must have a *static*, build-time-known leading dimension
            equal to `batch_size` (guaranteed by `_nms_dispatch`'s own
            dispatch condition, not re-checked here).
        batch_size: the static batch size (a plain Python `int`, not a
            tensor) - how many times to unroll the loop.
        n_classes, max_output_size_per_class, max_detections, iou_threshold,
        score_threshold: as `_indexed_nms_gather`.

    Output
    ----------
        `(boxes, scores, classes, valid_detections)`, matching
        `combined_non_max_suppression`'s own convention exactly (`classes`
        as `float32`) - `boxes` `(batch_size, max_detections, 4)`,
        `scores`/`classes` `(batch_size, max_detections)`,
        `valid_detections` `(batch_size,)`.
    '''
    per_image_boxes, per_image_scores, per_image_labels = [], [], []
    for i in range(batch_size):
        top_boxes, top_scores, top_labels, _ = _indexed_nms_gather(
            boxes[i], scores[i], {}, n_classes, max_output_size_per_class,
            max_detections, iou_threshold, score_threshold,
        )
        per_image_boxes.append(top_boxes)
        per_image_scores.append(top_scores)
        per_image_labels.append(top_labels)

    nms_boxes = tf.stack(per_image_boxes, axis=0)
    nms_scores = tf.stack(per_image_scores, axis=0)
    nms_classes = tf.cast(tf.stack(per_image_labels, axis=0), tf.float32)
    valid_detections = tf.reduce_sum(tf.cast(nms_scores > 0, tf.int32), axis=-1)
    return nms_boxes, nms_scores, nms_classes, valid_detections


def _nms_dispatch(input_tensor, boxes, scores, n_classes, max_output_size_per_class,
                  max_detections, iou_threshold, score_threshold, name=None):
    '''Replaces `tf.image.combined_non_max_suppression` in
    `fcos_lite_detection`/`gfl_lite_detection`'s own full models - fixing a
    real, confirmed limitation found while verifying an unrelated fix
    (plan.md §13.6's own `ScaleLayer` write-up) on the already-shipped
    `fcos_lite_detection`: `combined_non_max_suppression` does not convert
    under strict `TFLITE_BUILTINS_INT8` export at all (`tf.
    CombinedNonMaxSuppression` "is neither a custom op nor a flex op",
    `ERROR_NEEDS_FLEX_OPS`) - not a hypothetical, this repo's own
    Axis-ACAP-targeted int8 export path could never actually produce a
    working `.tflite` for the full (NMS'd) detection models before this.

    **Dispatches on `input_tensor`'s own batch dimension, not a separate
    parameter** - matching every other builder's own "the caller already
    built `input_tensor`" convention in this file:
    - **Dynamic batch** (`Input((H, W, 3))`, the default/usual case,
      `input_tensor.shape[0] is None`) -> `_indexed_nms_gather_batch1`
      (batch-size-1 only, runtime-asserted) - matches
      `fcos_lite_instance_segmentation`'s own already-shipped precedent,
      and the actual Axis deployment reality (one camera frame at a time).
    - **Static batch** (`Input(batch_shape=(N, H, W, 3))`, an explicit
      opt-in) -> `_indexed_nms_gather_unrolled` (real multi-image
      batching, no dynamic-batch-size flexibility once built/exported) -
      for deployment scenarios where batching genuinely helps (e.g.
      serving many images per call through Triton or similar).

    See `_indexed_nms_gather_unrolled`'s own docstring for why a
    `tf.map_fn`-based dynamic-batch alternative isn't used instead (it
    crashes the TFLite converter on a real backbone).
    '''
    static_batch = input_tensor.shape[0]
    if static_batch is not None:
        return _indexed_nms_gather_unrolled(
            boxes, scores, static_batch, n_classes, max_output_size_per_class,
            max_detections, iou_threshold, score_threshold,
        )
    return _indexed_nms_gather_batch1(
        boxes, scores, n_classes, max_output_size_per_class, max_detections,
        iou_threshold, score_threshold, name=name,
    )


# ========================= #
# Model builders


def fcos_lite_detection_base(input_tensor, n_classes, feature_levels=(3, 4, 5), fpn_channels=96,
                             stem_channels=96, stem_layers=2, backbone='MobileNetV2', weights='imagenet',
                             freeze_backbone=True, freeze_batch_norm=True, name='fcos_lite'):
    '''Backbone -> PAFPN-lite neck -> FCOS-lite head, raw per-level
    predictions (no NMS/decoding) — the "base" half of the `base.py`/
    `models.py` split every other model in this plan uses.

    Input
    ----------
        input_tensor: the input tensor, e.g. `keras.layers.Input((H, W, 3))`.
        n_classes: number of object classes.
        feature_levels: 1-indexed encoder feature levels (P3, P4, P5) — the
                        same convention `pp_liteseg` uses; default `(3, 4,
                        5)` for a 5-tap backbone, `(2, 3, 4)` for a 4-tap one
                        (RepViT/FastViT).
        fpn_channels: `pafpn_lite_neck`'s common output width.
        stem_channels, stem_layers: `fcos_lite_head`'s shared-stem config.
        backbone, weights, freeze_backbone, freeze_batch_norm: as every
            other model in this plan (`backbone_zoo`-based).
        name: prefix of the created Keras model and its layers.

    Output
    ----------
        A `tf.keras.Model` with a `.preprocessing` attribute and a
        `.strides` attribute (the 3 levels' input-pixel strides, needed by
        `atss_assign_targets`/`decode_boxes`), outputting a flat 9-tensor
        list `[cls_p3, reg_p3, centerness_p3, cls_p4, reg_p4, centerness_p4,
        cls_p5, reg_p5, centerness_p5]` (flattened rather than nested, so
        every output is a plain `Model` output tensor).
    '''
    backbone_ = backbone_zoo(
        backbone, weights, input_tensor, max(feature_levels), freeze_backbone, freeze_batch_norm)
    x_encoder = backbone_([input_tensor, ])

    features = [x_encoder[i - 1] for i in feature_levels]
    fpn_features = pafpn_lite_neck(features, out_channels=fpn_channels, name='{}_pafpn'.format(name))

    strides = [int(input_tensor.shape[1] // f.shape[1]) for f in fpn_features]
    head_outputs = fcos_lite_head(
        fpn_features, n_classes, strides=strides, stem_channels=stem_channels,
        stem_layers=stem_layers, name='{}_head'.format(name),
    )

    flat_outputs = []
    for cls_logits, reg_ltrb, centerness in head_outputs:
        flat_outputs.extend([cls_logits, reg_ltrb, centerness])

    m = Model([input_tensor, ], flat_outputs)

    with tfmot.quantization.keras.quantize_scope():
        m = tfmot.quantization.keras.quantize_apply(m)

    m.preprocessing = backbone_.preprocessing
    m.strides = strides

    return m


def fcos_lite_detection(input_tensor, n_classes, feature_levels=(3, 4, 5), fpn_channels=96,
                        stem_channels=96, stem_layers=2, backbone='MobileNetV2', weights='imagenet',
                        freeze_backbone=True, freeze_batch_norm=True, score_threshold=0.05,
                        iou_threshold=0.5, max_detections=100, name='fcos_lite'):
    '''Full FCOS-lite detector: `fcos_lite_detection_base` plus box
    decoding and `_nms_dispatch`-based NMS, for inference use.

    **NMS mechanism changed from `tf.image.combined_non_max_suppression`
    to `_nms_dispatch`** — a real, confirmed bug fix, not a design change:
    `combined_non_max_suppression` does not convert under strict
    `TFLITE_BUILTINS_INT8` export at all (`tf.CombinedNonMaxSuppression`
    "is neither a custom op nor a flex op", `ERROR_NEEDS_FLEX_OPS`) —
    found while verifying an unrelated fix (plan.md §13.6's own
    `ScaleLayer` write-up). See `_nms_dispatch`'s own docstring for the
    full account, including **a real batching capability change**: the
    old `combined_non_max_suppression` natively supported any batch size;
    its replacement dispatches on `input_tensor`'s own batch dimension —
    a normal dynamic-batch `Input((H, W, 3))` (the default, matching
    every other builder in this file) now gets **batch-size-1-only**
    NMS (runtime-asserted, matching `fcos_lite_instance_segmentation`'s
    own precedent), while a static-batch `Input(batch_shape=(N, H, W, 3))`
    gets real, fixed-size multi-image batching instead (opt-in, e.g. for
    Triton-style serving) — a `tf.map_fn`-based dynamic-batch alternative
    was tried first and found to crash the TFLite converter outright on a
    real backbone, so this is not a design preference, it's the only
    verified-working option. Output shape/dtype/padding convention is
    otherwise unchanged from the old `combined_non_max_suppression`-based
    version.

    Input
    ----------
        (see `fcos_lite_detection_base` for the shared arguments)
        score_threshold: minimum `sqrt(classification_score * centerness)`
                         (FCOS's own inference-time score combination) for a
                         detection to be considered at all.
        iou_threshold: NMS IoU threshold.
        max_detections: maximum detections returned per image (also used as
                        `_nms_dispatch`'s own `max_output_size_per_class`,
                        matching `combined_non_max_suppression`'s own
                        former `max_output_size_per_class=max_total_size=
                        max_detections` call convention).

    Output
    ----------
        A `tf.keras.Model` with a `.preprocessing` attribute, producing
        `(boxes, scores, classes, valid_detections)` — `boxes` shape
        `(B, max_detections, 4)` (`x_min, y_min, x_max, y_max`, input-pixel
        units), `scores`/`classes` shape `(B, max_detections)`,
        `valid_detections` shape `(B,)` (how many of the `max_detections`
        slots are real — matching `combined_non_max_suppression`'s own
        former convention exactly, the rest zero-padded; `B` is `1` unless
        `input_tensor` was built with a static batch dimension — see
        `_nms_dispatch`).
    '''
    base_model = fcos_lite_detection_base(
        input_tensor, n_classes, feature_levels=feature_levels, fpn_channels=fpn_channels,
        stem_channels=stem_channels, stem_layers=stem_layers, backbone=backbone, weights=weights,
        freeze_backbone=freeze_backbone, freeze_batch_norm=freeze_batch_norm, name=name,
    )

    raw_outputs = base_model(input_tensor)
    strides = base_model.strides
    feature_shapes = [(int(raw_outputs[i * 3].shape[1]), int(raw_outputs[i * 3].shape[2])) for i in range(3)]
    anchor_points_per_level = generate_anchor_points(feature_shapes, strides)

    all_boxes, all_scores = [], []
    for level_idx in range(3):
        cls_logits, reg_ltrb, centerness = raw_outputs[level_idx * 3: level_idx * 3 + 3]

        height, width = feature_shapes[level_idx]
        anchor_points = tf.constant(anchor_points_per_level[level_idx], dtype=tf.float32)

        cls_scores = tf.reshape(tf.sigmoid(cls_logits), (-1, height * width, n_classes))
        centerness_scores = tf.reshape(tf.sigmoid(centerness), (-1, height * width, 1))
        combined_scores = tf.sqrt(cls_scores * centerness_scores)

        reg_flat = tf.reshape(reg_ltrb, (-1, height * width, 4))
        boxes = decode_boxes(anchor_points, reg_flat)

        all_boxes.append(boxes)
        all_scores.append(combined_scores)

    boxes = tf.concat(all_boxes, axis=1)
    scores = tf.concat(all_scores, axis=1)

    nms_boxes, nms_scores, nms_classes, valid_detections = _nms_dispatch(
        input_tensor, boxes, scores, n_classes, max_output_size_per_class=max_detections,
        max_detections=max_detections, iou_threshold=iou_threshold, score_threshold=score_threshold,
        name='{}_nms'.format(name),
    )

    m = Model([input_tensor, ], [nms_boxes, nms_scores, nms_classes, valid_detections])
    m.preprocessing = base_model.preprocessing

    return m


# ========================= #
# GFL-style model builders (plan.md §9.5/§9.6, Phase 9) - a second,
# independent detector in this same file, mirroring `fcos_lite_detection*`'s
# own base/full split exactly. See module docstring's "GFL-style upgrade"
# section for the full architectural account.


def gfl_lite_detection_base(input_tensor, n_classes, feature_levels=(3, 4, 5), neck_channels=96,
                            stem_channels=96, stem_layers=2, reg_max=7, num_extra_levels=0,
                            backbone='MobileNetV2', weights='imagenet', freeze_backbone=True,
                            freeze_batch_norm=True, name='gfl_lite'):
    '''Backbone -> `ghost_pan_lite_neck` -> `gfl_lite_head`, raw per-level
    predictions (no decoding/NMS) - the "base" half of the `base.py`/
    `models.py` split every other model in this plan uses, exactly mirroring
    `fcos_lite_detection_base`'s own structure and argument conventions
    (`backbone`/`weights`/`freeze_backbone`/`freeze_batch_norm`/
    `feature_levels`, `.preprocessing`/`.strides` attributes on the
    returned model).

    Input
    ----------
        (see `fcos_lite_detection_base` for `feature_levels`/`backbone`/
        `weights`/`freeze_backbone`/`freeze_batch_norm`)
        neck_channels: `ghost_pan_lite_neck`'s common output width.
        stem_channels, stem_layers: `gfl_lite_head`'s per-level independent
                                    stem config.
        reg_max: `gfl_lite_head`'s distribution bin count minus one.
        num_extra_levels: `ghost_pan_lite_neck`'s optional extra, coarser
                          pyramid level(s) - `0` (off) by default, matching
                          `fcos_lite_detection_base`'s own fixed-3-level
                          design; not retrofitted onto that function, which
                          this one leaves untouched.

    Output
    ----------
        A `tf.keras.Model` with a `.preprocessing` attribute and a
        `.strides` attribute (per-level input-pixel strides, needed by
        `generate_anchor_points`/the assigners/`gfl_integral_decode`),
        outputting a flat `2 * (3 + num_extra_levels)`-tensor list
        `[cls_p0, reg_dist_p0, cls_p1, reg_dist_p1, ...]`.
    '''
    backbone_ = backbone_zoo(
        backbone, weights, input_tensor, max(feature_levels), freeze_backbone, freeze_batch_norm)
    x_encoder = backbone_([input_tensor, ])

    features = [x_encoder[i - 1] for i in feature_levels]
    neck_features = ghost_pan_lite_neck(
        features, out_channels=neck_channels, num_extra_levels=num_extra_levels, name='{}_neck'.format(name))

    strides = [int(input_tensor.shape[1] // f.shape[1]) for f in neck_features]
    head_outputs = gfl_lite_head(
        neck_features, n_classes, reg_max=reg_max, stem_channels=stem_channels,
        stem_layers=stem_layers, name='{}_head'.format(name),
    )

    flat_outputs = []
    for cls_logits, reg_dist_logits in head_outputs:
        flat_outputs.extend([cls_logits, reg_dist_logits])

    m = Model([input_tensor, ], flat_outputs)

    with tfmot.quantization.keras.quantize_scope():
        m = tfmot.quantization.keras.quantize_apply(m)

    m.preprocessing = backbone_.preprocessing
    m.strides = strides
    m.reg_max = reg_max

    return m


def gfl_lite_detection(input_tensor, n_classes, feature_levels=(3, 4, 5), neck_channels=96,
                       stem_channels=96, stem_layers=2, reg_max=7, num_extra_levels=0,
                       backbone='MobileNetV2', weights='imagenet', freeze_backbone=True,
                       freeze_batch_norm=True, score_threshold=0.05, iou_threshold=0.5,
                       max_detections=100, name='gfl_lite'):
    '''Full GFL-lite detector: `gfl_lite_detection_base` plus
    `gfl_integral_decode` (distribution -> pixel distance), `decode_boxes`,
    and `_nms_dispatch`-based NMS, for inference use - mirrors
    `fcos_lite_detection`'s own full-model structure exactly (including the
    same `combined_non_max_suppression` -> `_nms_dispatch` fix, and the
    same batching-capability change that comes with it - see that
    function's own docstring for the full "why" account), with
    `sigmoid(cls_logits)` standing in for FCOS-lite's own
    `sqrt(cls_score * centerness)` combined score (no centerness branch
    here - see module docstring).

    Input
    ----------
        (see `gfl_lite_detection_base` for the shared arguments)
        score_threshold, iou_threshold, max_detections: as `fcos_lite_detection`.

    Output
    ----------
        A `tf.keras.Model` with a `.preprocessing` attribute, producing
        `(boxes, scores, classes, valid_detections)` - same convention as
        `fcos_lite_detection`'s own output.
    '''
    base_model = gfl_lite_detection_base(
        input_tensor, n_classes, feature_levels=feature_levels, neck_channels=neck_channels,
        stem_channels=stem_channels, stem_layers=stem_layers, reg_max=reg_max,
        num_extra_levels=num_extra_levels, backbone=backbone, weights=weights,
        freeze_backbone=freeze_backbone, freeze_batch_norm=freeze_batch_norm, name=name,
    )

    raw_outputs = base_model(input_tensor)
    strides = base_model.strides
    n_levels = len(strides)
    feature_shapes = [(int(raw_outputs[i * 2].shape[1]), int(raw_outputs[i * 2].shape[2])) for i in range(n_levels)]
    anchor_points_per_level = generate_anchor_points(feature_shapes, strides)

    all_boxes, all_scores = [], []
    for level_idx in range(n_levels):
        cls_logits, reg_dist_logits = raw_outputs[level_idx * 2: level_idx * 2 + 2]
        stride = strides[level_idx]

        height, width = feature_shapes[level_idx]
        anchor_points = tf.constant(anchor_points_per_level[level_idx], dtype=tf.float32)

        cls_scores = tf.reshape(tf.sigmoid(cls_logits), (-1, height * width, n_classes))

        reg_dist_flat = tf.reshape(reg_dist_logits, (-1, height * width, 4, reg_max + 1))
        reg_ltrb = gfl_integral_decode(reg_dist_flat, stride)
        boxes = decode_boxes(anchor_points, reg_ltrb)

        all_boxes.append(boxes)
        all_scores.append(cls_scores)

    boxes = tf.concat(all_boxes, axis=1)
    scores = tf.concat(all_scores, axis=1)

    nms_boxes, nms_scores, nms_classes, valid_detections = _nms_dispatch(
        input_tensor, boxes, scores, n_classes, max_output_size_per_class=max_detections,
        max_detections=max_detections, iou_threshold=iou_threshold, score_threshold=score_threshold,
        name='{}_nms'.format(name),
    )

    m = Model([input_tensor, ], [nms_boxes, nms_scores, nms_classes, valid_detections])
    m.preprocessing = base_model.preprocessing

    return m


# ========================= #
# Instance segmentation model builders (plan.md §13.6, Phase 11) - mirrors
# `fcos_lite_detection_base`/`fcos_lite_detection`'s own base/full split
# exactly, with `fcos_lite_instance_head`/`_prototype_branch` replacing
# `fcos_lite_head`, and `_indexed_nms_gather` (not `combined_non_max_
# suppression`) driving the full model's own NMS.


def fcos_lite_instance_segmentation_base(input_tensor, n_classes, k=32, feature_levels=(3, 4, 5),
                                         fpn_channels=96, proto_channels=64, stem_channels=96, stem_layers=2,
                                         backbone='MobileNetV2', weights='imagenet', freeze_backbone=True,
                                         freeze_batch_norm=True, name='fcos_lite_instseg'):
    '''Backbone -> `pafpn_lite_neck` -> `fcos_lite_instance_head` (per-level
    predictions, including mask coefficients) + `_prototype_branch` (one
    shared prototype-mask tensor) - the "base" half of the `base.py`/
    `models.py` split every other model in this plan uses. **Any batch
    size** - unlike the full model below, nothing here needs NMS.

    Input
    ----------
        (see `fcos_lite_detection_base` for `feature_levels`/`fpn_channels`/
        `stem_channels`/`stem_layers`/`backbone`/`weights`/
        `freeze_backbone`/`freeze_batch_norm`)
        k: number of mask coefficients per box / prototype masks - must
           match between `fcos_lite_instance_head` and `_prototype_branch`
           (both driven by this one parameter, not independently settable).
        proto_channels: `_prototype_branch`'s own hidden channel width.

    Output
    ----------
        A `tf.keras.Model` with `.preprocessing`/`.strides`/`.k` attributes,
        outputting a flat `4*3 + 1 = 13`-tensor list: `[cls_p0, reg_p0,
        centerness_p0, mask_coefs_p0, cls_p1, ..., mask_coefs_p2,
        prototypes]` (per-level 4-tuples flattened, prototypes last).
    '''
    backbone_ = backbone_zoo(
        backbone, weights, input_tensor, max(feature_levels), freeze_backbone, freeze_batch_norm)
    x_encoder = backbone_([input_tensor, ])

    features = [x_encoder[i - 1] for i in feature_levels]
    fpn_features = pafpn_lite_neck(features, out_channels=fpn_channels, name='{}_pafpn'.format(name))

    strides = [int(input_tensor.shape[1] // f.shape[1]) for f in fpn_features]
    head_outputs = fcos_lite_instance_head(
        fpn_features, n_classes, k=k, strides=strides, stem_channels=stem_channels,
        stem_layers=stem_layers, name='{}_head'.format(name),
    )
    prototypes = _prototype_branch(
        fpn_features[0], k=k, proto_channels=proto_channels, name='{}_proto'.format(name))

    flat_outputs = []
    for cls_logits, reg_ltrb, centerness, mask_coefs in head_outputs:
        flat_outputs.extend([cls_logits, reg_ltrb, centerness, mask_coefs])
    flat_outputs.append(prototypes)

    m = Model([input_tensor, ], flat_outputs)

    with tfmot.quantization.keras.quantize_scope():
        m = tfmot.quantization.keras.quantize_apply(m)

    m.preprocessing = backbone_.preprocessing
    m.strides = strides
    m.k = k

    return m


def fcos_lite_instance_segmentation(input_tensor, n_classes, k=32, feature_levels=(3, 4, 5),
                                    fpn_channels=96, proto_channels=64, stem_channels=96, stem_layers=2,
                                    backbone='MobileNetV2', weights='imagenet', freeze_backbone=True,
                                    freeze_batch_norm=True, score_threshold=0.05, iou_threshold=0.5,
                                    max_output_size_per_class=100, max_detections=100, name='fcos_lite_instseg'):
    '''Full YOLACT-style FCOS-lite instance segmentation detector:
    `fcos_lite_instance_segmentation_base` plus box decoding,
    `_indexed_nms_gather`-based NMS (not `combined_non_max_suppression` -
    see that function's own docstring for why), and mask assembly
    (`sigmoid(prototypes @ coefficients)`, zeroed outside each detection's
    own box), for inference use.

    **Batch size 1 only, explicitly - a real, deliberate limitation, not
    an oversight.** `tf.image.non_max_suppression_padded` (which
    `_indexed_nms_gather` uses instead of `combined_non_max_suppression`,
    to keep mask coefficients attached to their own box through
    suppression) has no native batched form; a runtime
    `tf.debugging.assert_equal` (via `_AssertBatchSizeOne`) enforces this
    at call time in the Keras/TF graph (raises instead of silently only
    processing the first image of a larger batch). Batching this properly
    (e.g. via `tf.map_fn`) is a real, separate, **unverified** risk - the
    standalone spike this feature was built from (plan.md §13.6) only
    tested the batch-size-1 case. Training uses
    `fcos_lite_instance_segmentation_base` instead, which has no such
    restriction.

    **The runtime assert does not survive TFLite export - confirmed, not
    assumed.** Converting this model (wrapped in `uint8_input_wrapper`)
    through the same strict `TFLITE_BUILTINS_INT8` path used elsewhere in
    this file produces zero `ASSERT`-related op in the output (inspected
    directly via the converted flatbuffer's own op list) - the TFLite
    converter silently prunes the graph-only `Assert` op once nothing
    downstream consumes its (non-existent) output, a well-known TFLite
    optimization behavior, not a bug in this code. In practice this is a
    soft landing, not a silent hazard: a `.tflite` export of this model
    still only has a `batch=1`-shaped input dimension baked in (from
    `input_tensor` at build time) unless the caller explicitly built it
    with a larger batch dimension, and Axis's own inference pipeline feeds
    one camera frame at a time regardless - but the deployed `.tflite`
    artifact itself no longer enforces the batch-size-1 contract the way
    the Keras/TF `.predict()` path does, so this must not be relied on as
    a deployment-time safety net.

    Input
    ----------
        (see `fcos_lite_instance_segmentation_base` for the shared
        arguments)
        score_threshold, iou_threshold, max_output_size_per_class,
        max_detections: as `fcos_lite_detection`'s own arguments (per-class
            NMS parameters, not `combined_non_max_suppression`'s own single
            `max_output_size_per_class`/`max_total_size` pair, though the
            role is the same).

    Output
    ----------
        A `tf.keras.Model` with a `.preprocessing` attribute, producing
        `(boxes, scores, classes, valid_detections, masks)` - `boxes`
        shape `(1, max_detections, 4)`, `scores`/`classes` shape
        `(1, max_detections)`, `valid_detections` shape `(1,)` (matching
        `fcos_lite_detection`'s own convention: how many of the
        `max_detections` slots are real, the rest zero-padded), `masks`
        shape `(1, max_detections, H_proto, W_proto)` - per-detection
        assembled masks at `_prototype_branch`'s own (sub-input) resolution,
        `[0, 1]`-valued, zeroed outside that detection's own predicted box.
    '''
    base_model = fcos_lite_instance_segmentation_base(
        input_tensor, n_classes, k=k, feature_levels=feature_levels, fpn_channels=fpn_channels,
        proto_channels=proto_channels, stem_channels=stem_channels, stem_layers=stem_layers,
        backbone=backbone, weights=weights, freeze_backbone=freeze_backbone,
        freeze_batch_norm=freeze_batch_norm, name=name,
    )

    raw_outputs = base_model(input_tensor)
    strides = base_model.strides
    n_levels = len(strides)
    prototypes = raw_outputs[-1]
    level_outputs = raw_outputs[:-1]

    feature_shapes = [
        (int(level_outputs[i * 4].shape[1]), int(level_outputs[i * 4].shape[2])) for i in range(n_levels)
    ]
    anchor_points_per_level = generate_anchor_points(feature_shapes, strides)

    all_boxes, all_scores, all_mask_coefs = [], [], []
    for level_idx in range(n_levels):
        cls_logits, reg_ltrb, centerness, mask_coefs = level_outputs[level_idx * 4: level_idx * 4 + 4]
        height, width = feature_shapes[level_idx]
        anchor_points = tf.constant(anchor_points_per_level[level_idx], dtype=tf.float32)

        cls_scores = tf.reshape(tf.sigmoid(cls_logits), (-1, height * width, n_classes))
        centerness_scores = tf.reshape(tf.sigmoid(centerness), (-1, height * width, 1))
        combined_scores = tf.sqrt(cls_scores * centerness_scores)

        reg_flat = tf.reshape(reg_ltrb, (-1, height * width, 4))
        boxes = decode_boxes(anchor_points, reg_flat)

        mask_coefs_flat = tf.reshape(mask_coefs, (-1, height * width, k))

        all_boxes.append(boxes)
        all_scores.append(combined_scores)
        all_mask_coefs.append(mask_coefs_flat)

    boxes = tf.concat(all_boxes, axis=1)
    scores = tf.concat(all_scores, axis=1)
    mask_coefs = tf.concat(all_mask_coefs, axis=1)

    boxes = _AssertBatchSizeOne(name='{}_assert_batch1'.format(name))(boxes)
    batch_boxes = boxes[0]
    batch_scores = scores[0]
    batch_mask_coefs = mask_coefs[0]

    top_boxes, top_scores, top_labels, top_extras = _indexed_nms_gather(
        batch_boxes, batch_scores, {'mask_coefs': batch_mask_coefs}, n_classes,
        max_output_size_per_class, max_detections, iou_threshold, score_threshold,
    )
    top_mask_coefs = top_extras['mask_coefs']

    proto_h, proto_w = int(prototypes.shape[1]), int(prototypes.shape[2])
    proto_flat = tf.reshape(prototypes[0], (proto_h * proto_w, k))
    assembled = tf.matmul(proto_flat, top_mask_coefs, transpose_b=True)
    assembled = tf.reshape(assembled, (proto_h, proto_w, max_detections))
    assembled = tf.sigmoid(assembled)
    assembled = tf.transpose(assembled, [2, 0, 1])  # (max_detections, Hp, Wp)

    input_height = int(input_tensor.shape[1])
    input_width = int(input_tensor.shape[2])
    scale_x = proto_w / float(input_width)
    scale_y = proto_h / float(input_height)

    grid_y, grid_x = tf.meshgrid(
        tf.range(proto_h, dtype=tf.float32), tf.range(proto_w, dtype=tf.float32), indexing='ij')
    grid_x = grid_x[tf.newaxis, :, :]
    grid_y = grid_y[tf.newaxis, :, :]

    box_x1 = (top_boxes[:, 0] * scale_x)[:, tf.newaxis, tf.newaxis]
    box_y1 = (top_boxes[:, 1] * scale_y)[:, tf.newaxis, tf.newaxis]
    box_x2 = (top_boxes[:, 2] * scale_x)[:, tf.newaxis, tf.newaxis]
    box_y2 = (top_boxes[:, 3] * scale_y)[:, tf.newaxis, tf.newaxis]

    inside_box = tf.cast(
        (grid_x >= box_x1) & (grid_x < box_x2) & (grid_y >= box_y1) & (grid_y < box_y2), assembled.dtype,
    )
    masks = assembled * inside_box  # (max_detections, Hp, Wp)

    top_boxes = top_boxes[tf.newaxis, ...]
    top_scores = top_scores[tf.newaxis, ...]
    top_labels = top_labels[tf.newaxis, ...]
    masks = masks[tf.newaxis, ...]
    valid_detections = tf.reduce_sum(tf.cast(top_scores > 0, tf.int32), axis=-1)

    m = Model([input_tensor, ], [top_boxes, top_scores, top_labels, valid_detections, masks])
    m.preprocessing = base_model.preprocessing

    return m
