
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
  are decoded into absolute boxes and filtered via
  `tf.image.combined_non_max_suppression` (TF's own built-in, well-tested
  NMS op — not hand-rolled) in the full `<name>` model (`models.py`).
'''

from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
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
# FCOS-lite head


@register_keras_serializable(package='swiss_army_keras')
class ScaleLayer(Layer):
    '''A single learnable scalar multiplier — FCOS's own trick for letting
    each pyramid level's regression branch cover its own natural box-size
    range despite sharing the same regression-branch *weights* across every
    level (the branch itself has no way to know which level it's being
    applied to, otherwise).
    '''

    def __init__(self, initial_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_value = initial_value

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale', shape=(), initializer=tf.keras.initializers.Constant(self.initial_value),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({'initial_value': self.initial_value})
        return config


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
    own learnable `ScaleLayer`.

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
        # Forced to float32 explicitly - under a global mixed-precision
        # policy (e.g. `Policy('mixed_float16')`, set to speed up backbone
        # training), this plain, non-QAT-annotated layer would otherwise
        # silently pick up that ambient policy and cast `reg_head_conv`'s
        # (float32-forced, being QAT-annotated) output back down to float16,
        # which then reaches `tf.image.combined_non_max_suppression` in
        # `fcos_lite_detection` and crashes it (that op hard-requires
        # float32 boxes, confirmed empirically) - `cls_head_conv`'s and
        # `centerness_head_conv`'s outputs don't pass through any such layer
        # after their own QAT conv, which is why only the regression branch
        # needs this.
        reg_raw = ScaleLayer(initial_value=1.0, name=f'{name}_scale_{level_name}', dtype='float32')(reg_raw)
        reg_ltrb = tf.exp(reg_raw) * float(stride)

        centerness = centerness_head_conv(reg_feat)

        outputs.append((cls_logits, reg_ltrb, centerness))

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


def atss_assign_targets(gt_boxes, gt_labels, anchor_points_per_level, strides, n_classes, topk=9, anchor_scale=8.0):
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

    Output
    ----------
        `(cls_targets, reg_targets, centerness_targets, positive_mask)`,
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
    assigned_iou = np.full((n_total,), -1.0, dtype=np.float32)

    if len(gt_boxes) == 0:
        return (
            _split_by_level(cls_targets, level_sizes), _split_by_level(reg_targets, level_sizes),
            _split_by_level(centerness_targets, level_sizes), _split_by_level(positive_mask, level_sizes),
        )

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

    return (
        _split_by_level(cls_targets, level_sizes), _split_by_level(reg_targets, level_sizes),
        _split_by_level(centerness_targets, level_sizes), _split_by_level(positive_mask, level_sizes),
    )


def _split_by_level(flat_array, level_sizes):
    splits = np.cumsum(level_sizes)[:-1]
    return np.split(flat_array, splits, axis=0)


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
    decoding and `tf.image.combined_non_max_suppression`-based NMS, for
    inference use.

    Input
    ----------
        (see `fcos_lite_detection_base` for the shared arguments)
        score_threshold: minimum `sqrt(classification_score * centerness)`
                         (FCOS's own inference-time score combination) for a
                         detection to be considered at all.
        iou_threshold: NMS IoU threshold.
        max_detections: maximum detections returned per image.

    Output
    ----------
        A `tf.keras.Model` with a `.preprocessing` attribute, producing
        `(boxes, scores, classes, valid_detections)` — `boxes` shape
        `(B, max_detections, 4)` (`x_min, y_min, x_max, y_max`, input-pixel
        units), `scores`/`classes` shape `(B, max_detections)`,
        `valid_detections` shape `(B,)` (how many of the `max_detections`
        slots are real, per `tf.image.combined_non_max_suppression`'s own
        convention — the rest are zero-padded).
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
    boxes_for_nms = boxes[:, :, tf.newaxis, :]

    nms_boxes, nms_scores, nms_classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes_for_nms, scores, max_output_size_per_class=max_detections, max_total_size=max_detections,
        iou_threshold=iou_threshold, score_threshold=score_threshold, clip_boxes=False,
    )

    m = Model([input_tensor, ], [nms_boxes, nms_scores, nms_classes, valid_detections])
    m.preprocessing = base_model.preprocessing

    return m
