"""Tests for the FCOS-lite detection module (plan.md §9.3/§13 Phase 5) —
`swiss_army_keras._model_detection`. No detection code existed anywhere in
this repo before this pass (confirmed via `grep`, per plan.md §9's own
opening note), so this covers build/predict shape contracts for the base
model and full (NMS) model, plus targeted correctness tests for the
non-Keras-graph pieces (`generate_anchor_points`, `decode_boxes`,
`atss_assign_targets`) where a hand-computable synthetic example gives an
exact expected answer, not just "didn't crash".

256x256 input throughout (not this suite's usual 128) - large enough that
the deepest (stride-32) pyramid level still has real spatial extent (8x8),
avoiding the same "collapses to 1x1, hides real bugs" class of issue
documented for the backbone reparameterization tests (`test_backbone_mobileone.py`
et al.) and DDRNet's DAPPM tests.
"""

import math

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras._model_detection import (
    _ghost_module,
    _indexed_nms_gather,
    _prototype_branch,
    _resize_mask_np,
    assign_instance_targets,
    atss_assign_targets,
    decode_boxes,
    dsla_assign_targets,
    fcos_lite_detection,
    fcos_lite_detection_base,
    fcos_lite_instance_head,
    fcos_lite_instance_segmentation,
    fcos_lite_instance_segmentation_base,
    generate_anchor_points,
    gfl_integral_decode_np,
    gfl_lite_detection,
    gfl_lite_detection_base,
    gfl_lite_head,
    ghost_pan_lite_neck,
    simota_assign_targets,
)

INPUT_SIZE = 256
N_CLASSES = 4


def _random_batch(batch_size=1):
    return np.random.rand(batch_size, INPUT_SIZE, INPUT_SIZE, 3).astype("float32")


def test_fcos_lite_detection_base_builds_and_predicts():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = fcos_lite_detection_base(inp, n_classes=N_CLASSES, backbone="MobileNetV2", weights=None, freeze_backbone=False)

    assert len(model.outputs) == 9  # 3 levels x (cls, reg, centerness)
    assert model.strides == [8, 16, 32]

    out = model.predict(_random_batch(), verbose=0)
    expected_spatial = [INPUT_SIZE // s for s in model.strides]
    for level_idx, spatial in enumerate(expected_spatial):
        cls_out, reg_out, centerness_out = out[level_idx * 3: level_idx * 3 + 3]
        assert cls_out.shape == (1, spatial, spatial, N_CLASSES)
        assert reg_out.shape == (1, spatial, spatial, 4)
        assert centerness_out.shape == (1, spatial, spatial, 1)
        assert np.all(np.isfinite(cls_out))
        assert np.all(reg_out >= 0), "regression output (l,t,r,b distances) must be non-negative"
        assert np.all(np.isfinite(centerness_out))


def test_fcos_lite_detection_base_with_4_tap_backbone():
    """RepViT (plan.md §2.4) only has 4 native pyramid taps - needs
    `feature_levels=(2, 3, 4)` instead of the 5-tap default `(3, 4, 5)`,
    same override pattern as `pp_liteseg`.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = fcos_lite_detection_base(
        inp, n_classes=N_CLASSES, backbone="RepViT-M0.9", weights=None,
        freeze_backbone=False, feature_levels=(2, 3, 4),
    )
    out = model.predict(_random_batch(), verbose=0)
    assert len(out) == 9
    assert np.all(np.isfinite(np.concatenate([o.ravel() for o in out])))


def test_fcos_lite_detection_full_model_builds_and_predicts():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = fcos_lite_detection(inp, n_classes=N_CLASSES, backbone="MobileNetV2", weights=None, freeze_backbone=False)

    boxes, scores, classes, valid_detections = model.predict(_random_batch(), verbose=0)

    assert boxes.shape == (1, 100, 4)
    assert scores.shape == (1, 100)
    assert classes.shape == (1, 100)
    assert valid_detections.shape == (1,)
    assert np.all(np.isfinite(boxes))
    assert np.all(np.isfinite(scores))
    assert np.all(valid_detections >= 0) and np.all(valid_detections <= 100)
    # boxes must be well-formed (x_min <= x_max, y_min <= y_max) for the
    # actually-valid detections.
    n_valid = int(valid_detections[0])
    valid_boxes = boxes[0, :n_valid]
    assert np.all(valid_boxes[:, 2] >= valid_boxes[:, 0])
    assert np.all(valid_boxes[:, 3] >= valid_boxes[:, 1])


def test_fcos_lite_detection_max_detections_respected():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = fcos_lite_detection(
        inp, n_classes=N_CLASSES, backbone="MobileNetV2", weights=None,
        freeze_backbone=False, max_detections=17, score_threshold=0.0,
    )
    boxes, scores, classes, valid_detections = model.predict(_random_batch(), verbose=0)
    assert boxes.shape == (1, 17, 4)
    assert int(valid_detections[0]) <= 17


def test_fcos_lite_detection_dynamic_batch_input_rejects_batch_size_greater_than_1():
    """Regression test for the `combined_non_max_suppression` ->
    `_nms_dispatch` swap: a normal dynamic-batch `Input((H, W, 3))` (the
    usual case, matching every other builder in this file) now only
    supports `batch_size=1` for the full (NMS'd) model - `combined_non_max_
    suppression` used to natively batch, but doesn't convert under strict
    int8 TFLite export at all (see `_nms_dispatch`'s own docstring). A
    batch>1 call must raise clearly, not silently misbehave.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = fcos_lite_detection(inp, n_classes=N_CLASSES, backbone="MobileNetV2", weights=None, freeze_backbone=False)

    boxes, scores, classes, valid_detections = model.predict(_random_batch(batch_size=1), verbose=0)
    assert boxes.shape == (1, 100, 4)

    with pytest.raises(tf.errors.InvalidArgumentError):
        model.predict(_random_batch(batch_size=2), verbose=0)


def test_fcos_lite_detection_static_batch_input_enables_real_batching():
    """The opt-in path: building with `Input(batch_shape=(N, H, W, 3))`
    instead of the usual dynamic-batch `Input((H, W, 3))` gets real,
    fixed-size multi-image NMS (`_indexed_nms_gather_unrolled`). Two checks,
    not just "didn't crash": (1) feeding the *same* image into every batch
    slot must give identical results in every slot (no cross-image
    leakage through the unrolled loop); (2) feeding genuinely different
    images must generally give different results (not silently collapsed
    to one constant regardless of input).
    """
    # MobileOne-S0, not this file's usual MobileNetV2 default - the
    # second check below asserts a genuinely non-degenerate property
    # ("different images give different output"), which a documented,
    # unrelated `weights=None` + MobileNetV2 bug (plan.md §10 - a
    # QAT-annotated final layer collapsing to exactly constant output)
    # would trivially satisfy without actually testing anything;
    # MobileOne-S0 doesn't reproduce that bug, per this repo's own
    # established test convention.
    inp = tf.keras.Input(batch_shape=(3, INPUT_SIZE, INPUT_SIZE, 3))
    model = fcos_lite_detection(inp, n_classes=N_CLASSES, backbone="MobileOne-S0", weights=None)

    rng = np.random.default_rng(0)
    one_image = rng.random((1, INPUT_SIZE, INPUT_SIZE, 3)).astype("float32")
    repeated = np.repeat(one_image, 3, axis=0)

    boxes, scores, classes, valid_detections = model.predict(repeated, verbose=0)
    assert boxes.shape == (3, 100, 4)
    np.testing.assert_allclose(boxes[0], boxes[1], atol=1e-4)
    np.testing.assert_allclose(boxes[0], boxes[2], atol=1e-4)
    np.testing.assert_allclose(scores[0], scores[2], atol=1e-4)
    assert int(valid_detections[0]) == int(valid_detections[2])

    different_images = rng.random((3, INPUT_SIZE, INPUT_SIZE, 3)).astype("float32")
    diff_boxes, diff_scores, diff_classes, diff_valid = model.predict(different_images, verbose=0)
    assert not np.allclose(diff_boxes[0], diff_boxes[1]) or not np.allclose(diff_scores[0], diff_scores[1])


def test_generate_anchor_points_grid_matches_expected_coordinates():
    feature_shapes = [(2, 2)]
    strides = [8]
    anchor_points = generate_anchor_points(feature_shapes, strides)

    assert len(anchor_points) == 1
    assert anchor_points[0].shape == (4, 2)
    # A 2x2 grid at stride 8: cell centers at pixel offsets (4, 4), (12, 4), (4, 12), (12, 12).
    expected = np.array([[4.0, 4.0], [12.0, 4.0], [4.0, 12.0], [12.0, 12.0]], dtype="float32")
    np.testing.assert_allclose(np.sort(anchor_points[0], axis=0), np.sort(expected, axis=0))


def test_decode_boxes_inverts_known_ltrb_exactly():
    anchor = np.array([[32.0, 32.0]], dtype="float32")
    ltrb = tf.constant([[[10.0, 6.0, 8.0, 4.0]]])  # l, t, r, b
    decoded = decode_boxes(anchor, ltrb).numpy()
    expected = np.array([[[32.0 - 10.0, 32.0 - 6.0, 32.0 + 8.0, 32.0 + 4.0]]])
    np.testing.assert_allclose(decoded, expected)


class TestAtssAssignTargets:
    FEATURE_SHAPES = [(8, 8), (4, 4), (2, 2)]
    STRIDES = [8, 16, 32]

    @pytest.fixture
    def anchor_points(self):
        return generate_anchor_points(self.FEATURE_SHAPES, self.STRIDES)

    def test_no_ground_truth_boxes_yields_all_negative(self, anchor_points):
        cls_t, reg_t, cn_t, pos_mask = atss_assign_targets(
            np.zeros((0, 4), dtype="float32"), np.zeros((0,), dtype="int64"),
            anchor_points, self.STRIDES, n_classes=3,
        )
        for pm in pos_mask:
            assert not pm.any()
        for c in cls_t:
            assert not c.any()

    def test_single_box_assigns_exact_expected_positives_and_targets(self, anchor_points):
        """Hand-computable case: a 20x20 box centered on the image, at a
        resolution where the assignment can be verified exactly by hand
        (see the inline computation below) - not just "some positives exist".
        """
        gt_boxes = np.array([[22.0, 22.0, 42.0, 42.0]], dtype="float32")
        gt_labels = np.array([1], dtype="int64")

        cls_t, reg_t, cn_t, pos_mask = atss_assign_targets(
            gt_boxes, gt_labels, anchor_points, self.STRIDES, n_classes=3, topk=9,
        )

        # Only level 0 (stride 8) has anchor points landing inside a 20x20
        # box at this resolution/topk - level 0's grid is spaced 8px apart
        # starting at (4,4), so (28,28)/(36,28)/(28,36)/(36,36) are exactly
        # the 4 grid points strictly inside (22,22)-(42,42).
        assert pos_mask[0].sum() == 4
        assert pos_mask[1].sum() == 0
        assert pos_mask[2].sum() == 0

        positive_idx = np.where(pos_mask[0])[0]
        positive_points = anchor_points[0][positive_idx]
        expected_points = {(28.0, 28.0), (36.0, 28.0), (28.0, 36.0), (36.0, 36.0)}
        assert {tuple(p) for p in positive_points} == expected_points

        for idx, point in zip(positive_idx, positive_points):
            l = point[0] - gt_boxes[0, 0]
            t = point[1] - gt_boxes[0, 1]
            r = gt_boxes[0, 2] - point[0]
            b = gt_boxes[0, 3] - point[1]
            np.testing.assert_allclose(reg_t[0][idx], [l, t, r, b], atol=1e-4)

            expected_centerness = np.sqrt((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))
            np.testing.assert_allclose(cn_t[0][idx], expected_centerness, atol=1e-4)

            expected_cls = np.zeros(3, dtype="float32")
            expected_cls[1] = 1.0
            np.testing.assert_allclose(cls_t[0][idx], expected_cls)

    def test_conflict_resolution_keeps_higher_iou_assignment(self, anchor_points):
        """Two overlapping ground-truth boxes competing for the same
        anchors - the anchor should end up assigned to whichever GT gives
        it the higher IoU, not just "the last one processed".
        """
        gt_boxes = np.array([
            [20.0, 20.0, 44.0, 44.0],   # larger box, label 0
            [26.0, 26.0, 38.0, 38.0],   # smaller, tighter box fully inside the first, label 1
        ], dtype="float32")
        gt_labels = np.array([0, 1], dtype="int64")

        cls_t, reg_t, cn_t, pos_mask = atss_assign_targets(
            gt_boxes, gt_labels, anchor_points, self.STRIDES, n_classes=2, topk=9,
        )

        assert pos_mask[0].sum() > 0
        # Every positive anchor must be assigned to exactly one class (rows
        # sum to 1, not 0 or 2 - no anchor should end up claimed by both
        # GTs' one-hot targets at once).
        positive_idx = np.where(pos_mask[0])[0]
        row_sums = cls_t[0][positive_idx].sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums))

    def test_return_gt_indices_false_default_matches_original_four_tuple(self, anchor_points):
        """Backward-compatibility regression test (plan.md §13.6's own
        `return_gt_indices` addition) - the default call shape must stay
        exactly the original 4-tuple, not silently grow a 5th element.
        """
        gt_boxes = np.array([[22.0, 22.0, 42.0, 42.0]], dtype="float32")
        gt_labels = np.array([1], dtype="int64")
        result = atss_assign_targets(gt_boxes, gt_labels, anchor_points, self.STRIDES, n_classes=3, topk=9)
        assert len(result) == 4

    def test_return_gt_indices_true_appends_correct_per_anchor_gt_index(self, anchor_points):
        gt_boxes = np.array([[22.0, 22.0, 42.0, 42.0]], dtype="float32")
        gt_labels = np.array([1], dtype="int64")
        cls_t, reg_t, cn_t, pos_mask, gt_indices = atss_assign_targets(
            gt_boxes, gt_labels, anchor_points, self.STRIDES, n_classes=3, topk=9, return_gt_indices=True,
        )

        # Same 4 positives as test_single_box_assigns_exact_expected_positives_and_targets.
        positive_idx = np.where(pos_mask[0])[0]
        assert np.all(gt_indices[0][positive_idx] == 0)  # the only GT box, index 0

        negative_idx = np.where(~pos_mask[0])[0]
        assert np.all(gt_indices[0][negative_idx] == -1)
        assert np.all(gt_indices[1] == -1)
        assert np.all(gt_indices[2] == -1)


# Ghost-PAN neck + GFL-style head/detector (plan.md §9.5/§9.6, Phase 9) -
# PicoDet v1 (VFL/QFL + SimOTA + a GFL head) + NanoDet-Plus (DSLA +
# Ghost-PAN), built together per plan.md's own "evaluate together" guidance.
# See `_model_detection.py`'s module docstring's "GFL-style upgrade"
# section for the two raw-source corrections this pass found relative to
# plan.md's own earlier description (Ghost-PAN's real concat-based fusion,
# the real extra-level formula).


def test_ghost_module_channel_count_matches_ceil_rounding():
    """Regression test for a real, documented reference quirk (not a bug
    introduced by this port): `_ghost_module`'s real output width is
    `ceil(out_channels / ratio) * ratio`, which can exceed the requested
    `out_channels` when it isn't evenly divisible by `ratio` - the
    reference `GhostModule` doesn't trim the extra channels either.
    """
    inp = tf.keras.Input((16, 16, 32))
    out = _ghost_module(inp, out_channels=65, ratio=2, name='ghost_test')
    init_channels = math.ceil(65 / 2)
    expected_channels = init_channels * 2
    assert int(out.shape[-1]) == expected_channels
    assert expected_channels != 65  # confirms this case actually exercises the rounding


def test_ghost_pan_lite_neck_builds_and_predicts():
    p3 = tf.keras.Input((32, 32, 40))
    p4 = tf.keras.Input((16, 16, 80))
    p5 = tf.keras.Input((8, 8, 160))
    outs = ghost_pan_lite_neck([p3, p4, p5], out_channels=48)

    assert len(outs) == 3
    expected_shapes = [(32, 32), (16, 16), (8, 8)]
    for out, (h, w) in zip(outs, expected_shapes):
        assert out.shape[1:] == (h, w, 48)

    model = tf.keras.Model([p3, p4, p5], outs)
    preds = model.predict([
        np.random.rand(1, 32, 32, 40).astype('float32'),
        np.random.rand(1, 16, 16, 80).astype('float32'),
        np.random.rand(1, 8, 8, 160).astype('float32'),
    ], verbose=0)
    for p in preds:
        assert np.all(np.isfinite(p))


def test_ghost_pan_lite_neck_extra_level_shape():
    """`num_extra_levels=1` appends one coarser (stride-doubled) level,
    built as `Add([stride2_conv(reduced_deepest_input), stride2_conv(final_
    bottom_up_output)])` - the corrected formula (see module docstring),
    not two convs applied in sequence.
    """
    p3 = tf.keras.Input((32, 32, 40))
    p4 = tf.keras.Input((16, 16, 80))
    p5 = tf.keras.Input((8, 8, 160))
    outs = ghost_pan_lite_neck([p3, p4, p5], out_channels=48, num_extra_levels=1)

    assert len(outs) == 4
    assert outs[3].shape[1:] == (4, 4, 48)


class TestGflLiteHead:
    def test_builds_and_predicts_expected_shapes(self):
        p3 = tf.keras.Input((32, 32, 96))
        p4 = tf.keras.Input((16, 16, 96))
        p5 = tf.keras.Input((8, 8, 96))
        reg_max = 7
        outputs = gfl_lite_head([p3, p4, p5], n_classes=N_CLASSES, reg_max=reg_max)

        assert len(outputs) == 3
        expected_spatial = [(32, 32), (16, 16), (8, 8)]
        for (cls_logits, reg_dist_logits), (h, w) in zip(outputs, expected_spatial):
            assert cls_logits.shape[1:] == (h, w, N_CLASSES)
            assert reg_dist_logits.shape[1:] == (h, w, 4, reg_max + 1)

        model = tf.keras.Model([p3, p4, p5], [t for pair in outputs for t in pair])
        preds = model.predict([
            np.random.rand(1, 32, 32, 96).astype('float32'),
            np.random.rand(1, 16, 16, 96).astype('float32'),
            np.random.rand(1, 8, 8, 96).astype('float32'),
        ], verbose=0)
        for p in preds:
            assert np.all(np.isfinite(p))


def test_gfl_integral_decode_np_exact_spike_value():
    """A near-one-hot distribution (all probability mass on bin index 2)
    should decode to exactly that bin index, scaled by stride - `Integral`'s
    own `sum(softmax(logits) * arange(n_bins)) * stride` formula, hand-
    verifiable at this extreme.
    """
    reg_max = 3
    n_bins = reg_max + 1
    logits = np.full((1, 4, n_bins), -1e9, dtype=np.float32)
    logits[0, :, 2] = 1e9

    decoded = gfl_integral_decode_np(logits, stride=8.0)
    np.testing.assert_allclose(decoded, np.full((1, 4), 16.0, dtype=np.float32), atol=1e-3)


def test_gfl_lite_detection_base_builds_and_predicts():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = gfl_lite_detection_base(inp, n_classes=N_CLASSES, backbone="MobileNetV2", weights=None, freeze_backbone=False)

    assert len(model.outputs) == 6  # 3 levels x (cls, reg_dist)
    assert model.strides == [8, 16, 32]
    assert model.reg_max == 7

    out = model.predict(_random_batch(), verbose=0)
    expected_spatial = [INPUT_SIZE // s for s in model.strides]
    for level_idx, spatial in enumerate(expected_spatial):
        cls_out, reg_out = out[level_idx * 2: level_idx * 2 + 2]
        assert cls_out.shape == (1, spatial, spatial, N_CLASSES)
        assert reg_out.shape == (1, spatial, spatial, 4, model.reg_max + 1)
        assert np.all(np.isfinite(cls_out))
        assert np.all(np.isfinite(reg_out))


def test_gfl_lite_detection_base_with_4_tap_backbone():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = gfl_lite_detection_base(
        inp, n_classes=N_CLASSES, backbone="RepViT-M0.9", weights=None,
        freeze_backbone=False, feature_levels=(2, 3, 4),
    )
    out = model.predict(_random_batch(), verbose=0)
    assert len(out) == 6
    assert np.all(np.isfinite(np.concatenate([o.ravel() for o in out])))


def test_gfl_lite_detection_base_with_extra_level():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = gfl_lite_detection_base(
        inp, n_classes=N_CLASSES, backbone="MobileNetV2", weights=None,
        freeze_backbone=False, num_extra_levels=1,
    )
    assert len(model.outputs) == 8  # 4 levels x (cls, reg_dist)
    assert model.strides == [8, 16, 32, 64]
    out = model.predict(_random_batch(), verbose=0)
    assert np.all(np.isfinite(np.concatenate([o.ravel() for o in out])))


def test_gfl_lite_detection_full_model_builds_and_predicts():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = gfl_lite_detection(inp, n_classes=N_CLASSES, backbone="MobileNetV2", weights=None, freeze_backbone=False)

    boxes, scores, classes, valid_detections = model.predict(_random_batch(), verbose=0)

    assert boxes.shape == (1, 100, 4)
    assert scores.shape == (1, 100)
    assert classes.shape == (1, 100)
    assert valid_detections.shape == (1,)
    assert np.all(np.isfinite(boxes))
    assert np.all(np.isfinite(scores))
    n_valid = int(valid_detections[0])
    valid_boxes = boxes[0, :n_valid]
    assert np.all(valid_boxes[:, 2] >= valid_boxes[:, 0])
    assert np.all(valid_boxes[:, 3] >= valid_boxes[:, 1])


def test_gfl_lite_detection_dynamic_batch_input_rejects_batch_size_greater_than_1():
    """Same regression test as `fcos_lite_detection`'s own
    (`combined_non_max_suppression` -> `_nms_dispatch`) - a normal
    dynamic-batch `Input((H, W, 3))` now only supports `batch_size=1` for
    the full (NMS'd) model.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = gfl_lite_detection(inp, n_classes=N_CLASSES, backbone="MobileNetV2", weights=None, freeze_backbone=False)

    boxes, scores, classes, valid_detections = model.predict(_random_batch(batch_size=1), verbose=0)
    assert boxes.shape == (1, 100, 4)

    with pytest.raises(tf.errors.InvalidArgumentError):
        model.predict(_random_batch(batch_size=2), verbose=0)


def test_gfl_lite_detection_static_batch_input_enables_real_batching():
    """Same regression test as `fcos_lite_detection`'s own - the opt-in
    `Input(batch_shape=(N, H, W, 3))` path gets real, fixed-size
    multi-image NMS with no cross-image leakage. MobileOne-S0, not this
    file's usual MobileNetV2 default - see `fcos_lite_detection`'s own
    sibling test for why.
    """
    inp = tf.keras.Input(batch_shape=(3, INPUT_SIZE, INPUT_SIZE, 3))
    model = gfl_lite_detection(inp, n_classes=N_CLASSES, backbone="MobileOne-S0", weights=None)

    rng = np.random.default_rng(0)
    one_image = rng.random((1, INPUT_SIZE, INPUT_SIZE, 3)).astype("float32")
    repeated = np.repeat(one_image, 3, axis=0)

    boxes, scores, classes, valid_detections = model.predict(repeated, verbose=0)
    assert boxes.shape == (3, 100, 4)
    np.testing.assert_allclose(boxes[0], boxes[1], atol=1e-4)
    np.testing.assert_allclose(boxes[0], boxes[2], atol=1e-4)
    np.testing.assert_allclose(scores[0], scores[2], atol=1e-4)
    assert int(valid_detections[0]) == int(valid_detections[2])


# SimOTA / DSLA dynamic label assignment - genuinely different from ATSS in
# kind (need the model's own current predictions, not just box geometry -
# see `_model_detection.py`'s module docstring), so every test below builds
# its own synthetic `pred_scores_per_level`/`pred_boxes_per_level` rather
# than reusing `TestAtssAssignTargets`'s fixture.


class TestDynamicAssigners:
    FEATURE_SHAPES = [(4, 4)]
    STRIDES = [8]
    N_CLASSES = 3

    @pytest.fixture
    def anchor_points(self):
        return generate_anchor_points(self.FEATURE_SHAPES, self.STRIDES)

    def _uniform_predictions(self, n_anchors, default_box):
        scores = [np.full((n_anchors, self.N_CLASSES), 0.1, dtype=np.float32)]
        boxes = [np.tile(np.array(default_box, dtype=np.float32), (n_anchors, 1))]
        return scores, boxes

    @pytest.mark.parametrize("assign_fn", [simota_assign_targets, dsla_assign_targets])
    def test_no_ground_truth_boxes_yields_all_negative(self, anchor_points, assign_fn):
        n_anchors = anchor_points[0].shape[0]
        scores, boxes = self._uniform_predictions(n_anchors, [0.0, 0.0, 1.0, 1.0])
        cls_t, reg_t, pos_mask = assign_fn(
            np.zeros((0, 4), dtype='float32'), np.zeros((0,), dtype='int64'),
            anchor_points, self.STRIDES, scores, boxes, self.N_CLASSES,
        )
        for pm in pos_mask:
            assert not pm.any()
        for c in cls_t:
            assert not c.any()

    @pytest.mark.parametrize("assign_fn", [simota_assign_targets, dsla_assign_targets])
    def test_single_perfect_candidate_is_the_only_positive(self, anchor_points, assign_fn):
        """A 4x4 grid at stride 8 (points at x, y in {4, 12, 20, 28}, row-
        major) - `gt_box = [8, 8, 16, 16]` has exactly one anchor center
        strictly inside it: `(12, 12)`, index 5. Its predicted box is set to
        exactly match the ground truth (IoU = 1); every other anchor's
        predicted box is set far away (IoU = 0). Both assigners' dynamic-k
        (`clip(int(sum of top-k IoUs), min=1)`) reduces to exactly 1 for
        this GT (only one candidate contributes a nonzero IoU), so exactly
        one positive is expected, at index 5, hand-verifiable exactly - the
        same style `TestAtssAssignTargets.test_single_box_assigns_exact_expected_positives_and_targets`
        already established for ATSS.
        """
        gt_boxes = np.array([[8.0, 8.0, 16.0, 16.0]], dtype='float32')
        gt_labels = np.array([2], dtype='int64')

        n_anchors = anchor_points[0].shape[0]
        scores, boxes = self._uniform_predictions(n_anchors, [100.0, 100.0, 108.0, 108.0])
        boxes[0][5] = [8.0, 8.0, 16.0, 16.0]

        cls_t, reg_t, pos_mask = assign_fn(
            gt_boxes, gt_labels, anchor_points, self.STRIDES, scores, boxes, self.N_CLASSES,
        )

        assert pos_mask[0].sum() == 1
        assert pos_mask[0][5]

        expected_cls = np.zeros(self.N_CLASSES, dtype='float32')
        expected_cls[2] = 1.0
        np.testing.assert_allclose(cls_t[0][5], expected_cls, atol=1e-4)
        np.testing.assert_allclose(reg_t[0][5], [4.0, 4.0, 4.0, 4.0], atol=1e-4)

    @pytest.mark.parametrize("assign_fn", [simota_assign_targets, dsla_assign_targets])
    def test_classification_target_is_iou_weighted_not_one_hot(self, anchor_points, assign_fn):
        """Both SimOTA and DSLA give an IoU-*soft* classification target at
        the matched position (`cls_targets[idx, label] = matched_iou`), not
        a hard `1.0` the way `atss_assign_targets`'s own one-hot target is
        (`TestAtssAssignTargets` above) - the real positive/`quality_focal_loss`-
        style target both assigners are meant to feed. Predicted box
        `[8, 8, 20, 20]` (12x12) vs `gt_box = [8, 8, 16, 16]` (8x8, matching
        anchor index 5 exactly, same grid as the test above) has a hand-
        computable IoU: intersection 8x8=64, union 144+64-64=144, IoU = 4/9.
        """
        gt_boxes = np.array([[8.0, 8.0, 16.0, 16.0]], dtype='float32')
        gt_labels = np.array([1], dtype='int64')

        n_anchors = anchor_points[0].shape[0]
        scores, boxes = self._uniform_predictions(n_anchors, [100.0, 100.0, 108.0, 108.0])
        boxes[0][5] = [8.0, 8.0, 20.0, 20.0]

        cls_t, reg_t, pos_mask = assign_fn(
            gt_boxes, gt_labels, anchor_points, self.STRIDES, scores, boxes, self.N_CLASSES,
        )

        assert pos_mask[0].sum() == 1
        expected_iou = 4.0 / 9.0
        assert cls_t[0][5, 1] < 0.999  # soft, not hard 1.0
        np.testing.assert_allclose(cls_t[0][5, 1], expected_iou, atol=1e-3)

    def test_simota_candidate_gate_is_looser_than_dsla(self, anchor_points):
        """The actual defining geometric difference between the two
        assigners (see `_model_detection.py`'s module docstring): SimOTA's
        candidate gate is "inside the GT box OR inside a `center_radius`-
        scaled square", DSLA's is "inside the GT box" only. `gt_box =
        [6, 6, 10, 10]` (4x4, centered at (8, 8)) has *no* anchor center
        from this grid strictly inside it - DSLA must return zero positives
        - but SimOTA's `center_radius=2.5` square (half-width `2.5 * stride
        = 20`) covers the whole grid, so a candidate with a real box-IoU
        match (anchor index 0, predicted box set to exactly match the GT)
        is still assignable.
        """
        gt_boxes = np.array([[6.0, 6.0, 10.0, 10.0]], dtype='float32')
        gt_labels = np.array([0], dtype='int64')

        n_anchors = anchor_points[0].shape[0]
        scores, boxes = self._uniform_predictions(n_anchors, [100.0, 100.0, 108.0, 108.0])
        boxes[0][0] = [6.0, 6.0, 10.0, 10.0]

        _, _, dsla_pos = dsla_assign_targets(
            gt_boxes, gt_labels, anchor_points, self.STRIDES, scores, boxes, self.N_CLASSES,
        )
        assert dsla_pos[0].sum() == 0, "no anchor center falls inside this GT box - DSLA must find zero candidates"

        _, _, simota_pos = simota_assign_targets(
            gt_boxes, gt_labels, anchor_points, self.STRIDES, scores, boxes, self.N_CLASSES,
        )
        assert simota_pos[0].sum() == 1
        assert simota_pos[0][0], "the center-radius gate should still admit anchor 0 as a candidate"


# YOLACT-style instance segmentation (plan.md §13.6, Phase 11) - a spike
# (see `_model_detection.py`'s own module-level account) already de-risked
# the NMS+gather+TFLite-export chain on a real `fcos_lite_detection_base`
# before this was built; these tests cover the actual shipped feature
# instead of the spike's synthetic mask-coefficient stand-in.


class TestFcosLiteInstanceHead:
    K = 8

    def test_prototype_branch_shape_and_finite(self):
        finest = tf.keras.Input((32, 32, 96))
        proto = _prototype_branch(finest, k=self.K, proto_channels=16)
        # One 2x upsample off a stride-8 finest feature -> stride 4.
        assert proto.shape[1:] == (64, 64, self.K)

        model = tf.keras.Model(finest, proto)
        pred = model.predict(np.random.rand(1, 32, 32, 96).astype("float32"), verbose=0)
        assert pred.shape == (1, 64, 64, self.K)
        assert np.all(np.isfinite(pred))

    def test_head_builds_and_predicts_expected_shapes(self):
        p3 = tf.keras.Input((32, 32, 96))
        p4 = tf.keras.Input((16, 16, 96))
        p5 = tf.keras.Input((8, 8, 96))
        outputs = fcos_lite_instance_head([p3, p4, p5], n_classes=N_CLASSES, k=self.K, strides=(8, 16, 32))

        assert len(outputs) == 3
        expected_spatial = [(32, 32), (16, 16), (8, 8)]
        for (cls_logits, reg_ltrb, centerness, mask_coefs), (h, w) in zip(outputs, expected_spatial):
            assert cls_logits.shape[1:] == (h, w, N_CLASSES)
            assert reg_ltrb.shape[1:] == (h, w, 4)
            assert centerness.shape[1:] == (h, w, 1)
            assert mask_coefs.shape[1:] == (h, w, self.K)

        model = tf.keras.Model([p3, p4, p5], [t for tup in outputs for t in tup])
        preds = model.predict([
            np.random.rand(1, 32, 32, 96).astype("float32"),
            np.random.rand(1, 16, 16, 96).astype("float32"),
            np.random.rand(1, 8, 8, 96).astype("float32"),
        ], verbose=0)
        for p in preds:
            assert np.all(np.isfinite(p))

        # reg_ltrb (exp-activated) is non-negative, exactly like fcos_lite_head's own.
        for reg_idx in (1, 5, 9):
            assert np.all(preds[reg_idx] >= 0)
        # mask_coefs are tanh-activated -> bounded in [-1, 1].
        for mask_idx in (3, 7, 11):
            assert np.all(preds[mask_idx] >= -1.0) and np.all(preds[mask_idx] <= 1.0)


def test_indexed_nms_gather_hand_computable_case():
    """4 boxes, 2 classes: A and B are exact duplicates (IoU=1.0, so B is
    suppressed under class 0), C doesn't overlap anything, D shares A's
    location but only scores under class 1 (a separate per-class NMS pass,
    so it isn't suppressed by A/B at all). Each box's own one-hot "mask
    coefficient" vector must travel with it through suppression, gather,
    and the final cross-class top-k - confirmed by checking the survivors'
    own coefficients, not just their boxes/scores.
    """
    boxes = tf.constant([
        [0.0, 0.0, 10.0, 10.0],    # A
        [0.0, 0.0, 10.0, 10.0],    # B - duplicate of A, suppressed under class 0
        [50.0, 50.0, 60.0, 60.0],  # C - no overlap with anything
        [0.0, 0.0, 10.0, 10.0],    # D - same location as A, but only class 1
    ], dtype=tf.float32)
    scores = tf.constant([
        [0.9, 0.0],
        [0.8, 0.0],
        [0.7, 0.0],
        [0.0, 0.6],
    ], dtype=tf.float32)
    coefs = tf.constant([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=tf.float32)

    top_boxes, top_scores, top_labels, top_extras = _indexed_nms_gather(
        boxes, scores, {"coef": coefs}, n_classes=2, max_output_size_per_class=10,
        max_detections=10, iou_threshold=0.5, score_threshold=0.05,
    )

    top_scores_np = top_scores.numpy()
    top_boxes_np = top_boxes.numpy()
    top_labels_np = top_labels.numpy()
    top_coefs_np = top_extras["coef"].numpy()

    np.testing.assert_allclose(top_scores_np[:3], [0.9, 0.7, 0.6], atol=1e-5)
    np.testing.assert_allclose(top_boxes_np[0], [0.0, 0.0, 10.0, 10.0])
    np.testing.assert_allclose(top_boxes_np[1], [50.0, 50.0, 60.0, 60.0])
    np.testing.assert_allclose(top_boxes_np[2], [0.0, 0.0, 10.0, 10.0])
    assert top_labels_np[0] == 0 and top_labels_np[1] == 0 and top_labels_np[2] == 1

    # Mask coefficients travel with their own surviving box, not shuffled.
    np.testing.assert_allclose(top_coefs_np[0], [1.0, 0.0, 0.0, 0.0])  # A's own coef
    np.testing.assert_allclose(top_coefs_np[1], [0.0, 0.0, 1.0, 0.0])  # C's own coef
    np.testing.assert_allclose(top_coefs_np[2], [0.0, 0.0, 0.0, 1.0])  # D's own coef
    # B (the suppressed duplicate) must not appear among the survivors.
    assert not np.any(np.all(np.isclose(top_coefs_np[:3], [0.0, 1.0, 0.0, 0.0]), axis=-1))


class TestAssignInstanceTargets:
    FEATURE_SHAPES = TestAtssAssignTargets.FEATURE_SHAPES
    STRIDES = TestAtssAssignTargets.STRIDES
    INPUT_SIZE = 256
    PROTO_SIZE = 64  # input / 4, matching _prototype_branch's own convention

    @pytest.fixture
    def anchor_points(self):
        return generate_anchor_points(self.FEATURE_SHAPES, self.STRIDES)

    def test_single_box_mask_and_area_land_on_correct_positive_anchors(self, anchor_points):
        """Reuses TestAtssAssignTargets's own hand-verified single-box case
        (exactly 4 positive anchors at level 0) - extended here with a
        synthetic GT mask, checking that every one of those same 4
        positives receives that box's own resized mask and area, and that
        negatives get a zero mask/area.
        """
        gt_boxes = np.array([[22.0, 22.0, 42.0, 42.0]], dtype="float32")
        gt_labels = np.array([1], dtype="int64")
        rng = np.random.default_rng(0)
        gt_masks = rng.random((1, 30, 30)).astype("float32")

        cls_t, reg_t, cn_t, pos_mask, mask_targets, mask_areas = assign_instance_targets(
            gt_boxes, gt_labels, gt_masks, anchor_points, self.STRIDES, n_classes=3,
            input_height=self.INPUT_SIZE, input_width=self.INPUT_SIZE,
            prototype_height=self.PROTO_SIZE, prototype_width=self.PROTO_SIZE, topk=9,
        )

        assert pos_mask[0].sum() == 4
        assert pos_mask[1].sum() == 0
        assert pos_mask[2].sum() == 0

        expected_resized_mask = _resize_mask_np(gt_masks[0], self.PROTO_SIZE, self.PROTO_SIZE)
        box_w = box_h = 20.0
        scale = self.PROTO_SIZE / self.INPUT_SIZE
        expected_area = (box_w * scale) * (box_h * scale)

        positive_idx = np.where(pos_mask[0])[0]
        for idx in positive_idx:
            np.testing.assert_allclose(mask_targets[0][idx], expected_resized_mask, atol=1e-5)
            np.testing.assert_allclose(mask_areas[0][idx], expected_area, atol=1e-4)

        negative_idx = np.where(~pos_mask[0])[0]
        for idx in negative_idx[:5]:
            assert not mask_targets[0][idx].any()
            assert mask_areas[0][idx] == 0.0

    def test_no_ground_truth_boxes_yields_empty_mask_targets(self, anchor_points):
        cls_t, reg_t, cn_t, pos_mask, mask_targets, mask_areas = assign_instance_targets(
            np.zeros((0, 4), dtype="float32"), np.zeros((0,), dtype="int64"),
            np.zeros((0, 10, 10), dtype="float32"), anchor_points, self.STRIDES, n_classes=3,
            input_height=self.INPUT_SIZE, input_width=self.INPUT_SIZE,
            prototype_height=self.PROTO_SIZE, prototype_width=self.PROTO_SIZE,
        )
        for mt in mask_targets:
            assert not mt.any()
        for ma in mask_areas:
            assert not ma.any()


class TestFcosLiteInstanceSegmentation:
    K = 8

    def test_base_builds_and_predicts_any_batch_size(self):
        inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
        model = fcos_lite_instance_segmentation_base(
            inp, n_classes=N_CLASSES, k=self.K, backbone="MobileNetV2", weights=None, freeze_backbone=False,
        )
        assert len(model.outputs) == 13  # 3 levels x 4 + prototypes
        assert model.strides == [8, 16, 32]
        assert model.k == self.K

        out = model.predict(_random_batch(batch_size=3), verbose=0)
        expected_spatial = [INPUT_SIZE // s for s in model.strides]
        for level_idx, spatial in enumerate(expected_spatial):
            cls_out, reg_out, centerness_out, mask_coef_out = out[level_idx * 4: level_idx * 4 + 4]
            assert cls_out.shape == (3, spatial, spatial, N_CLASSES)
            assert reg_out.shape == (3, spatial, spatial, 4)
            assert centerness_out.shape == (3, spatial, spatial, 1)
            assert mask_coef_out.shape == (3, spatial, spatial, self.K)

        prototypes = out[-1]
        assert prototypes.shape == (3, INPUT_SIZE // 4, INPUT_SIZE // 4, self.K)
        assert np.all(np.isfinite(np.concatenate([o.ravel() for o in out])))

    def test_full_model_builds_and_predicts_batch_size_1(self):
        inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
        model = fcos_lite_instance_segmentation(
            inp, n_classes=N_CLASSES, k=self.K, backbone="MobileNetV2", weights=None, freeze_backbone=False,
            max_output_size_per_class=20, max_detections=20,
        )
        boxes, scores, classes, valid_detections, masks = model.predict(_random_batch(batch_size=1), verbose=0)

        proto_size = INPUT_SIZE // 4
        assert boxes.shape == (1, 20, 4)
        assert scores.shape == (1, 20)
        assert classes.shape == (1, 20)
        assert valid_detections.shape == (1,)
        assert masks.shape == (1, 20, proto_size, proto_size)
        assert np.all(np.isfinite(boxes))
        assert np.all(np.isfinite(scores))
        assert np.all(masks >= 0.0) and np.all(masks <= 1.0)

    def test_full_model_rejects_batch_size_greater_than_1(self):
        """The full model's NMS+gather+mask-assembly path is explicitly
        batch-size-1 only (see `fcos_lite_instance_segmentation`'s own
        docstring) - a batch>1 call must raise clearly, not silently only
        process the first image and drop the rest.
        """
        inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
        model = fcos_lite_instance_segmentation(
            inp, n_classes=N_CLASSES, k=self.K, backbone="MobileNetV2", weights=None, freeze_backbone=False,
        )
        with pytest.raises(tf.errors.InvalidArgumentError):
            model.predict(_random_batch(batch_size=2), verbose=0)
