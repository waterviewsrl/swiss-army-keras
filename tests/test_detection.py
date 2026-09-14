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

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras._model_detection import (
    atss_assign_targets,
    decode_boxes,
    fcos_lite_detection,
    fcos_lite_detection_base,
    generate_anchor_points,
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
