"""Tests for the new metrics added under plan.md §6/§13 Phase 6:
`BoundaryFScore` and `MeanAveragePrecision`. Only `MeanIOU` existed before
this pass (confirmed by reading `metrics.py`) - neither segmentation
boundary quality nor detection quality had a metric at all.
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras.metrics import BoundaryFScore, DepthAbsRel, DepthDeltaAccuracy, DepthRMSE, MeanAveragePrecision


def _rect_mask(shape, box):
    y0, x0, y1, x1 = box
    mask = np.zeros(shape, dtype="int32")
    mask[:, y0:y1, x0:x1] = 1
    return mask


def test_boundary_f_score_identical_masks_is_one():
    mask = _rect_mask((1, 32, 32), (8, 8, 24, 24))
    y = tf.one_hot(mask, 2)

    metric = BoundaryFScore(tolerance=2)
    metric.update_state(y, y)
    np.testing.assert_allclose(float(metric.result()), 1.0, atol=1e-4)


def test_boundary_f_score_small_shift_within_tolerance_scores_highly():
    y_true = tf.one_hot(_rect_mask((1, 32, 32), (8, 8, 24, 24)), 2)
    y_pred = tf.one_hot(_rect_mask((1, 32, 32), (9, 8, 25, 24)), 2)  # shifted 1px

    metric = BoundaryFScore(tolerance=2)
    metric.update_state(y_true, y_pred)
    assert float(metric.result()) > 0.95


def test_boundary_f_score_distant_boundary_scores_low():
    y_true = tf.one_hot(_rect_mask((1, 32, 32), (8, 8, 24, 24)), 2)
    y_pred = tf.one_hot(_rect_mask((1, 32, 32), (2, 2, 10, 10)), 2)  # unrelated box

    metric = BoundaryFScore(tolerance=2)
    metric.update_state(y_true, y_pred)
    assert float(metric.result()) < 0.5


def test_boundary_f_score_both_uniform_is_perfect_not_undefined():
    """Two all-background images have no boundary to disagree about -
    should score a perfect match, not divide-by-zero/NaN.
    """
    y_true = tf.one_hot(np.zeros((1, 16, 16), dtype="int32"), 2)
    y_pred = tf.one_hot(np.zeros((1, 16, 16), dtype="int32"), 2)

    metric = BoundaryFScore(tolerance=2)
    metric.update_state(y_true, y_pred)
    result = float(metric.result())
    assert np.isfinite(result)
    np.testing.assert_allclose(result, 1.0)


def test_boundary_f_score_accumulates_across_batches():
    y_true = tf.one_hot(_rect_mask((1, 32, 32), (8, 8, 24, 24)), 2)

    metric = BoundaryFScore(tolerance=2)
    metric.update_state(y_true, y_true)  # perfect match, score 1.0
    metric.update_state(y_true, tf.one_hot(_rect_mask((1, 32, 32), (2, 2, 10, 10)), 2))  # bad match

    result = float(metric.result())
    assert 0.0 < result < 1.0  # average of a good and a bad match


def test_boundary_f_score_reset_state():
    y_true = tf.one_hot(_rect_mask((1, 32, 32), (8, 8, 24, 24)), 2)
    metric = BoundaryFScore()
    metric.update_state(y_true, y_true)
    assert float(metric.result()) > 0
    metric.reset_state()
    np.testing.assert_allclose(float(metric.result()), 0.0)


def test_boundary_f_score_config_roundtrip():
    metric = BoundaryFScore(tolerance=5)
    restored = BoundaryFScore.from_config(metric.get_config())
    assert restored.tolerance == 5


# ========================= #
# MeanAveragePrecision


def test_map_perfect_predictions_scores_one():
    gt_boxes = np.array([[[0, 0, 10, 10], [20, 20, 30, 30]]], dtype="float32")
    gt_classes = np.array([[0, 0]], dtype="int64")
    gt_valid = np.array([2], dtype="int64")

    pred_boxes = np.array([[[0, 0, 10, 10], [20, 20, 30, 30], [100, 100, 110, 110]]], dtype="float32")
    pred_scores = np.array([[0.9, 0.8, 0.5]], dtype="float32")
    pred_classes = np.array([[0, 0, 0]], dtype="int64")
    valid_detections = np.array([3], dtype="int64")

    metric = MeanAveragePrecision(n_classes=1, iou_threshold=0.5)
    metric.update_state(
        (gt_boxes, gt_classes, gt_valid), (pred_boxes, pred_scores, pred_classes, valid_detections),
    )
    np.testing.assert_allclose(float(metric.result()), 1.0, atol=1e-4)


def test_map_missed_ground_truth_scores_below_one():
    gt_boxes = np.array([[[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]]], dtype="float32")
    gt_classes = np.array([[0, 0, 0]], dtype="int64")
    gt_valid = np.array([3], dtype="int64")

    pred_boxes = np.array([[[0, 0, 10, 10], [20, 20, 30, 30], [100, 100, 110, 110]]], dtype="float32")
    pred_scores = np.array([[0.9, 0.8, 0.5]], dtype="float32")
    pred_classes = np.array([[0, 0, 0]], dtype="int64")
    valid_detections = np.array([3], dtype="int64")

    metric = MeanAveragePrecision(n_classes=1, iou_threshold=0.5)
    metric.update_state(
        (gt_boxes, gt_classes, gt_valid), (pred_boxes, pred_scores, pred_classes, valid_detections),
    )
    result = float(metric.result())
    assert 0.0 < result < 1.0


def test_map_no_predictions_scores_zero():
    gt_boxes = np.array([[[0, 0, 10, 10]]], dtype="float32")
    gt_classes = np.array([[0]], dtype="int64")
    gt_valid = np.array([1], dtype="int64")

    pred_boxes = np.zeros((1, 0, 4), dtype="float32")
    pred_scores = np.zeros((1, 0), dtype="float32")
    pred_classes = np.zeros((1, 0), dtype="int64")
    valid_detections = np.array([0], dtype="int64")

    metric = MeanAveragePrecision(n_classes=1, iou_threshold=0.5)
    metric.update_state(
        (gt_boxes, gt_classes, gt_valid), (pred_boxes, pred_scores, pred_classes, valid_detections),
    )
    np.testing.assert_allclose(float(metric.result()), 0.0)


def test_map_no_ground_truth_for_a_class_is_excluded_not_zero():
    """A class with zero ground-truth instances anywhere shouldn't drag the
    mean down to 0 for that class - it should be excluded from the
    per-class average entirely (undefined, not "worst possible").
    """
    gt_boxes = np.array([[[0, 0, 10, 10]]], dtype="float32")
    gt_classes = np.array([[0]], dtype="int64")  # only class 0 ever appears
    gt_valid = np.array([1], dtype="int64")

    pred_boxes = np.array([[[0, 0, 10, 10]]], dtype="float32")
    pred_scores = np.array([[0.9]], dtype="float32")
    pred_classes = np.array([[0]], dtype="int64")
    valid_detections = np.array([1], dtype="int64")

    metric = MeanAveragePrecision(n_classes=2, iou_threshold=0.5)  # class 1 never appears in GT
    metric.update_state(
        (gt_boxes, gt_classes, gt_valid), (pred_boxes, pred_scores, pred_classes, valid_detections),
    )
    np.testing.assert_allclose(float(metric.result()), 1.0, atol=1e-4)


def test_map_higher_iou_threshold_is_stricter():
    gt_boxes = np.array([[[0, 0, 10, 10]]], dtype="float32")
    gt_classes = np.array([[0]], dtype="int64")
    gt_valid = np.array([1], dtype="int64")

    # A prediction that overlaps ~64% with the GT box (IoU between 0.5 and 0.9).
    pred_boxes = np.array([[[1, 1, 11, 11]]], dtype="float32")
    pred_scores = np.array([[0.9]], dtype="float32")
    pred_classes = np.array([[0]], dtype="int64")
    valid_detections = np.array([1], dtype="int64")

    metric_loose = MeanAveragePrecision(n_classes=1, iou_threshold=0.5)
    metric_loose.update_state(
        (gt_boxes, gt_classes, gt_valid), (pred_boxes, pred_scores, pred_classes, valid_detections),
    )
    metric_strict = MeanAveragePrecision(n_classes=1, iou_threshold=0.9)
    metric_strict.update_state(
        (gt_boxes, gt_classes, gt_valid), (pred_boxes, pred_scores, pred_classes, valid_detections),
    )

    assert float(metric_loose.result()) > float(metric_strict.result())


def test_map_reset_state():
    gt_boxes = np.array([[[0, 0, 10, 10]]], dtype="float32")
    gt_classes = np.array([[0]], dtype="int64")
    gt_valid = np.array([1], dtype="int64")
    pred_boxes = np.array([[[0, 0, 10, 10]]], dtype="float32")
    pred_scores = np.array([[0.9]], dtype="float32")
    pred_classes = np.array([[0]], dtype="int64")
    valid_detections = np.array([1], dtype="int64")

    metric = MeanAveragePrecision(n_classes=1)
    metric.update_state(
        (gt_boxes, gt_classes, gt_valid), (pred_boxes, pred_scores, pred_classes, valid_detections),
    )
    assert float(metric.result()) > 0
    metric.reset_state()
    np.testing.assert_allclose(float(metric.result()), 0.0)


def test_map_config_roundtrip():
    metric = MeanAveragePrecision(n_classes=7, iou_threshold=0.75)
    restored = MeanAveragePrecision.from_config(metric.get_config())
    assert restored.n_classes == 7
    assert restored.iou_threshold == 0.75


def test_map_output_matches_fcos_lite_detection_shape():
    """Regression/documentation test confirming `MeanAveragePrecision`'s
    `y_pred` contract is literally `fcos_lite_detection`'s (§9.3) own output
    tuple shape, so the two plug together with no reshaping - build a tiny
    detector and feed its raw output straight into the metric.
    """
    from swiss_army_keras._model_detection import fcos_lite_detection

    inp = tf.keras.Input((128, 128, 3))
    model = fcos_lite_detection(inp, n_classes=3, backbone="MobileNetV2", weights=None, freeze_backbone=False)
    x = np.random.rand(1, 128, 128, 3).astype("float32")
    pred_boxes, pred_scores, pred_classes, valid_detections = model.predict(x, verbose=0)

    gt_boxes = np.array([[[10.0, 10.0, 40.0, 40.0]]], dtype="float32")
    gt_classes = np.array([[0]], dtype="int64")
    gt_valid = np.array([1], dtype="int64")

    metric = MeanAveragePrecision(n_classes=3)
    metric.update_state(
        (gt_boxes, gt_classes, gt_valid), (pred_boxes, pred_scores, pred_classes, valid_detections),
    )
    assert np.isfinite(float(metric.result()))


# Monocular depth metrics: hand-computed values, the missing-depth mask and
# the median alignment used for depth predicted up to scale.


def _depth_pair():
    # one image, four pixels; the last target is 0 = missing
    y_true = tf.constant([[[[2.0], [4.0]], [[8.0], [0.0]]]])
    y_pred = tf.constant([[[[2.0], [5.0]], [[4.0], [100.0]]]])
    return y_true, y_pred


def test_depth_metrics_hand_computed_and_ignore_missing_pixels():
    y_true, y_pred = _depth_pair()

    abs_rel = DepthAbsRel()
    abs_rel.update_state(y_true, y_pred)
    np.testing.assert_allclose(float(abs_rel.result()), (0.0 + 0.25 + 0.5) / 3, rtol=1e-6)

    rmse = DepthRMSE()
    rmse.update_state(y_true, y_pred)
    np.testing.assert_allclose(float(rmse.result()), np.sqrt((0.0 + 1.0 + 16.0) / 3), rtol=1e-6)

    rmse_log = DepthRMSE(log=True)
    rmse_log.update_state(y_true, y_pred)
    expected = np.sqrt((0.0 + np.log(5 / 4) ** 2 + np.log(4 / 8) ** 2) / 3)
    np.testing.assert_allclose(float(rmse_log.result()), expected, rtol=1e-5)
    assert rmse_log.name == "depth_rmse_log"

    # ratios: 1.0, 1.25, 2.0 -> only the first is strictly below 1.25
    delta = DepthDeltaAccuracy(threshold=1.25)
    delta.update_state(y_true, y_pred)
    np.testing.assert_allclose(float(delta.result()), 1 / 3, rtol=1e-6)
    delta2 = DepthDeltaAccuracy(threshold=1.25 ** 2)
    delta2.update_state(y_true, y_pred)
    np.testing.assert_allclose(float(delta2.result()), 2 / 3, rtol=1e-6)


def test_depth_metrics_accumulate_over_pixels_and_reset():
    y_true, y_pred = _depth_pair()
    metric = DepthAbsRel()
    metric.update_state(y_true, y_pred)
    metric.update_state(tf.constant([[[[1.0]]]]), tf.constant([[[[2.0]]]]))  # one more valid pixel, error 1.0
    np.testing.assert_allclose(float(metric.result()), (0.0 + 0.25 + 0.5 + 1.0) / 4, rtol=1e-6)
    metric.reset_state()
    assert float(metric.result()) == 0.0


def test_depth_median_alignment_removes_a_global_scale():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(1.0, 10.0, size=(3, 8, 8, 1)).astype("float32")
    y_true[:, 0] = 0.0  # a row of missing depth
    y_pred = y_true * np.array([0.1, 3.0, 7.0], dtype="float32").reshape(3, 1, 1, 1)
    y_pred[:, 0] = 55.0  # garbage where depth is missing must not matter

    raw = DepthAbsRel()
    raw.update_state(y_true, y_pred)
    aligned = DepthAbsRel(align="median")
    aligned.update_state(y_true, y_pred)
    assert float(raw.result()) > 0.5
    assert float(aligned.result()) < 1e-5

    delta = DepthDeltaAccuracy(align="median")
    delta.update_state(y_true, y_pred)
    assert float(delta.result()) == 1.0


def test_depth_metric_config_roundtrip_and_bad_align():
    metric = DepthDeltaAccuracy(threshold=1.5, align="median")
    clone = DepthDeltaAccuracy.from_config(metric.get_config())
    assert clone.threshold == 1.5 and clone.align == "median"
    assert DepthRMSE.from_config(DepthRMSE(log=True).get_config()).log is True
    with pytest.raises(ValueError, match="align"):
        DepthAbsRel(align="mean")
