'''Metrics: `MeanIOU` (one-hot aware mean IoU), `BoundaryFScore` and
`MeanAveragePrecision`.
'''
import numpy as np
import tensorflow as tf

from scipy.ndimage import binary_erosion, distance_transform_edt


class MeanIOU(tf.keras.metrics.MeanIoU):
    '''`tf.keras.metrics.MeanIoU` for one-hot labels and per-class score maps:
    both `y_true` and `y_pred` are reduced with `argmax` over the last axis
    before the confusion matrix is updated, so it can be used directly with
    the segmentation models and data loaders of this library (logits or
    probabilities).
    '''
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


# ========================= #
# Boundary F-score (plan.md §6/§13 Phase 6)


def _boundary_map_np(mask):
    '''`mask`: `(H, W)` binary numpy array -> `1` at boundary pixels
    (pixels in the mask whose 4/8-neighborhood isn't entirely mask), `0`
    elsewhere. A morphological-gradient boundary, `mask & ~erode(mask)`.
    '''
    mask_bool = mask.astype(bool)
    eroded = binary_erosion(mask_bool)
    return (mask_bool & ~eroded).astype(np.float32)


def _bf_score_batch_np(y_true, y_pred, tolerance):
    '''`y_true`, `y_pred`: `(batch, H, W)` binary numpy arrays. Returns a
    `(batch,)` array of per-image BF-scores (boundary F1, matched within
    `tolerance` pixels via a signed/unsigned distance transform).
    '''
    tolerance = float(tolerance)
    scores = np.zeros(y_true.shape[0], dtype=np.float32)

    for b in range(y_true.shape[0]):
        true_boundary = _boundary_map_np(y_true[b])
        pred_boundary = _boundary_map_np(y_pred[b])

        true_has_boundary = true_boundary.sum() > 0
        pred_has_boundary = pred_boundary.sum() > 0

        if not true_has_boundary and not pred_has_boundary:
            # Both masks are uniform (e.g. an all-background image) - no
            # boundary to disagree about, a perfect (not undefined) match.
            scores[b] = 1.0
            continue
        if not true_has_boundary or not pred_has_boundary:
            scores[b] = 0.0
            continue

        dist_to_true_boundary = distance_transform_edt(1 - true_boundary)
        dist_to_pred_boundary = distance_transform_edt(1 - pred_boundary)

        precision = float((dist_to_true_boundary[pred_boundary.astype(bool)] <= tolerance).mean())
        recall = float((dist_to_pred_boundary[true_boundary.astype(bool)] <= tolerance).mean())

        scores[b] = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

    return scores


class BoundaryFScore(tf.keras.metrics.Metric):
    '''
    Boundary F-score (BF-score) — precision/recall of predicted *boundary*
    pixels matching ground-truth boundary pixels within `tolerance` pixels
    (via a distance transform, same technique `losses.py`'s `boundary_loss`
    already uses for its own signed-distance-map computation), combined
    into an F1. Segmentation quality metrics that only look at region
    overlap (`MeanIOU` above, Dice) can rate two masks similarly despite
    very different boundary precision on classes with a genuinely soft/
    diffuse edge (translucent or reflective materials, thin structures) —
    this metric targets exactly that gap.

    BoundaryFScore(n_classes, tolerance=2, name='boundary_f_score')

    ----------
    Csurka, G., Larlus, D. and Perronnin, F., 2013. What is a good
    evaluation measure for semantic segmentation?. In BMVC.

    Input
    ----------
        tolerance: matching distance, in pixels, a predicted boundary pixel
                   is allowed to be from the nearest true boundary pixel
                   (and vice versa) and still count as correct. 2, a common
                   default in the segmentation-boundary-metric literature.

    * **Simplification**: computed on a single foreground/background
      boundary (`argmax(y) > 0` vs. `== 0`), not per-class boundaries
      averaged together — a genuine simplification for the multi-class
      case, not a full per-class BF-score. Sufficient for the common case
      of one primary class-of-interest against everything else (translucent/
      reflective object vs. background); extend to per-class averaging
      yourself if multiple classes' boundaries all need separate scoring.
    * Uses `scipy.ndimage` (`binary_erosion`, `distance_transform_edt`) via
      `tf.numpy_function`, the same pattern `losses.py`'s `boundary_loss`
      already established — not a differentiable computation (this is a
      metric, evaluated in `training=False`/eval contexts, not backpropagated
      through), and per-image, not batched, since the distance transform
      has no native batched form.
    '''

    def __init__(self, tolerance=2, name='boundary_f_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tolerance = tolerance
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_labels = tf.argmax(y_true, axis=-1)
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        y_true_bin = tf.cast(y_true_labels > 0, tf.float32)
        y_pred_bin = tf.cast(y_pred_labels > 0, tf.float32)

        scores = tf.numpy_function(_bf_score_batch_np, [y_true_bin, y_pred_bin, self.tolerance], tf.float32)
        scores.set_shape([None])

        if sample_weight is not None:
            sample_weight = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
            self.total.assign_add(tf.reduce_sum(scores * sample_weight))
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.total.assign_add(tf.reduce_sum(scores))
            self.count.assign_add(tf.cast(tf.shape(scores)[0], tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({'tolerance': self.tolerance})
        return config


# ========================= #
# COCO-style mean Average Precision (plan.md §6/§13 Phase 6)


def _compute_ap_from_pr(recall, precision):
    '''COCO's own 101-point interpolated AP: at each of 101 evenly-spaced
    recall levels (0, 0.01, ..., 1.0), take the *maximum* precision at any
    recall >= that level, then average those 101 values. Monotonic
    precision envelope + fixed recall sampling, rather than a plain
    trapezoidal area under the raw (jagged) PR curve.
    '''
    recall_levels = np.linspace(0.0, 1.0, 101)
    if len(recall) == 0:
        return 0.0

    # Precision envelope: precision[i] = max(precision[i:]) - makes the
    # curve monotonically non-increasing as recall grows, COCO's own
    # convention (matches the standard `pycocotools` implementation).
    envelope = np.maximum.accumulate(precision[::-1])[::-1]

    interpolated = np.zeros_like(recall_levels)
    for i, r in enumerate(recall_levels):
        idx = np.searchsorted(recall, r, side='left')
        if idx < len(envelope):
            interpolated[i] = envelope[idx]
    return float(interpolated.mean())


def _box_iou_matrix_np(boxes_a, boxes_b):
    '''`(N, 4)` x `(M, 4)` -> `(N, M)` pairwise IoU, `(x_min, y_min, x_max, y_max)` boxes.'''
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])

    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[..., 0] * wh[..., 1]

    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


class MeanAveragePrecision(tf.keras.metrics.Metric):
    '''
    Mean Average Precision (mAP) at a single IoU threshold, COCO-style
    per-class AP (101-point interpolation, `_compute_ap_from_pr`) averaged
    across classes.

    MeanAveragePrecision(n_classes, iou_threshold=0.5, name='mAP')

    ----------
    Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D.,
    Dollár, P. and Zitnick, C.L., 2014. Microsoft COCO: Common objects in
    context. In ECCV. (defines the AP-computation protocol this follows).

    Input
    ----------
        n_classes: number of object classes.
        iou_threshold: the single IoU threshold a detection must clear
                       against a ground-truth box to count as a match
                       (0.5, the classic "mAP@0.5"/PASCAL-VOC-style value).

    Usage
    ----------
        `update_state(y_true, y_pred)`, where:
        y_true: `(gt_boxes, gt_classes, gt_valid_counts)` — `gt_boxes`
                shape `(batch, max_boxes, 4)`, `gt_classes` shape
                `(batch, max_boxes)`, `gt_valid_counts` shape `(batch,)`
                (how many of each image's `max_boxes` slots hold a real
                box — the rest are padding, ignored).
        y_pred: `(pred_boxes, pred_scores, pred_classes, valid_detections)`
                — exactly `fcos_lite_detection`'s (§9.3) own output tuple
                shape, so this metric plugs directly into that model's
                predictions with no reshaping.

    * **Scope note**: this computes AP at *one* IoU threshold, not COCO's
      full primary metric (the average of AP at 10 thresholds, 0.5:0.05:0.95)
      — a deliberate, documented scope reduction for a first pass (mirroring
      plan.md §9.3's own FCOS-lite-not-GFL precedent), not an attempt at the
      complete COCO protocol. Extending to the full sweep is mechanically
      straightforward (run this at each of the 10 thresholds and average
      the results) but multiplies the per-`result()`-call cost by 10x, so
      it's left as an explicit opt-in rather than the default here.
    * **Not a `tf.Variable`-based streaming metric internally** — AP
      requires a *global*, cross-batch sort of every detection by
      confidence before precision/recall can be computed at all, which
      doesn't fit the "accumulate into a running scalar" pattern
      `tf.Variable`-based metrics (including `BoundaryFScore` above) use.
      Predictions and ground truth are instead accumulated in plain Python
      lists across `update_state` calls, and AP is computed once, in
      `result()` — the same accumulation strategy widely-used object
      detection metric implementations use elsewhere, for the same reason.
    '''

    def __init__(self, n_classes, iou_threshold=0.5, name='mAP', **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.iou_threshold = iou_threshold
        self._per_class_detections = {c: [] for c in range(n_classes)}  # list of (score, is_tp)
        self._per_class_gt_count = {c: 0 for c in range(n_classes)}

    def update_state(self, y_true, y_pred, sample_weight=None):
        gt_boxes, gt_classes, gt_valid_counts = y_true
        pred_boxes, pred_scores, pred_classes, valid_detections = y_pred

        gt_boxes = np.asarray(gt_boxes)
        gt_classes = np.asarray(gt_classes)
        gt_valid_counts = np.asarray(gt_valid_counts)
        pred_boxes = np.asarray(pred_boxes)
        pred_scores = np.asarray(pred_scores)
        pred_classes = np.asarray(pred_classes)
        valid_detections = np.asarray(valid_detections)

        batch_size = gt_boxes.shape[0]
        for b in range(batch_size):
            n_gt = int(gt_valid_counts[b])
            n_pred = int(valid_detections[b])

            image_gt_boxes = gt_boxes[b, :n_gt]
            image_gt_classes = gt_classes[b, :n_gt].astype(int)
            image_pred_boxes = pred_boxes[b, :n_pred]
            image_pred_scores = pred_scores[b, :n_pred]
            image_pred_classes = pred_classes[b, :n_pred].astype(int)

            for c in range(image_gt_classes.shape[0]):
                self._per_class_gt_count[image_gt_classes[c]] += 1

            for c in range(self.n_classes):
                class_gt_mask = image_gt_classes == c
                class_gt_boxes = image_gt_boxes[class_gt_mask]
                class_pred_mask = image_pred_classes == c
                class_pred_boxes = image_pred_boxes[class_pred_mask]
                class_pred_scores = image_pred_scores[class_pred_mask]

                if len(class_pred_boxes) == 0:
                    continue

                order = np.argsort(-class_pred_scores)
                class_pred_boxes = class_pred_boxes[order]
                class_pred_scores = class_pred_scores[order]

                matched_gt = np.zeros(len(class_gt_boxes), dtype=bool)
                ious = _box_iou_matrix_np(class_pred_boxes, class_gt_boxes)

                for pred_idx in range(len(class_pred_boxes)):
                    is_tp = False
                    if len(class_gt_boxes) > 0:
                        best_gt_idx = np.argmax(ious[pred_idx])
                        best_iou = ious[pred_idx, best_gt_idx]
                        if best_iou >= self.iou_threshold and not matched_gt[best_gt_idx]:
                            matched_gt[best_gt_idx] = True
                            is_tp = True
                    self._per_class_detections[c].append((float(class_pred_scores[pred_idx]), is_tp))

    def result(self):
        average_precisions = []
        for c in range(self.n_classes):
            n_gt = self._per_class_gt_count[c]
            detections = self._per_class_detections[c]

            if n_gt == 0:
                continue
            if len(detections) == 0:
                average_precisions.append(0.0)
                continue

            detections = sorted(detections, key=lambda d: -d[0])
            tp = np.array([1.0 if d[1] else 0.0 for d in detections])
            fp = 1.0 - tp

            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recall = tp_cumsum / n_gt
            precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-12)

            average_precisions.append(_compute_ap_from_pr(recall, precision))

        if len(average_precisions) == 0:
            return tf.constant(0.0)
        return tf.constant(float(np.mean(average_precisions)))

    def reset_state(self):
        self._per_class_detections = {c: [] for c in range(self.n_classes)}
        self._per_class_gt_count = {c: 0 for c in range(self.n_classes)}

    def get_config(self):
        config = super().get_config()
        config.update({'n_classes': self.n_classes, 'iou_threshold': self.iou_threshold})
        return config
