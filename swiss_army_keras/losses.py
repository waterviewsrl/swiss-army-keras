'''Loss functions for segmentation (Dice, Tversky, focal variants, Lovasz,
boundary, structure, OHEM), detection (IoU family, QFL/DFL/VFL), depth
(scale-invariant log, BerHu), instance masks and ranking, plus the
`loss_adapter` and `deep_supervision_loss` wrappers.
'''
from __future__ import absolute_import

import math

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from scipy.ndimage import distance_transform_edt


def _crps_tf(y_true, y_pred, factor=0.05):
    '''
    core of (pseudo) CRPS loss.

    y_true: two-dimensional arrays
    y_pred: two-dimensional arrays
    factor: importance of std term
    '''

    # mean absolute error
    mae = K.mean(tf.abs(y_pred - y_true))

    dist = tf.math.reduce_std(y_pred)

    return mae - factor*dist


def crps2d_tf(y_true, y_pred, factor=0.05):
    '''
    (Experimental)
    An approximated continuous ranked probability score (CRPS) loss function:

        CRPS = mean_abs_err - factor * std

    * Note that the "real CRPS" = mean_abs_err - mean_pairwise_abs_diff

     Replacing mean pairwise absolute difference by standard deviation offers
     a complexity reduction from O(N^2) to O(N*logN)

    ** factor > 0.1 may yield negative loss values.

    Compatible with high-level Keras training methods

    Input
    ----------
        y_true: training target with shape=(batch_num, x, y, 1)
        y_pred: a forward pass with shape=(batch_num, x, y, 1)
        factor: relative importance of standard deviation term.

    '''

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    batch_num = y_pred.shape.as_list()[0]

    crps_out = 0
    for i in range(batch_num):
        crps_out += _crps_tf(y_true[i, ...], y_pred[i, ...], factor=factor)

    return crps_out/batch_num


def _crps_np(y_true, y_pred, factor=0.05):
    '''
    Numpy version of _crps_tf.
    '''

    # mean absolute error
    mae = np.nanmean(np.abs(y_pred - y_true))
    dist = np.nanstd(y_pred)

    return mae - factor*dist


def crps2d_np(y_true, y_pred, factor=0.05):
    '''
    (Experimental)
    Nunpy version of `crps2d_tf`.

    Documentation refers to `crps2d_tf`.
    '''

    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    batch_num = len(y_pred)

    crps_out = 0
    for i in range(batch_num):
        crps_out += _crps_np(y_true[i, ...], y_pred[i, ...], factor=factor)

    return crps_out/batch_num

# ========================= #
# Dice loss and variants


def dice_coef(y_true, y_pred, const=K.epsilon()):
    '''
    Sørensen–Dice coefficient for 2-d samples.

    Input
    ----------
        y_true, y_pred: predicted outputs and targets.
        const: a constant that smooths the loss gradient and reduces numerical instabilities.

    '''

    # flatten 2-d tensors
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])

    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))
    false_pos = tf.reduce_sum((1-y_true_pos) * y_pred_pos)

    # 2TP/(2TP+FP+FN) == 2TP/()
    coef_val = (2.0 * true_pos + const) / \
        (2.0 * true_pos + false_pos + false_neg)

    return coef_val


def dice(y_true, y_pred, const=K.epsilon()):
    '''
    Sørensen–Dice Loss.

    dice(y_true, y_pred, const=K.epsilon())

    Input
    ----------
        const: a constant that smooths the loss gradient and reduces numerical instabilities.

    '''
    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # <--- squeeze-out length-1 dimensions.
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    loss_val = 1 - dice_coef(y_true, y_pred, const=const)

    return loss_val

# ========================= #
# Tversky loss and variants


def tversky_coef(y_true, y_pred, alpha=0.5, const=K.epsilon()):
    '''
    Weighted Sørensen–Dice coefficient.

    Input
    ----------
        y_true, y_pred: predicted outputs and targets.
        const: a constant that smooths the loss gradient and reduces numerical instabilities.

    '''

    # flatten 2-d tensors
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])

    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))
    false_pos = tf.reduce_sum((1-y_true_pos) * y_pred_pos)

    # TP/(TP + a*FN + b*FP); a+b = 1
    coef_val = (true_pos + const)/(true_pos + alpha *
                                   false_neg + (1-alpha)*false_pos + const)

    return coef_val


def tversky(y_true, y_pred, alpha=0.5, const=K.epsilon()):
    '''
    Tversky Loss.

    tversky(y_true, y_pred, alpha=0.5, const=K.epsilon())

    ----------
    Hashemi, S.R., Salehi, S.S.M., Erdogmus, D., Prabhu, S.P., Warfield, S.K. and Gholipour, A., 2018.
    Tversky as a loss function for highly unbalanced image segmentation using 3d fully convolutional deep networks.
    arXiv preprint arXiv:1803.11078.

    Input
    ----------
        alpha: tunable parameter within [0, 1]. Alpha handles imbalance classification cases.
        const: a constant that smooths the loss gradient and reduces numerical instabilities.

    '''
    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # <--- squeeze-out length-1 dimensions.
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    loss_val = 1 - tversky_coef(y_true, y_pred, alpha=alpha, const=const)

    return loss_val


def focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3, const=K.epsilon()):
    '''
    Focal Tversky Loss (FTL)

    focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3)

    ----------
    Abraham, N. and Khan, N.M., 2019, April. A novel focal tversky loss function with improved
    attention u-net for lesion segmentation. In 2019 IEEE 16th International Symposium on Biomedical Imaging
    (ISBI 2019) (pp. 683-687). IEEE.

    ----------
    Input
        alpha: tunable parameter within [0, 1]. Alpha handles imbalance classification cases
        gamma: tunable parameter within [1, 3].
        const: a constant that smooths the loss gradient and reduces numerical instabilities.

    '''
    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # <--- squeeze-out length-1 dimensions.
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    # (Tversky loss)**(1/gamma)
    loss_val = tf.math.pow(
        (1-tversky_coef(y_true, y_pred, alpha=alpha, const=const)), 1/gamma)

    return loss_val

# ========================= #
# MS-SSIM


def ms_ssim(y_true, y_pred, **kwargs):
    """
    Multiscale structural similarity (MS-SSIM) loss.

    `ms_ssim(y_true, y_pred, **tf_ssim_kw)`

    ----------
    Wang, Z., Simoncelli, E.P. and Bovik, A.C., 2003, November. Multiscale structural similarity for image quality assessment.
    In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003 (Vol. 2, pp. 1398-1402). Ieee.

    ----------
    Input
        kwargs: keywords of `tf.image.ssim_multiscale`
                https://www.tensorflow.org/api_docs/python/tf/image/ssim_multiscale

        *Issues of `tf.image.ssim_multiscale`refers to:
                https://stackoverflow.com/questions/57127626/error-in-calculation-of-inbuilt-ms-ssim-function-in-tensorflow

    """

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    tf_ms_ssim = tf.image.ssim_multiscale(y_true, y_pred, **kwargs)

    return 1 - tf_ms_ssim

# ======================== #


def iou_box_coef(y_true, y_pred, mode='giou', dtype=tf.float32):
    """
    Inersection over Union (IoU) and generalized/complete/SCYLLA IoU
    coefficients for bounding boxes.

    iou_box_coef(y_true, y_pred, mode='giou', dtype=tf.float32)

    ----------
    Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I. and Savarese, S., 2019.
    Generalized intersection over union: A metric and a loss for bounding box regression.
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 658-666).

    Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R. and Ren, D., 2020.
    Distance-IoU Loss: Faster and Better Learning for Bounding Box
    Regression. In AAAI (introduces CIoU, `mode='ciou'`).

    Gevorgyan, Z., 2022. SIoU Loss: More Powerful Learning for Bounding Box
    Regression. arXiv:2205.12740 (`mode='siou'`).

    ----------
    Input
        y_true: the target bounding box.
        y_pred: the predicted bounding box.

        Elements of a bounding box should be organized as: [y_min, x_min, y_max, x_max].

        mode: 'iou' for IoU coeff (i.e., Jaccard index);
              'giou' for generalized IoU coeff;
              'ciou' for complete IoU coeff (adds a center-distance penalty
                     and an aspect-ratio-consistency penalty on top of IoU —
                     converges faster and localizes more precisely than
                     plain IoU/GIoU, particularly early in training when
                     predicted/target boxes may not even overlap, since GIoU
                     alone degrades toward a weak signal in that regime);
              'siou' for SCYLLA-IoU coeff (adds an *angle* cost — penalizing
                     the predicted box center's angular offset from the
                     target's — alongside the distance and shape costs CIoU
                     already has; the paper reports faster convergence than
                     CIoU on top of its own accuracy gains, at the cost of
                     one more hyperparameter, `siou_theta`, the shape-cost
                     exponent).

        dtype: the data type of input tensors.
               Default is tf.float32.

    """

    zero = tf.convert_to_tensor(0.0, dtype)
    eps = tf.convert_to_tensor(1e-7, dtype)

    # subtrack bounding box coords
    ymin_true, xmin_true, ymax_true, xmax_true = tf.unstack(y_true, 4, axis=-1)
    ymin_pred, xmin_pred, ymax_pred, xmax_pred = tf.unstack(y_pred, 4, axis=-1)

    # true area
    w_true = tf.maximum(zero, xmax_true - xmin_true)
    h_true = tf.maximum(zero, ymax_true - ymin_true)
    area_true = w_true * h_true

    # pred area
    w_pred = tf.maximum(zero, xmax_pred - xmin_pred)
    h_pred = tf.maximum(zero, ymax_pred - ymin_pred)
    area_pred = w_pred * h_pred

    # intersections
    intersect_ymin = tf.maximum(ymin_true, ymin_pred)
    intersect_xmin = tf.maximum(xmin_true, xmin_pred)
    intersect_ymax = tf.minimum(ymax_true, ymax_pred)
    intersect_xmax = tf.minimum(xmax_true, xmax_pred)

    w_intersect = tf.maximum(zero, intersect_xmax - intersect_xmin)
    h_intersect = tf.maximum(zero, intersect_ymax - intersect_ymin)
    area_intersect = w_intersect * h_intersect

    # IoU
    area_union = area_true + area_pred - area_intersect
    iou = tf.math.divide_no_nan(area_intersect, area_union)

    if mode == "iou":

        return iou

    # encolsed coords - shared by 'giou'/'ciou'/'siou'
    enclose_ymin = tf.minimum(ymin_true, ymin_pred)
    enclose_xmin = tf.minimum(xmin_true, xmin_pred)
    enclose_ymax = tf.maximum(ymax_true, ymax_pred)
    enclose_xmax = tf.maximum(xmax_true, xmax_pred)

    w_enclose = tf.maximum(zero, enclose_xmax - enclose_xmin)
    h_enclose = tf.maximum(zero, enclose_ymax - enclose_ymin)

    if mode == "giou":

        area_enclose = w_enclose * h_enclose
        giou = iou - \
            tf.math.divide_no_nan((area_enclose - area_union), area_enclose)

        return giou

    # Box centers - shared by 'ciou'/'siou'.
    cx_true, cy_true = (xmin_true + xmax_true) / 2.0, (ymin_true + ymax_true) / 2.0
    cx_pred, cy_pred = (xmin_pred + xmax_pred) / 2.0, (ymin_pred + ymax_pred) / 2.0
    center_dx = cx_pred - cx_true
    center_dy = cy_pred - cy_true

    if mode == "ciou":

        # Squared diagonal of the smallest enclosing box, and squared
        # center distance - CIoU's distance penalty term.
        c2 = tf.square(w_enclose) + tf.square(h_enclose) + eps
        rho2 = tf.square(center_dx) + tf.square(center_dy)

        # Aspect-ratio-consistency penalty (`v`) and its trade-off weight
        # (`alpha`) - `alpha` is treated as a constant w.r.t. the gradient
        # (matching the original paper, which does not backpropagate
        # through it, for training stability), not because it's
        # numerically constant.
        v = (4.0 / (math.pi ** 2)) * tf.square(
            tf.atan(tf.math.divide_no_nan(w_true, h_true + eps))
            - tf.atan(tf.math.divide_no_nan(w_pred, h_pred + eps))
        )
        alpha = tf.stop_gradient(tf.math.divide_no_nan(v, (1.0 - iou) + v))

        return iou - tf.math.divide_no_nan(rho2, c2) - alpha * v

    if mode == "siou":

        sigma = tf.sqrt(tf.square(center_dx) + tf.square(center_dy)) + eps
        sin_alpha_1 = tf.abs(center_dx) / sigma
        sin_alpha_2 = tf.abs(center_dy) / sigma
        sin_alpha = tf.where(sin_alpha_1 > (2.0 ** 0.5) / 2.0, sin_alpha_2, sin_alpha_1)
        # Angle cost: 0 when the centers are aligned along an axis
        # (sin_alpha in {0, 1}), maximal at a 45-degree offset.
        angle_cost = tf.cos(tf.asin(tf.clip_by_value(sin_alpha, -1.0, 1.0)) * 2.0 - math.pi / 2.0)

        rho_x = tf.square(tf.math.divide_no_nan(center_dx, w_enclose + eps))
        rho_y = tf.square(tf.math.divide_no_nan(center_dy, h_enclose + eps))
        gamma = 2.0 - angle_cost
        distance_cost = 2.0 - tf.exp(-gamma * rho_x) - tf.exp(-gamma * rho_y)

        omega_w = tf.abs(w_pred - w_true) / (tf.maximum(w_pred, w_true) + eps)
        omega_h = tf.abs(h_pred - h_true) / (tf.maximum(h_pred, h_true) + eps)
        siou_theta = 4.0
        shape_cost = tf.pow(1.0 - tf.exp(-omega_w), siou_theta) + \
            tf.pow(1.0 - tf.exp(-omega_h), siou_theta)

        return iou - 0.5 * (distance_cost + shape_cost)

    raise ValueError(f"Unknown mode '{mode}' - expected one of 'iou', 'giou', 'ciou', 'siou'.")


def iou_box(y_true, y_pred, mode='giou', dtype=tf.float32):
    """
    Inersection over Union (IoU) and generalized IoU losses for bounding boxes.

    iou_box(y_true, y_pred, mode='giou', dtype=tf.float32)

    ----------
    Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I. and Savarese, S., 2019.
    Generalized intersection over union: A metric and a loss for bounding box regression.
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 658-666).

    ----------
    Input
        y_true: the target bounding box.
        y_pred: the predicted bounding box.

        Elements of a bounding box should be organized as: [y_min, x_min, y_max, x_max].

        mode: 'iou' for IoU coeff (i.e., Jaccard index);
              'giou' for generalized IoU coeff.

        dtype: the data type of input tensors.
               Default is tf.float32.

    """

    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.cast(y_pred, dtype)

    y_true = tf.cast(y_true, dtype)

    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    return 1 - iou_box_coef(y_true, y_pred, mode=mode, dtype=dtype)


def iou_seg(y_true, y_pred, dtype=tf.float32):
    """
    Inersection over Union (IoU) loss for segmentation maps.

    iou_seg(y_true, y_pred, dtype=tf.float32)

    ----------
    Rahman, M.A. and Wang, Y., 2016, December. Optimizing intersection-over-union in deep neural networks for
    image segmentation. In International symposium on visual computing (pp. 234-244). Springer, Cham.

    ----------
    Input
        y_true: segmentation targets, c.f. `keras.losses.categorical_crossentropy`
        y_pred: segmentation predictions.

        dtype: the data type of input tensors.
               Default is tf.float32.

    """

    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.cast(y_pred, dtype)
    y_true = tf.cast(y_true, y_pred.dtype)

    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])

    area_intersect = tf.reduce_sum(tf.multiply(y_true_pos, y_pred_pos))

    area_true = tf.reduce_sum(y_true_pos)
    area_pred = tf.reduce_sum(y_pred_pos)
    area_union = area_true + area_pred - area_intersect

    return 1-tf.math.divide_no_nan(area_intersect, area_union)

# ========================= #
# Lovász-Softmax loss


def _lovasz_grad(gt_sorted):
    '''Gradient of the Lovász extension of the Jaccard index w.r.t. sorted
    errors (Berman et al. 2018, eq. 3) — the piece that makes the surrogate
    below differentiable and IoU-consistent.
    '''
    gt_sorted = tf.cast(gt_sorted, tf.float32)
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1.0 - gt_sorted)
    jaccard = 1.0 - intersection / union
    jaccard = tf.concat([jaccard[0:1], jaccard[1:] - jaccard[:-1]], axis=0)
    return jaccard


def lovasz_softmax(y_true, y_pred, classes='present'):
    '''
    Lovász-Softmax loss — a differentiable surrogate that directly optimizes
    the (multi-class) Jaccard/IoU index, rather than a proxy like
    cross-entropy or Dice. Known to help on imbalanced/thin classes more
    than Dice/Tversky alone, since it's derived from the actual evaluation
    metric most segmentation tasks are scored on.

    lovasz_softmax(y_true, y_pred, classes='present')

    ----------
    Berman, M., Triki, A.R. and Blaschko, M.B., 2018. The Lovász-Softmax
    loss: a tractable surrogate for the optimization of the
    intersection-over-union measure in neural networks. In CVPR.

    Input
    ----------
        y_true: one-hot segmentation targets, shape (batch, H, W, n_classes).
        y_pred: softmax probabilities, same shape as `y_true`.
        classes: 'present' (default) averages only over classes with at
                 least one ground-truth pixel in the batch — the paper's own
                 recommendation, avoiding a spurious zero contribution from
                 classes wholly absent from a given batch; 'all' averages
                 over every class unconditionally.

    '''
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    n_classes = y_pred.shape[-1]
    probas = tf.reshape(y_pred, [-1, n_classes])
    labels = tf.argmax(tf.reshape(y_true, [-1, n_classes]), axis=-1)

    losses = []
    present_flags = []
    for c in range(n_classes):
        fg = tf.cast(tf.equal(labels, c), probas.dtype)
        present_flags.append(tf.reduce_sum(fg) > 0)

        errors = tf.abs(fg - probas[:, c])
        errors_sorted, perm = tf.math.top_k(errors, k=tf.shape(errors)[0])
        fg_sorted = tf.gather(fg, perm)
        grad = _lovasz_grad(fg_sorted)
        losses.append(tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1))

    losses = tf.stack(losses)
    if classes == 'present':
        present_mask = tf.cast(tf.stack(present_flags), losses.dtype)
        return tf.reduce_sum(losses * present_mask) / tf.maximum(tf.reduce_sum(present_mask), 1.0)
    return tf.reduce_mean(losses)

# ========================= #
# Boundary loss


def _signed_distance_map_np(mask):
    '''mask: (H, W) binary numpy array. Negative inside the foreground,
    positive outside — the sign convention `boundary_loss` below expects.
    '''
    posmask = mask.astype(bool)
    if posmask.any() and not posmask.all():
        neg_dist = distance_transform_edt(~posmask)
        pos_dist = distance_transform_edt(posmask)
        return (neg_dist - pos_dist).astype(np.float32)
    return np.zeros_like(mask, dtype=np.float32)


def _signed_distance_map_batch_np(y_true):
    '''y_true: (batch, H, W, n_classes) one-hot numpy array -> same-shape
    signed distance maps, computed independently per class.
    '''
    out = np.zeros_like(y_true, dtype=np.float32)
    for b in range(y_true.shape[0]):
        for c in range(y_true.shape[-1]):
            out[b, ..., c] = _signed_distance_map_np(y_true[b, ..., c])
    return out


def boundary_loss(y_true, y_pred):
    '''
    Boundary loss — directly penalizes predicted probability mass in
    proportion to its distance from the ground-truth boundary (via a
    precomputed signed distance map), rather than only comparing regions
    pixel-by-pixel the way Dice/Tversky/cross-entropy do. Pairs well with
    Dice for soft/diffuse boundaries typical of translucent or reflective
    classes, and doesn't degrade under severe class imbalance the way
    region-based losses alone can.

    boundary_loss(y_true, y_pred)

    ----------
    Kervadec, H., Bouchtiba, J., Desrosiers, C., Granger, E., Dolz, J. and
    Ben Ayed, I., 2019. Boundary loss for highly unbalanced segmentation.
    In International Conference on Medical Imaging with Deep Learning.

    Input
    ----------
        y_true: one-hot segmentation targets, shape (batch, H, W, n_classes).
        y_pred: softmax/sigmoid probabilities, same shape as `y_true`.

    * The signed distance map is computed once per batch via
      `scipy.ndimage.distance_transform_edt`, wrapped in `tf.numpy_function`
      since the distance transform itself is not a differentiable TF op —
      only `y_pred` carries a gradient here, the map itself is
      `tf.stop_gradient`'d, matching the paper's own formulation (a
      fixed, target-derived weighting, not a learned quantity). Meant to be
      combined with a region-based loss (e.g. `alpha * dice(...) + (1 -
      alpha) * boundary_loss(...)`, with `alpha` annealed toward the
      boundary term over training, per the paper) rather than used alone.

    '''
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    dist_map = tf.numpy_function(_signed_distance_map_batch_np, [y_true], tf.float32)
    dist_map = tf.stop_gradient(dist_map)
    dist_map.set_shape(y_pred.shape)

    return tf.reduce_mean(dist_map * y_pred)

# ========================= #
# Monocular depth losses (plan.md §13.5, Phase 11)


def scale_invariant_log_loss(y_true, y_pred, lam=0.5):
    '''
    Scale-invariant log loss for monocular depth regression — penalizes
    relative (log-space) depth error while being insensitive to a global
    multiplicative bias in the prediction, since absolute scale is
    generally unrecoverable from a single image.

    scale_invariant_log_loss(y_true, y_pred, lam=0.5)

    ----------
    Eigen, D., Puhrsch, C. and Fergus, R., 2014. Depth map prediction from
    a single image using a multi-scale deep network. NeurIPS.

    Eigen, D. and Fergus, R., 2015. Predicting depth, surface normals and
    semantic labels with a common multi-scale convolutional architecture.
    ICCV. (source of this function's own `lam=0.5` default, in place of the
    original paper's fully scale-invariant `lam=1.0`.)

    Input
    ----------
        y_true: ground-truth depth (or inverse depth), any positive-valued
                unit — pixels with `y_true <= 0` are treated as invalid/
                missing and excluded from the loss (a common convention for
                depth sensors with holes/missing returns).
        y_pred: predicted depth, same shape/units as `y_true`, must be
                strictly positive (clipped to `K.epsilon()` before the log).
        lam: trade-off between plain log-MSE (`lam=0`) and full scale
             invariance (`lam=1`) — `0.5` (this function's default) is
             Eigen & Fergus's own later-paper value, not the original
             paper's `1.0`.

    '''
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    valid_mask = tf.cast(y_true > 0, y_pred.dtype)
    y_pred_clipped = tf.maximum(y_pred, K.epsilon())
    y_true_clipped = tf.maximum(y_true, K.epsilon())

    e = (tf.math.log(y_pred_clipped) - tf.math.log(y_true_clipped)) * valid_mask
    n = tf.maximum(tf.reduce_sum(valid_mask), 1.0)

    return tf.reduce_sum(tf.square(e)) / n - lam * tf.square(tf.reduce_sum(e) / n)


def berhu_loss(y_true, y_pred):
    '''
    BerHu (reverse Huber) loss for monocular depth regression — L1 near
    zero error (preserves gradient signal on the many small-error pixels
    typical of depth regression), L2 beyond a per-batch threshold `c`
    (down-weights the gradient contribution of large outlier errors rather
    than letting them dominate, the way plain L1 would).

    berhu_loss(y_true, y_pred)

    ----------
    Laina, I., Rupprecht, C., Belagiannis, V., Tombari, F. and Navab, N.,
    2016. Deeper depth prediction with fully convolutional residual
    networks. 3DV.

    Input
    ----------
        y_true: ground-truth depth, same shape as `y_pred`.
        y_pred: predicted depth.

    * The threshold `c = 0.2 * max(|y_pred - y_true|)` is computed fresh
      per batch and `tf.stop_gradient`'d — a fixed, target-derived scalar
      by the paper's own formulation, not a learned quantity, the same
      "detach a target-derived statistic" idiom `boundary_loss` above
      already uses for its own distance map (no `tf.numpy_function` needed
      here, since `reduce_max` is already a plain, differentiable TF op
      that's being deliberately detached, not an external/non-differentiable
      computation).

    '''
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    x = y_pred - y_true
    abs_x = tf.abs(x)
    c = tf.stop_gradient(0.2 * tf.reduce_max(abs_x))

    quadratic = (tf.square(x) + tf.square(c)) / (2.0 * c + K.epsilon())
    loss = tf.where(abs_x <= c, abs_x, quadratic)

    return tf.reduce_mean(loss)


# ========================= #
# YOLACT instance-segmentation mask loss (plan.md §13.6, Phase 11)


def yolact_mask_loss(y_true, y_pred, box_area):
    '''Per-instance, area-normalized binary cross-entropy for YOLACT-style
    assembled instance masks (`swiss_army_keras._model_detection.
    fcos_lite_instance_segmentation`'s own `sigmoid(prototypes @
    coefficients)`-then-cropped-to-box masks) — YOLACT's own paper formula
    (Bolya, Zhou, Xiao, Lee, "YOLACT: Real-time Instance Segmentation",
    ICCV 2019, arXiv:1904.02689), confirmed from the reference
    implementation's own `multibox_loss.py`: sum the per-pixel BCE over
    each instance's own assembled mask, then divide by that instance's own
    ground-truth box area (in mask-resolution units) — not a plain mean —
    so small instances aren't drowned out by large ones the way a flat
    per-pixel average across a whole batch would.

    yolact_mask_loss(y_true, y_pred, box_area)

    ----------
    Bolya, D., Zhou, C., Xiao, F. and Lee, Y.J., 2019. YOLACT: Real-time
    instance segmentation. ICCV.

    Input
    ----------
        y_true: `(..., H, W)` ground-truth binary mask(s), resized to the
                prototype/mask resolution (e.g.
                `_model_detection.assign_instance_targets`'s own
                `mask_targets` output).
        y_pred: `(..., H, W)` predicted masks, already `sigmoid`-activated
                probabilities, same shape as `y_true`.
        box_area: `(...,)` — that instance's own ground-truth box area in
                  the *same* mask-resolution units as `y_true`/`y_pred`
                  (e.g. `assign_instance_targets`'s own `mask_areas` — pass
                  the matching per-instance value, not recomputed here).

    '''
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    box_area = tf.cast(box_area, y_pred.dtype)

    y_pred_clipped = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
    bce = -(y_true * tf.math.log(y_pred_clipped) + (1.0 - y_true) * tf.math.log(1.0 - y_pred_clipped))

    per_instance_sum = tf.reduce_sum(bce, axis=[-2, -1])
    per_instance_loss = per_instance_sum / tf.maximum(box_area, 1.0)

    return tf.reduce_mean(per_instance_loss)


# ========================= #
# PIDNet boundary losses (plan.md §3.4, Phase 9)


def _canny_boundary_map_np(label, dilate_size):
    '''label: (H, W) integer class-index map -> binary boundary map (H, W)
    float32, 1.0 at/near a class-boundary pixel. `cv2.Canny` run directly
    on the integer class-index map (not a photo) - works because a class
    boundary is exactly where consecutive label values jump, which Canny's
    gradient detector picks up as an edge - confirmed from the reference
    (`XuJiacong/PIDNet`, `datasets/base_dataset.py::gen_sample`:
    `edge = cv2.Canny(label, 0.1, 0.2); edge = (cv2.dilate(edge,
    ones((edge_size,edge_size)), iterations=1) > 50)`), not guessed.
    '''
    edge = cv2.Canny(label.astype(np.uint8), 0.1, 0.2)
    kernel = np.ones((dilate_size, dilate_size), np.uint8)
    return (cv2.dilate(edge, kernel, iterations=1) > 50).astype(np.float32)


def _canny_boundary_map_batch_np(y_true, dilate_size):
    '''y_true: (batch, H, W, n_classes) one-hot numpy array -> (batch, H,
    W, 1) binary boundary maps, derived from the class-index map
    (`argmax(y_true, axis=-1)`) - the boundary target is *derived*, not
    supplied separately, same "compute the target from y_true internally"
    pattern `boundary_loss`'s own `_signed_distance_map_batch_np` above
    already uses.
    '''
    class_map = np.argmax(y_true, axis=-1)
    out = np.zeros(y_true.shape[:3] + (1,), dtype=np.float32)
    for b in range(y_true.shape[0]):
        out[b, ..., 0] = _canny_boundary_map_np(class_map[b], dilate_size)
    return out


def pidnet_boundary_loss(y_true, y_pred, dilate_size=4):
    '''
    PIDNet's own boundary-branch supervision (term 3 of its 4-term total
    loss, plan.md §3.4) - a class-balanced weighted BCE against a
    Canny+dilate boundary map derived from `y_true` internally (via
    `tf.numpy_function`, `tf.stop_gradient`'d, same idiom as `boundary_loss`
    above), not a separately-supplied target.

    pidnet_boundary_loss(y_true, y_pred, dilate_size=4)

    ----------
    Xu, J., Xiong, Z. and Bhattacharyya, S.P., 2023. PIDNet: A real-time
    semantic segmentation network inspired by PID controllers. In CVPR.

    Input
    ----------
        y_true: one-hot segmentation labels, shape (batch, H, W, n_classes)
                - the *semantic* labels, not a boundary map; the boundary
                target is derived from these.
        y_pred: the D (boundary) branch's raw single-channel logit, shape
                (batch, H, W, 1) - matches `pidnet()`'s own `d_aux` output
                exactly (`augment=True`'s 3rd output), no activation
                applied.
        dilate_size: the boundary-map dilation kernel size - directly
                     controls the recall/precision tradeoff of what counts
                     as a "boundary" pixel; `4` matches the reference's own
                     default (`edge_size=4`).

    * Confirmed from the reference (`utils/criterion.py::weighted_bce`):
      per-pixel weight = (count of the *other* class) / (total pixel
      count) - boundary pixels (typically the rare minority) get weighted
      by the non-boundary count (close to 1), non-boundary pixels get
      weighted by the boundary count (small) - standard inverse-class-
      frequency balancing, not a fixed constant.
    * The reference's own `BondaryLoss` bakes an additional `coeff_bce=20.0`
      scale into its return value before summing with the other 3 loss
      terms; left out of this function (which returns the plain weighted
      BCE) so the ~20x weighting can be applied at the call site, matching
      this file's `deep_supervision_loss`-style "raw loss, caller weights"
      convention used everywhere else.
    '''
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred)

    def _batch_fn(y_true_np):
        return _canny_boundary_map_batch_np(y_true_np, dilate_size)

    boundary_map = tf.numpy_function(_batch_fn, [y_true], tf.float32)
    boundary_map = tf.stop_gradient(boundary_map)
    boundary_map.set_shape(y_pred.shape)

    pos_mask = boundary_map
    neg_mask = 1.0 - boundary_map
    pos_num = tf.reduce_sum(pos_mask)
    neg_num = tf.reduce_sum(neg_mask)
    total = pos_num + neg_num + K.epsilon()
    weight = pos_mask * (neg_num / total) + neg_mask * (pos_num / total)
    weight = tf.stop_gradient(weight)

    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=boundary_map, logits=y_pred)
    return tf.reduce_mean(weight * bce)


def pidnet_boundary_attentive_loss(y_true, main_pred, boundary_pred, threshold=0.8):
    '''
    PIDNet's "boundary-attentive" semantic loss (term 4 of its 4-term total
    loss, plan.md §3.4) - extra cross-entropy supervision on the *main*
    output, restricted to pixels the D branch is confident are boundary
    pixels; everywhere else contributes nothing.

    pidnet_boundary_attentive_loss(y_true, main_pred, boundary_pred, threshold=0.8)

    ----------
    Xu, J., Xiong, Z. and Bhattacharyya, S.P., 2023. PIDNet: A real-time
    semantic segmentation network inspired by PID controllers. In CVPR.

    Input
    ----------
        y_true: one-hot segmentation labels, shape (batch, H, W, n_classes).
        main_pred: the main/final head's raw logit, shape
                   (batch, H, W, n_classes) - `pidnet()`'s main output
                   (`augment=True`'s 2nd output, or the only output when
                   `augment=False`).
        boundary_pred: the D branch's raw single-channel logit, shape
                       (batch, H, W, 1) - the same tensor
                       `pidnet_boundary_loss`'s `y_pred` is.
        threshold: confident-boundary cutoff on `sigmoid(boundary_pred)` -
                   `0.8` matches the reference exactly
                   (`utils/utils.py::FullModel.forward`:
                   `bd_label = where(sigmoid(outputs[-1]) > 0.8, labels,
                   ignore_label)`, `loss_sb = sem_loss(outputs[-2],
                   bd_label)`).

    * The reference masks out non-boundary pixels via a sparse-label
      `ignore_label`, consumed by a `CrossEntropyLoss(ignore_index=...)`;
      this file's losses work with one-hot `y_true` instead (no ignore-
      class convention), so masking here is a plain "zero out this
      pixel's loss contribution, average only over the kept ones" instead
      - the same effect (unmasked pixels contribute nothing to the mean),
      not a literal ignore-index reproduction.
    '''
    y_true = tf.cast(y_true, tf.float32)
    main_pred = tf.convert_to_tensor(main_pred)
    boundary_pred = tf.convert_to_tensor(boundary_pred)

    mask = tf.cast(tf.sigmoid(boundary_pred) > threshold, tf.float32)
    mask = tf.stop_gradient(mask)

    log_probs = tf.nn.log_softmax(main_pred, axis=-1)
    ce = -tf.reduce_sum(y_true * log_probs, axis=-1, keepdims=True)

    denom = tf.reduce_sum(mask) + K.epsilon()
    return tf.reduce_sum(mask * ce) / denom

# ========================= #
# Structure loss (PraNet)


def structure_loss(y_true, y_pred, pool_size=31, const=K.epsilon()):
    '''
    Structure loss (weighted BCE + weighted IoU), as used in PraNet — cheap,
    strong baseline for camouflage/translucent-object segmentation. Gives
    extra weight to pixels near object boundaries (where a plain
    pixel-uniform BCE/IoU under-penalizes errors) via a local-contrast
    weight map derived directly from `y_true`.

    structure_loss(y_true, y_pred, pool_size=31)

    ----------
    Fan, D.P., Ji, G.P., Zhou, T., Chen, G., Fu, H., Shen, J. and Shao, L.,
    2020. PraNet: Parallel reverse attention network for polyp segmentation.
    In MICCAI.

    Input
    ----------
        y_true: segmentation targets, shape (batch, H, W, n_classes)
                (one-hot, or a single-channel binary mask).
        y_pred: sigmoid/softmax probabilities, same shape as `y_true`.
        pool_size: the local-average-pool window used to derive the
                   boundary-emphasis weight map (31 in the original paper).
        const: numerical-stability constant, as `dice_coef`.

    '''
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    local_avg = tf.nn.avg_pool2d(y_true, ksize=pool_size, strides=1, padding='SAME')
    weit = 1.0 + 5.0 * tf.abs(local_avg - y_true)

    y_pred_clipped = tf.clip_by_value(y_pred, const, 1.0 - const)
    bce = -(y_true * tf.math.log(y_pred_clipped) + (1.0 - y_true) * tf.math.log(1.0 - y_pred_clipped))
    wbce = tf.reduce_sum(weit * bce, axis=[1, 2]) / tf.reduce_sum(weit, axis=[1, 2])

    inter = tf.reduce_sum(y_pred * y_true * weit, axis=[1, 2])
    union = tf.reduce_sum((y_pred + y_true) * weit, axis=[1, 2])
    wiou = 1.0 - (inter + const) / (union - inter + const)

    return tf.reduce_mean(wbce + wiou)

# ========================= #
# Log-Cosh Dice loss


def log_cosh_dice(y_true, y_pred, const=K.epsilon()):
    '''
    Log-Cosh Dice loss — `log(cosh(dice_loss))`. The `cosh`/`log`
    composition flattens the loss landscape around zero, giving smoother
    gradients than plain Dice as the loss approaches convergence and
    reducing training instability on small-class-fraction masks.

    log_cosh_dice(y_true, y_pred, const=K.epsilon())

    ----------
    Jadon, S., 2020. A survey of loss functions for semantic segmentation.
    In IEEE CIBCB (introduces this as a smoothed wrapper around Dice).

    Input
    ----------
        y_true, y_pred, const: as `dice`.

    '''
    return tf.math.log(tf.math.cosh(dice(y_true, y_pred, const=const)))

# ========================= #
# OHEM cross-entropy


def ohem_categorical_crossentropy(keep_ratio=0.3, min_kept=1024):
    '''
    Online Hard Example Mining (OHEM) cross-entropy — computes per-pixel
    categorical cross-entropy, then averages only over the hardest
    `keep_ratio` fraction of pixels (by loss magnitude) in the batch,
    rather than every pixel uniformly. Mines hard shadow/reflection/
    texture-similar false positives that a uniform average would otherwise
    dilute into the easy-majority-pixel signal.

    ohem_categorical_crossentropy(keep_ratio=0.3, min_kept=1024)

    ----------
    Shrivastava, A., Gupta, A. and Girshick, R., 2016. Training
    region-based object detectors with online hard example mining. In CVPR
    (OHEM's original formulation; adapted here per-pixel for segmentation,
    the common adaptation used e.g. by PSPNet/ICNet's own training recipes).

    Input
    ----------
        keep_ratio: fraction of pixels (by hardest loss) kept per batch.
        min_kept: a floor on how many pixels are kept regardless of
                  `keep_ratio`, so an unusually easy batch doesn't collapse
                  to too few pixels for a stable gradient.

    Usage
    ----------
        model.compile(loss=ohem_categorical_crossentropy(keep_ratio=0.3), ...)

    '''
    keep_ratio_v = keep_ratio
    min_kept_v = min_kept

    def ohem_categorical_crossentropy_fixed(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        y_pred_clipped = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        per_pixel_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred_clipped), axis=-1)
        flat_loss = tf.reshape(per_pixel_loss, [-1])

        n_pixels = tf.shape(flat_loss)[0]
        n_keep = tf.maximum(
            tf.cast(tf.cast(n_pixels, tf.float32) * keep_ratio_v, tf.int32),
            tf.minimum(min_kept_v, n_pixels),
        )
        hardest, _ = tf.math.top_k(flat_loss, k=n_keep)
        return tf.reduce_mean(hardest)

    return ohem_categorical_crossentropy_fixed

# ========================= #
# Class-balanced weighting helper


def class_balanced_weights(samples_per_class, beta=0.999):
    '''
    Effective-number-of-samples class weighting (Cui et al. 2019) — a small
    numpy utility (not a loss itself) that turns raw per-class pixel/sample
    counts into weights suitable for `categorical_focal_loss`'s/
    `modified_categorical_focal_loss`'s `alpha` argument, or any other
    per-class loss weighting in this module.

    class_balanced_weights(samples_per_class, beta=0.999)

    ----------
    Cui, Y., Jia, M., Lin, T.Y., Song, Y. and Belongie, S., 2019.
    Class-balanced loss based on effective number of samples. In CVPR.

    Input
    ----------
        samples_per_class: 1-d array-like of per-class sample (or pixel)
                            counts.
        beta: the re-weighting hyperparameter controlling how strongly the
              "effective number" diverges from the raw count as counts grow
              — beta -> 1 approaches raw inverse-frequency weighting, beta=0
              gives uniform weights. 0.999 (the paper's own large-scale
              default) is a reasonable starting point; smaller datasets
              often want a smaller beta (e.g. 0.9-0.99).

    Output
    ----------
        A 1-d numpy array of per-class weights, normalized to sum to
        `n_classes` (an O(1) scale, matching the convention
        `categorical_focal_loss`'s own `alpha=[[.25, .25, .25]]`-style
        examples in this file use, rather than summing to 1).

    '''
    samples_per_class = np.asarray(samples_per_class, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, samples_per_class)
    weights = (1.0 - beta) / np.maximum(effective_num, K.epsilon())
    weights = weights / np.sum(weights) * len(samples_per_class)
    return weights.astype(np.float32)

# ========================= #
# Semi-hard triplet


def triplet_1d(y_true, y_pred, N, margin=5.0):
    '''
    (Experimental)
    Semi-hard triplet loss with one-dimensional vectors of anchor, positive, and negative.

    triplet_1d(y_true, y_pred, N, margin=5.0)

    Input
    ----------
        y_true: a dummy input, not used within this function. Appeared as a requirment of tf.keras.loss function format.
        y_pred: a single pass of triplet training, with `shape=(batch_num, 3*embeded_vector_size)`.
                i.e., `y_pred` is the ordered and concatenated anchor, positive, and negative embeddings.
        N: Size (dimensions) of embedded vectors
        margin: a positive number that prevents negative loss.

    '''

    # anchor sample pair separations.
    Embd_anchor = y_pred[:, 0:N]
    Embd_pos = y_pred[:, N:2*N]
    Embd_neg = y_pred[:, 2*N:]

    # squared distance measures
    d_pos = tf.reduce_sum(tf.square(Embd_anchor - Embd_pos), 1)
    d_neg = tf.reduce_sum(tf.square(Embd_anchor - Embd_neg), 1)
    loss_val = tf.maximum(0., margin + d_pos - d_neg)
    loss_val = tf.reduce_mean(loss_val)

    return loss_val


def focal_categorical_crossentropy(gamma, n_classes, label_smoothing=0.0):

    '''Focal categorical cross-entropy with optional label smoothing, for
    probability (not logit) predictions.

    Input
    ----------
        gamma: focusing parameter (0 reduces to plain cross-entropy).
        n_classes: number of classes (used to spread the smoothing mass).
        label_smoothing: smoothing factor in [0, 1).

    Output
    ----------
        a `loss(y_true, y_pred)` function.
    '''
    gamma_v = gamma
    n_classes_v = n_classes
    label_smoothing_v = label_smoothing

    def focal_categorical_crossentropy_(target, pred):
        return - ((1.0-label_smoothing_v)*(tf.reduce_mean(target*tf.pow((1.0-pred), gamma_v)*K.log(pred + K.epsilon()))) + (label_smoothing_v/(n_classes_v-1))*(tf.reduce_mean((1.0-target)*tf.pow((pred), gamma_v)*K.log(((1.0 - pred) + K.epsilon())))))

    return focal_categorical_crossentropy_


def modified_categorical_focal_loss(alpha=None, beta=0.5, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(
         alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha_v = np.array(alpha, dtype=np.float32) if alpha != None else 1.0
    beta_v = beta
    gamma_v = gamma

    def modified_categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy1 = -y_true * K.log(y_pred)
        cross_entropy2 = -(1-y_true) * K.log(1-y_pred)

        # Calculate Focal Loss
        loss1 = beta*alpha_v * K.pow(1 - y_pred, gamma_v) * cross_entropy1
        loss2 = (1.0 - beta)*alpha_v * K.pow(y_pred, gamma_v) * cross_entropy2

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss1+loss2, axis=-1))

    return modified_categorical_focal_loss_fixed


def combo_loss(alpha=0.5, beta=0.5, const=K.epsilon()):
    """Combo Loss (weighted BCE + Dice) — completes the stub this function
    used to be (plan.md §13 Phase 4 tracked "finish or deprecate"; finished
    against the actual paper formula rather than deprecated, since Structure
    loss above targets a different use case — binary/near-binary masks with
    a boundary-emphasis weighting — rather than superseding this one, which
    targets general multi-organ-style class imbalance).

    combo_loss(alpha=0.5, beta=0.5, const=K.epsilon())

    ----------
    Taghanaki, S.A., Abhishek, K., Cohen, J.P., Cohen-Adad, J. and Hamarneh,
    G., 2019. Combo loss: Handling input and output imbalance in
    multi-organ segmentation. Computer Medical Imaging and Graphics.

    Input
    ----------
        alpha: weight balancing the modified cross-entropy term against the
               Dice term — `L = alpha*mCE - (1-alpha)*Dice_coef`, the
               paper's own formulation. Note this loss is *not* bounded
               below by 0 the way Dice/Tversky alone are, since it's a
               difference of two terms, not a sum — this matches the paper,
               not a bug.
        beta: weight inside the modified cross-entropy term controlling the
              false-negative/false-positive trade-off — `beta` toward 1
              penalizes false negatives more, the identical convention
              `modified_categorical_focal_loss` above uses for its own
              `beta` argument.
        const: numerical-stability constant, as `dice_coef`.

    * The original stubbed signature also carried `gamma`/`label_smoothing`
      parameters — dropped here since the actual Combo Loss formula (unlike
      `symmetrical_unified_focal_loss` above, which legitimately composes a
      focal term) has no focal component; `gamma` never had a defined
      meaning for this loss, and keeping it in an unused, silently-ignored
      state once the function actually did something would have been
      actively misleading rather than just incomplete.

    """

    alpha_v = alpha
    beta_v = beta
    const_v = const

    def combo_loss_fixed(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        y_pred_clipped = tf.clip_by_value(y_pred, const_v, 1.0 - const_v)
        mce = -tf.reduce_mean(
            beta_v * y_true * tf.math.log(y_pred_clipped)
            + (1.0 - beta_v) * (1.0 - y_true) * tf.math.log(1.0 - y_pred_clipped)
        )

        d = dice_coef(tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1]), const=const_v)

        return alpha_v * mce - (1.0 - alpha_v) * d

    return combo_loss_fixed


def symmetrical_unified_focal_loss(gamma=0.5, delta=0.6, lamb=0.5, beta=1.0):
    """https://arxiv.org/pdf/2102.04525.pdf
    """

    modified_focal_loss_fixed = modified_categorical_focal_loss(
        beta=beta, gamma=1-gamma)

    def symmetrical_unified_focal_loss_fixed(y_true, y_pred):

        modified_focal_loss = delta*modified_focal_loss_fixed(y_true, y_pred)
        modified_focal_tversky_loss = focal_tversky(
            y_true, y_pred, alpha=delta, gamma=1.0/gamma)

        return lamb * modified_focal_loss + (1-lamb) * modified_focal_tversky_loss

    return symmetrical_unified_focal_loss_fixed


def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(
         alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha_v = np.array(alpha, dtype=np.float32)
    gamma_v = gamma
    classes_v = len(alpha_v)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha_v * K.pow(1 - y_pred, gamma_v) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed


def loss_adapter(loss_function, factor=1, expand=False, contract=False, from_logits=False, label_smoothing=0.0):

    '''Wrap a loss so it can be used with models whose output does not match
    the labels directly: raw logits, and/or an output resolution different
    from the label resolution.

    Input
    ----------
        loss_function: a `loss(y_true, y_pred)` function expecting
                       probabilities.
        factor: constant multiplier applied to the loss value.
        expand: resize `y_pred` (nearest) to the spatial size of `y_true`
                before computing the loss.
        contract: resize `y_true` (nearest) to the spatial size of `y_pred`
                  instead. Ignored if `expand` is True.
        from_logits: apply a softmax to `y_pred` first. Needed for the
                     segmentation decoders of this library, which output
                     logits.
        label_smoothing: smoothing factor applied to `y_true`.

    Output
    ----------
        a `loss(y_true, y_pred)` function.
    '''
    loss_f = loss_function
    expand_ = expand

    # scale = scale

    def loss(y_true, y_pred):

        if expand_:
            new_height = int(y_true.shape[1])
            new_width = int(y_true.shape[2])
            y_pred = tf.image.resize(
                y_pred, [new_height, new_width],
                antialias=False, method='nearest')
        elif contract:
            new_height = int(y_pred.shape[1])
            new_width = int(y_pred.shape[2])

            y_true = tf.image.resize(y_true, [
                new_height, new_width], antialias=False, method='nearest')

        if from_logits:
            y_pred = tf.keras.activations.softmax(y_pred)

        if label_smoothing > 0.0:
            shrink = (1.0-label_smoothing)
            y_true = label_smoothing/y_true.shape[-1] + y_true*shrink
            # y_true = tf.keras.activations.sigmoid(y_true)
        return factor*loss_f(y_true, y_pred)

    return loss

# ========================= #
# Deep-supervision loss aggregation


def deep_supervision_loss(loss_fn, weights=(1.0, 0.4)):
    '''
    Combine a shared loss across a multi-output model's main and
    auxiliary head(s) into one callable — supports training DDRNet's
    optional auxiliary deep-supervision head (`swiss_army_keras._model_ddrnet.ddrnet`,
    `augment=True`, see plan.md §3.3) and any future multi-output model the
    same way. `model.compile(loss=[...], loss_weights=[...])` already
    covers this exact need for Keras's own `model.fit` training loop; this
    helper's actual value is offering the equivalent behavior *outside*
    that path, for a custom training loop (`GradientTape` + `optimizer.
    apply_gradients`/`minimize`), the same style already established by
    this module's `SAM`/`AdaHessian` optimizers.

    deep_supervision_loss(loss_fn, weights=(1.0, 0.4))

    Input
    ----------
        loss_fn: a single loss callable `(y_true, y_pred) -> scalar` shared
                 across every output — the common deep-supervision setup,
                 where the auxiliary head is trained with the *same* loss as
                 the main head, just weighted down. Pass a list of one such
                 callable per output instead if different outputs need
                 different losses.
        weights: per-output weights, in output order. Default `(1.0, 0.4)`
                 matches full weight on the main/first output and DDRNet's
                 own paper default weight (0.4) for its auxiliary head.

    Output
    ----------
        combined_loss(y_trues, y_preds): a callable taking two same-length
        lists/tuples (one entry per model output — e.g. DDRNet's own
        `[main_output, aux_output]` output order) and returning the
        weighted sum of `weights[i] * loss_fn(y_true_i, y_pred_i)` (or
        `loss_fn[i](...)` if `loss_fn` was a list). A single (non-list)
        `y_true`/`y_pred` is treated as a length-1 list, for a plain
        single-output model.

    Usage
    ----------
        combined = deep_supervision_loss(dice, weights=(1.0, 0.4))
        model = ddrnet(input_tensor, n_labels, augment=True)
        with tf.GradientTape() as tape:
            main_pred, aux_pred = model(x, training=True)
            loss = combined([y_true, y_true], [main_pred, aux_pred])
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    '''
    weights_v = list(weights)

    def combined_loss(y_trues, y_preds):
        if not isinstance(y_trues, (list, tuple)):
            y_trues = [y_trues]
        if not isinstance(y_preds, (list, tuple)):
            y_preds = [y_preds]

        loss_fns = loss_fn if isinstance(loss_fn, (list, tuple)) else [loss_fn] * len(y_preds)

        total = 0.0
        for w, fn, y_true, y_pred in zip(weights_v, loss_fns, y_trues, y_preds):
            total = total + w * fn(y_true, y_pred)
        return total

    return combined_loss

# ========================= #
# Detection losses (plan.md §9.4)


def quality_focal_loss(beta=2.0):
    '''
    Quality Focal Loss (QFL, part of Generalized Focal Loss) — jointly
    represents classification confidence and localization (IoU) quality in
    a single continuous sigmoid target, instead of a hard 0/1 classification
    label plus a separate objectness/IoU-quality branch. Required by the
    NanoDet-Plus-style GFL detection head (plan.md §9.3); also usable
    standalone with any detector as a drop-in for plain focal loss.

    quality_focal_loss(beta=2.0)

    ----------
    Li, X., Wang, W., Wu, L., Chen, S., Hu, X., Li, J., Tang, J. and Yang,
    J., 2020. Generalized focal loss: Learning qualified and distributed
    bounding boxes for dense object detection. NeurIPS.

    Input
    ----------
        beta: focusing parameter — down-weights easy examples (where the
              predicted score is already close to the continuous quality
              target) the same way plain focal loss's `gamma` does, just
              keyed off `|target - prediction|` instead of a hard label.
              2.0, the paper's own default.

    Usage
    ----------
        y_true: per-class continuous quality target in `[0, 1]` — 0 for
                background/negative locations, the predicted box's IoU with
                its matched ground-truth box for positive ones (not a hard
                one-hot label).
        y_pred: per-class predicted score, sigmoid-activated (each class
                independent — the same convention `multilabel_classifier`
                above uses, since the *quality* target here isn't a hard
                0/1 label even though a location is only ever one true
                class in practice).
    '''
    beta_v = beta

    def quality_focal_loss_fixed(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        y_pred_clipped = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        scale_factor = tf.abs(y_true - y_pred_clipped) ** beta_v
        ce = -(y_true * tf.math.log(y_pred_clipped) + (1.0 - y_true) * tf.math.log(1.0 - y_pred_clipped))

        return tf.reduce_mean(scale_factor * ce)

    return quality_focal_loss_fixed


def distribution_focal_loss():
    '''
    Distribution Focal Loss (DFL, part of Generalized Focal Loss) — trains
    a box side's regression target as a small discrete probability
    distribution over adjacent integer bins, rather than a direct scalar
    regression, sharpening that distribution toward the two bins nearest
    the (generally non-integer) true target value. Pairs with
    `quality_focal_loss` above for the GFL detection head (plan.md §9.3).

    distribution_focal_loss()

    ----------
    Li, X., Wang, W., Wu, L., Chen, S., Hu, X., Li, J., Tang, J. and Yang,
    J., 2020. Generalized focal loss: Learning qualified and distributed
    bounding boxes for dense object detection. NeurIPS.

    Usage
    ----------
        y_true: the continuous regression target, in units of the
                discretization bin (e.g. a box side's offset), shape
                `(...,)` — matching `y_pred`'s shape *without* its trailing
                bins axis exactly (not `(..., 1)`; a rank ambiguous between
                "one extra trailing singleton axis" and "batch size 1" is
                deliberately not auto-detected/squeezed here — squeeze
                `y_true` yourself before calling this if needed).
        y_pred: the predicted per-bin logits (pre-softmax), shape
                `(..., n_bins)` — `n_bins - 1` is the largest representable
                offset; `y_true` is clipped into `[0, n_bins - 1)`.
    '''

    def distribution_focal_loss_fixed(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        n_bins = y_pred.shape[-1]
        y_true = tf.clip_by_value(y_true, 0.0, float(n_bins - 1) - K.epsilon())

        left = tf.floor(y_true)
        right = left + 1.0
        weight_left = right - y_true
        weight_right = y_true - left

        log_probs = tf.nn.log_softmax(y_pred, axis=-1)
        batch_dims = len(log_probs.shape) - 1
        log_prob_left = tf.gather(log_probs, tf.cast(left, tf.int32), batch_dims=batch_dims)
        log_prob_right = tf.gather(log_probs, tf.cast(right, tf.int32), batch_dims=batch_dims)

        loss = -(weight_left * log_prob_left + weight_right * log_prob_right)
        return tf.reduce_mean(loss)

    return distribution_focal_loss_fixed


def varifocal_loss(alpha=0.75, gamma=2.0):
    '''
    VariFocal Loss — an alternative to `quality_focal_loss` above for the
    same joint classification+IoU-quality target, using an *asymmetric*
    treatment of positives (weighted by the continuous IoU-quality target
    itself, with no separate focusing term) vs. negatives (down-weighted by
    `alpha` and focused by `gamma`, the same way plain focal loss's `gamma`
    works) — unlike QFL's single, symmetric `|target - prediction|^beta`
    scaling applied uniformly to both. Which of the two wins is
    dataset-dependent in practice, so this is kept as a second option
    rather than a QFL replacement.

    varifocal_loss(alpha=0.75, gamma=2.0)

    ----------
    Zhang, H., Wang, Y., Dayoub, F. and Sunderhauf, N., 2021. VarifocalNet:
    An IoU-aware dense object detector. In CVPR.

    Input
    ----------
        alpha: down-weighting factor for the negative (`y_true == 0`) term
               (0.75, the paper's own default).
        gamma: focusing parameter applied only to the negative term (2.0,
               the paper's own default).

    Usage
    ----------
        y_true, y_pred: same convention as `quality_focal_loss` — per-class
                        continuous IoU-quality target (0 for negatives) and
                        per-class sigmoid-activated predicted score.
    '''
    alpha_v = alpha
    gamma_v = gamma

    def varifocal_loss_fixed(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        y_pred_clipped = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        is_positive = tf.cast(y_true > 0.0, y_pred.dtype)

        weight = is_positive * y_true + (1.0 - is_positive) * alpha_v * tf.pow(y_pred_clipped, gamma_v)
        ce = -(y_true * tf.math.log(y_pred_clipped) + (1.0 - y_true) * tf.math.log(1.0 - y_pred_clipped))

        return tf.reduce_mean(weight * ce)

    return varifocal_loss_fixed
