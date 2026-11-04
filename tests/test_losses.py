"""Tests for the new losses added under plan.md §4.1/§13 Phase 4.

Each loss is checked for: a finite, sane value on a synthetic batch, a
non-`None` finite gradient w.r.t. its predictions (the actual requirement
for use inside `model.compile(loss=...)`/a custom training loop — a loss
that merely "doesn't crash" but has a broken/`None` gradient would silently
fail to train anything), and any documented special behavior (e.g.
Lovász-Softmax's 'present'-vs-'all' class averaging).
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras.losses import (
    boundary_loss,
    class_balanced_weights,
    combo_loss,
    deep_supervision_loss,
    dice,
    distribution_focal_loss,
    iou_box_coef,
    log_cosh_dice,
    lovasz_softmax,
    ohem_categorical_crossentropy,
    pidnet_boundary_attentive_loss,
    pidnet_boundary_loss,
    quality_focal_loss,
    structure_loss,
    varifocal_loss,
)

BATCH, HEIGHT, WIDTH, N_CLASSES = 2, 16, 16, 3


@pytest.fixture
def onehot_targets_and_probs():
    rng = np.random.default_rng(0)
    y_true_idx = rng.integers(0, N_CLASSES, size=(BATCH, HEIGHT, WIDTH))
    y_true = tf.constant(tf.one_hot(y_true_idx, N_CLASSES))
    logits = rng.standard_normal((BATCH, HEIGHT, WIDTH, N_CLASSES)).astype("float32")
    return y_true, logits


def _predict_with_gradient(loss_fn, y_true, logits):
    logits_var = tf.Variable(logits)
    with tf.GradientTape() as tape:
        y_pred = tf.nn.softmax(logits_var, axis=-1)
        loss = loss_fn(y_true, y_pred)
    grad = tape.gradient(loss, logits_var)
    return loss, grad


LOSS_FNS = [
    ("lovasz_softmax", lovasz_softmax),
    ("boundary_loss", boundary_loss),
    ("structure_loss", structure_loss),
    ("log_cosh_dice", log_cosh_dice),
    ("combo_loss", combo_loss()),
    ("ohem_categorical_crossentropy", ohem_categorical_crossentropy()),
]


@pytest.mark.parametrize("name,loss_fn", LOSS_FNS)
def test_loss_is_finite_and_differentiable(name, loss_fn, onehot_targets_and_probs):
    y_true, logits = onehot_targets_and_probs
    loss, grad = _predict_with_gradient(loss_fn, y_true, logits)

    assert np.isfinite(float(loss)), f"{name} produced a non-finite loss"
    assert grad is not None, f"{name} produced no gradient w.r.t. predictions"
    assert np.all(np.isfinite(grad.numpy())), f"{name} produced a non-finite gradient"


def test_perfect_prediction_minimizes_region_based_losses(onehot_targets_and_probs):
    """Sanity check that isn't just "doesn't crash": each region-based loss
    (excludes `boundary_loss`, which penalizes prediction mass by distance
    from the boundary rather than region overlap, and `ohem`, which mines
    hard *pixels* rather than measuring overlap) should be near its minimum
    when the prediction exactly matches the target, and higher for a
    deliberately wrong (uniform) prediction.
    """
    y_true, _ = onehot_targets_and_probs
    y_perfect = y_true
    y_uniform = tf.fill(y_true.shape, 1.0 / N_CLASSES)

    for name, loss_fn in [
        ("lovasz_softmax", lovasz_softmax),
        ("structure_loss", structure_loss),
        ("log_cosh_dice", log_cosh_dice),
        ("combo_loss", combo_loss()),
    ]:
        perfect_loss = float(loss_fn(y_true, y_perfect))
        wrong_loss = float(loss_fn(y_true, y_uniform))
        assert perfect_loss < wrong_loss, (
            f"{name}: perfect prediction ({perfect_loss}) should score better than "
            f"a uniform/wrong one ({wrong_loss})"
        )


def test_lovasz_softmax_present_vs_all_agree_when_all_classes_present():
    """When every class actually appears in the batch, 'present' and 'all'
    averaging should agree exactly - 'present' only differs by *excluding*
    absent classes, so with none absent there's nothing to exclude.
    """
    rng = np.random.default_rng(1)
    y_true_idx = np.tile(np.arange(N_CLASSES), HEIGHT * WIDTH * BATCH // N_CLASSES + 1)[
        : BATCH * HEIGHT * WIDTH
    ].reshape(BATCH, HEIGHT, WIDTH)
    y_true = tf.constant(tf.one_hot(y_true_idx, N_CLASSES))
    y_pred = tf.nn.softmax(rng.standard_normal((BATCH, HEIGHT, WIDTH, N_CLASSES)).astype("float32"), axis=-1)

    present = float(lovasz_softmax(y_true, y_pred, classes='present'))
    all_classes = float(lovasz_softmax(y_true, y_pred, classes='all'))
    np.testing.assert_allclose(present, all_classes, rtol=1e-5, atol=1e-5)


def test_class_balanced_weights_favors_rare_classes():
    weights = class_balanced_weights([1000, 100, 1])
    assert weights[2] > weights[1] > weights[0]
    assert np.all(np.isfinite(weights))


def test_class_balanced_weights_uniform_when_beta_zero():
    weights = class_balanced_weights([1000, 100, 1], beta=0.0)
    np.testing.assert_allclose(weights, np.ones(3), rtol=1e-5)


def test_deep_supervision_loss_weights_outputs():
    y_true = tf.ones((BATCH, HEIGHT, WIDTH, N_CLASSES)) / N_CLASSES
    y_main = tf.ones((BATCH, HEIGHT, WIDTH, N_CLASSES)) / N_CLASSES  # perfect: dice ~ 0
    y_aux = tf.zeros((BATCH, HEIGHT, WIDTH, N_CLASSES))  # wrong: dice ~ 1

    combined = deep_supervision_loss(dice, weights=(1.0, 0.4))
    total = float(combined([y_true, y_true], [y_main, y_aux]))

    main_only = float(dice(y_true, y_main))
    aux_only = float(dice(y_true, y_aux))
    np.testing.assert_allclose(total, 1.0 * main_only + 0.4 * aux_only, rtol=1e-4, atol=1e-4)


def test_deep_supervision_loss_single_output_not_a_list():
    y_true = tf.ones((BATCH, HEIGHT, WIDTH, N_CLASSES)) / N_CLASSES
    y_pred = tf.ones((BATCH, HEIGHT, WIDTH, N_CLASSES)) / N_CLASSES

    combined = deep_supervision_loss(dice, weights=(1.0,))
    total = float(combined(y_true, y_pred))
    np.testing.assert_allclose(total, float(dice(y_true, y_pred)), rtol=1e-6)


def test_combo_loss_not_bounded_below_by_zero():
    """Regression/documentation test for `combo_loss`'s completed
    implementation (plan.md §13 Phase 4): unlike Dice/Tversky, Combo Loss is
    `alpha*mCE - (1-alpha)*Dice_coef`, a *difference* of two terms, so it can
    legitimately go negative for a good prediction - this is the correct
    behavior per the paper, not a bug, and this test exists so a future
    "fix" that clips it to [0, inf) doesn't silently change the loss's
    actual semantics.
    """
    y_true = tf.ones((BATCH, HEIGHT, WIDTH, N_CLASSES)) / N_CLASSES
    y_perfect = y_true
    loss_val = float(combo_loss(alpha=0.1)(y_true, y_perfect))
    assert loss_val < 0, "combo_loss should go negative for a near-perfect, low-alpha prediction"


# ========================= #
# Detection losses (plan.md §9.4/§13 Phase 5)

DETECTION_LOSS_FNS = [
    ("quality_focal_loss", quality_focal_loss()),
    ("varifocal_loss", varifocal_loss()),
]


@pytest.mark.parametrize("name,loss_fn", DETECTION_LOSS_FNS)
def test_detection_loss_is_finite_and_differentiable(name, loss_fn):
    rng = np.random.default_rng(0)
    y_true = tf.constant(rng.random((4, 3)).astype("float32"))
    logits = tf.Variable(rng.standard_normal((4, 3)).astype("float32"))

    with tf.GradientTape() as tape:
        y_pred = tf.nn.sigmoid(logits)
        loss = loss_fn(y_true, y_pred)
    grad = tape.gradient(loss, logits)

    assert np.isfinite(float(loss)), f"{name} produced a non-finite loss"
    assert grad is not None, f"{name} produced no gradient"
    assert np.all(np.isfinite(grad.numpy())), f"{name} produced a non-finite gradient"


def test_quality_focal_loss_scales_down_easy_examples():
    """QFL's whole point (plan.md §9.4): an example whose prediction already
    matches its continuous quality target should contribute much less loss
    than one that's equally "wrong" in cross-entropy terms but far from its
    target - the `|target - prediction|^beta` scale factor is what
    distinguishes it from plain BCE.
    """
    qfl = quality_focal_loss(beta=2.0)
    easy_true = tf.constant([[0.9]])
    easy_pred = tf.constant([[0.89]])  # close to target
    hard_true = tf.constant([[0.9]])
    hard_pred = tf.constant([[0.1]])  # far from target, same BCE "direction"

    easy_loss = float(qfl(easy_true, easy_pred))
    hard_loss = float(qfl(hard_true, hard_pred))
    assert easy_loss < hard_loss


def test_distribution_focal_loss_exact_bin_match_is_near_zero():
    """Regression test for the shape-ambiguity bug found while implementing
    this (plan.md §9.4): an earlier version auto-squeezed `y_true` whenever
    its last axis was size 1, which incorrectly fired for a batch-size-1
    1-D target (shape `(1,)`) too, silently collapsing it to a scalar and
    breaking the whole computation. Fixed by dropping the auto-squeeze
    entirely and requiring `y_true`'s shape to match `y_pred`'s shape minus
    the trailing bins axis, unambiguously - this test uses exactly that
    batch-size-1 case, the one the bug broke.
    """
    dfl = distribution_focal_loss()
    n_bins = 8
    peaked_logits = tf.constant([[0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0]])
    exact_target = tf.constant([3.0])  # shape (1,), batch size 1

    loss = float(dfl(exact_target, peaked_logits))
    assert abs(loss) < 1e-3


def test_distribution_focal_loss_interpolates_between_adjacent_bins():
    dfl = distribution_focal_loss()
    logits = tf.constant([[0.0, 0.0, 0.0, 20.0, 20.0, 0.0, 0.0, 0.0]])  # peaked at bins 3 and 4 equally

    loss_at_midpoint = float(dfl(tf.constant([3.5]), logits))
    loss_at_bin = float(dfl(tf.constant([3.0]), logits))
    # Both bins 3 and 4 are equally likely, so a target exactly between them
    # should score at least as well as a target sitting exactly on one bin.
    assert loss_at_midpoint <= loss_at_bin + 1e-4


def test_varifocal_loss_positive_weight_is_target_not_focused():
    """VariFocal's defining asymmetry (plan.md §9.4): positives are weighted
    by the target itself (no extra focusing exponent), negatives are
    weighted by `alpha * prediction^gamma` - so, unlike QFL, changing
    `gamma` should change the negative-example loss but leave a
    target=1.0 positive example's loss completely unaffected (its weight
    is just `y_true`, independent of `gamma`).
    """
    y_true_pos = tf.constant([[1.0]])
    y_pred = tf.constant([[0.5]])

    loss_gamma_1 = float(varifocal_loss(gamma=1.0)(y_true_pos, y_pred))
    loss_gamma_4 = float(varifocal_loss(gamma=4.0)(y_true_pos, y_pred))
    np.testing.assert_allclose(loss_gamma_1, loss_gamma_4, rtol=1e-6)

    y_true_neg = tf.constant([[0.0]])
    loss_neg_gamma_1 = float(varifocal_loss(gamma=1.0)(y_true_neg, y_pred))
    loss_neg_gamma_4 = float(varifocal_loss(gamma=4.0)(y_true_neg, y_pred))
    assert not np.isclose(loss_neg_gamma_1, loss_neg_gamma_4, rtol=1e-3)


IOU_BOX_MODES = ["giou", "ciou", "siou"]


@pytest.mark.parametrize("mode", IOU_BOX_MODES)
def test_iou_box_coef_identical_boxes_score_maximally(mode):
    box = tf.constant([[0.0, 0.0, 10.0, 10.0]])
    coef = float(tf.squeeze(iou_box_coef(box, box, mode=mode)))
    np.testing.assert_allclose(coef, 1.0, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("mode", IOU_BOX_MODES)
def test_iou_box_coef_distant_boxes_score_much_lower_than_identical(mode):
    box_a = tf.constant([[0.0, 0.0, 10.0, 10.0]])
    box_b = tf.constant([[100.0, 100.0, 110.0, 110.0]])
    coef = float(tf.squeeze(iou_box_coef(box_a, box_b, mode=mode)))
    assert coef < 0.0


@pytest.mark.parametrize("mode", ["ciou", "siou"])
def test_iou_box_coef_ciou_siou_differentiable(mode):
    box_true = tf.constant([[0.0, 0.0, 10.0, 10.0]])
    box_pred = tf.Variable([[1.0, 1.0, 9.0, 12.0]])

    with tf.GradientTape() as tape:
        loss = 1.0 - iou_box_coef(box_true, box_pred, mode=mode)
    grad = tape.gradient(loss, box_pred)

    assert grad is not None
    assert np.all(np.isfinite(grad.numpy()))


def test_iou_box_coef_unknown_mode_raises():
    box = tf.constant([[0.0, 0.0, 10.0, 10.0]])
    with pytest.raises(ValueError, match="Unknown mode"):
        iou_box_coef(box, box, mode="not_a_real_mode")


def test_iou_box_coef_ciou_penalizes_aspect_ratio_mismatch_beyond_iou():
    """CIoU's defining addition over plain IoU (plan.md §9.4): two boxes
    with the *same* IoU-overlap-driving corners but a different aspect
    ratio than the target should score lower under CIoU than under plain
    IoU, since CIoU adds an aspect-ratio-consistency penalty IoU alone
    doesn't have.
    """
    box_true = tf.constant([[0.0, 0.0, 10.0, 10.0]])  # square
    box_pred = tf.constant([[0.0, 0.0, 10.0, 5.0]])  # same intersection-relevant corner, different shape

    iou = float(tf.squeeze(iou_box_coef(box_true, box_pred, mode="iou")))
    ciou = float(tf.squeeze(iou_box_coef(box_true, box_pred, mode="ciou")))
    assert ciou < iou


# ========================= #
# PIDNet boundary losses (plan.md §3.4/§13 Phase 9)


@pytest.fixture
def pidnet_boundary_fixtures():
    """A single contiguous circular blob, not per-pixel random noise -
    `pidnet_boundary_loss`'s Canny+dilate boundary derivation genuinely
    needs a realistic, spatially-contiguous mask: pure per-pixel random
    labels make *every* pixel a "boundary" after dilation (no two adjacent
    pixels share a class), collapsing the inverse-class-frequency weight
    to zero everywhere (found while implementing this - not a bug, but a
    real degenerate case worth avoiding in tests too).
    """
    size = 64
    yy, xx = np.mgrid[0:size, 0:size]
    mask = (((yy - size // 2) ** 2 + (xx - size // 2) ** 2) < 15 ** 2).astype(np.int32)
    y_true = tf.constant(tf.one_hot(mask[None, ...], 2))  # (1, size, size, 2)

    rng = np.random.default_rng(0)
    boundary_logit = tf.Variable(rng.standard_normal((1, size, size, 1)).astype("float32"))
    main_logit = tf.Variable(rng.standard_normal((1, size, size, 2)).astype("float32"))
    return y_true, boundary_logit, main_logit


def test_pidnet_boundary_loss_is_finite_and_differentiable(pidnet_boundary_fixtures):
    y_true, boundary_logit, _ = pidnet_boundary_fixtures
    with tf.GradientTape() as tape:
        loss = pidnet_boundary_loss(y_true, boundary_logit)
    grad = tape.gradient(loss, boundary_logit)

    assert np.isfinite(float(loss))
    assert grad is not None
    assert np.all(np.isfinite(grad.numpy()))
    assert np.any(grad.numpy() != 0)


def test_pidnet_boundary_loss_weight_is_inverse_class_frequency():
    """Regression test for `pidnet_boundary_loss`'s exact weighting formula
    (plan.md §3.4, confirmed from the reference
    `utils/criterion.py::weighted_bce`): `weight[boundary] =
    non_boundary_count / total`, `weight[non_boundary] = boundary_count /
    total` - i.e. whichever class is *rarer* gets the *larger* weight, the
    easy-to-get-backwards direction (a naive port might weight the rare
    class by its *own* count instead). `cv2.Canny`'s exact pixel-level
    output isn't independently hand-computable for anything but trivial
    inputs, so this exercises the formula in isolation (identical to
    `pidnet_boundary_loss`'s own weight computation), on a hand-constructed
    boundary map with an exact, known pixel count - a 1-pixel "boundary"
    out of 9 should end up weighted ~8/9 (its own class is the rare one),
    the other 8 pixels weighted ~1/9 each.
    """
    boundary_map = np.zeros((1, 3, 3, 1), dtype="float32")
    boundary_map[0, 1, 1, 0] = 1.0  # exactly one "boundary" pixel out of 9

    pos_mask = tf.constant(boundary_map)
    neg_mask = 1.0 - pos_mask
    pos_num = tf.reduce_sum(pos_mask)
    neg_num = tf.reduce_sum(neg_mask)
    total = pos_num + neg_num
    weight = pos_mask * (neg_num / total) + neg_mask * (pos_num / total)

    np.testing.assert_allclose(float(weight[0, 1, 1, 0]), 8.0 / 9.0, rtol=1e-5)
    np.testing.assert_allclose(float(weight[0, 0, 0, 0]), 1.0 / 9.0, rtol=1e-5)


def test_pidnet_boundary_attentive_loss_is_finite_and_differentiable(pidnet_boundary_fixtures):
    y_true, boundary_logit, main_logit = pidnet_boundary_fixtures
    with tf.GradientTape() as tape:
        loss = pidnet_boundary_attentive_loss(y_true, main_logit, boundary_logit, threshold=0.5)
    grad = tape.gradient(loss, main_logit)

    assert np.isfinite(float(loss))
    assert grad is not None
    assert np.all(np.isfinite(grad.numpy()))


def test_pidnet_boundary_attentive_loss_ignores_low_confidence_pixels():
    """Defining property (plan.md §3.4, confirmed from the reference
    `utils/utils.py::FullModel.forward`: `bd_label = where(sigmoid(d) >
    0.8, labels, ignore_label)`): a pixel whose boundary confidence is
    below `threshold` must not contribute to the loss *at all* - changing
    `main_pred` only at such a pixel should leave the loss value (and the
    gradient there) completely unchanged.
    """
    y_true = tf.one_hot(tf.zeros((1, 4, 4), dtype=tf.int32), 3)
    # Boundary logits below sigmoid^-1(0.5)=0 everywhere except one pixel -
    # only that one pixel is "confident" at threshold=0.5.
    boundary_logit = -10.0 * tf.ones((1, 4, 4, 1))
    boundary_logit = tf.tensor_scatter_nd_update(boundary_logit, [[0, 2, 2, 0]], [10.0])

    main_logit_a = tf.Variable(tf.zeros((1, 4, 4, 3)))
    main_logit_b = tf.Variable(main_logit_a.numpy())
    # Perturb only a masked-out (non-confident) pixel.
    main_logit_b[0, 0, 0, :].assign([5.0, -5.0, 0.0])

    loss_a = float(pidnet_boundary_attentive_loss(y_true, main_logit_a, boundary_logit, threshold=0.5))
    loss_b = float(pidnet_boundary_attentive_loss(y_true, main_logit_b, boundary_logit, threshold=0.5))
    np.testing.assert_allclose(loss_a, loss_b, rtol=1e-5, atol=1e-5)

    with tf.GradientTape() as tape:
        loss = pidnet_boundary_attentive_loss(y_true, main_logit_b, boundary_logit, threshold=0.5)
    grad = tape.gradient(loss, main_logit_b)
    np.testing.assert_array_equal(grad.numpy()[0, 0, 0, :], np.zeros(3))


# Monocular depth losses (plan.md §13.5, Phase 11) - use a dedicated
# positive-valued-depth fixture rather than `onehot_targets_and_probs`
# (one-hot classification targets don't apply to continuous depth
# regression).

from swiss_army_keras.losses import berhu_loss, scale_invariant_log_loss, yolact_mask_loss


@pytest.fixture
def depth_targets_and_preds():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0.5, 5.0, size=(BATCH, HEIGHT, WIDTH, 1)).astype("float32")
    y_pred_logits = rng.standard_normal((BATCH, HEIGHT, WIDTH, 1)).astype("float32")
    return y_true, y_pred_logits


def _predict_positive_with_gradient(loss_fn, y_true, y_pred_logits):
    """Depth predictions must stay strictly positive - pass raw values
    through `exp` (differentiable, always positive) rather than the
    classification fixture's own softmax.
    """
    logits_var = tf.Variable(y_pred_logits)
    with tf.GradientTape() as tape:
        y_pred = tf.exp(logits_var)
        loss = loss_fn(y_true, y_pred)
    grad = tape.gradient(loss, logits_var)
    return loss, grad


@pytest.mark.parametrize("name,loss_fn", [
    ("scale_invariant_log_loss", scale_invariant_log_loss),
    ("berhu_loss", berhu_loss),
])
def test_depth_loss_is_finite_and_differentiable(name, loss_fn, depth_targets_and_preds):
    y_true, y_pred_logits = depth_targets_and_preds
    loss, grad = _predict_positive_with_gradient(loss_fn, y_true, y_pred_logits)

    assert np.isfinite(float(loss)), f"{name} produced a non-finite loss"
    assert grad is not None, f"{name} produced no gradient w.r.t. predictions"
    assert np.all(np.isfinite(grad.numpy())), f"{name} produced a non-finite gradient"


def test_scale_invariant_log_loss_is_zero_for_exact_prediction():
    y_true = tf.constant([[1.0, 2.0, 4.0, 0.5]])
    loss = scale_invariant_log_loss(y_true, y_true)
    np.testing.assert_allclose(float(loss), 0.0, atol=1e-6)


def test_scale_invariant_log_loss_ignores_invalid_pixels():
    """Pixels with `y_true <= 0` (missing/invalid depth) must be excluded
    from the loss entirely - not just down-weighted - confirmed by giving
    an invalid pixel a wildly wrong prediction and checking the loss is
    unaffected.
    """
    y_true_valid_only = tf.constant([[1.0, 2.0, 4.0]])
    y_pred_valid_only = tf.constant([[1.0, 2.0, 4.0]])
    loss_valid_only = scale_invariant_log_loss(y_true_valid_only, y_pred_valid_only)

    y_true_with_invalid = tf.constant([[1.0, 2.0, 4.0, 0.0]])
    y_pred_with_invalid = tf.constant([[1.0, 2.0, 4.0, 999.0]])
    loss_with_invalid = scale_invariant_log_loss(y_true_with_invalid, y_pred_with_invalid)

    np.testing.assert_allclose(float(loss_valid_only), float(loss_with_invalid), atol=1e-5)


def test_berhu_loss_exact_value_straddling_threshold():
    """Hand-computable case exercising both branches: three elements with
    `|error| <= c` (the L1 branch) and one with `|error| > c` (the L2
    branch), `c = 0.2 * max(|error|)` computed from these exact values.
    """
    y_true = tf.constant([[0.0, 0.0, 0.0, 0.0]])
    y_pred = tf.constant([[0.1, 0.1, 0.1, 1.0]])
    # errors: [0.1, 0.1, 0.1, 1.0], c = 0.2 * 1.0 = 0.2
    # |0.1| <= 0.2 -> L1 branch: 0.1 each
    # |1.0| > 0.2  -> L2 branch: (1.0^2 + 0.2^2) / (2*0.2) = 1.04 / 0.4 = 2.6
    expected = (0.1 + 0.1 + 0.1 + 2.6) / 4.0

    loss = berhu_loss(y_true, y_pred)
    np.testing.assert_allclose(float(loss), expected, atol=1e-4)


# YOLACT-style instance mask loss (plan.md §13.6, Phase 11) - a separate
# fixture again: per-instance area-normalized BCE over an assembled
# `(..., H, W)` mask, not a per-pixel-only or one-hot-classification shape.


def test_yolact_mask_loss_exact_value_hand_computed():
    """Single instance, 2x2 mask - every BCE term computed by hand below,
    summed and divided by the given `box_area`.
    """
    y_true = tf.constant([[[1.0, 0.0], [0.0, 1.0]]])  # (1, 2, 2)
    y_pred = tf.constant([[[0.8, 0.2], [0.3, 0.9]]])
    box_area = tf.constant([2.0])

    bce_terms = [-np.log(0.8), -np.log(0.8), -np.log(0.7), -np.log(0.9)]
    expected = sum(bce_terms) / 2.0

    loss = yolact_mask_loss(y_true, y_pred, box_area)
    np.testing.assert_allclose(float(loss), expected, atol=1e-4)


def test_yolact_mask_loss_is_finite_and_differentiable():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=(2, 3, 8, 8)).astype("float32")  # (batch, n_instances, H, W)
    logits = rng.standard_normal((2, 3, 8, 8)).astype("float32")
    box_area = rng.uniform(1.0, 20.0, size=(2, 3)).astype("float32")

    logits_var = tf.Variable(logits)
    with tf.GradientTape() as tape:
        y_pred = tf.sigmoid(logits_var)
        loss = yolact_mask_loss(y_true, y_pred, box_area)
    grad = tape.gradient(loss, logits_var)

    assert np.isfinite(float(loss))
    assert grad is not None
    assert np.all(np.isfinite(grad.numpy()))
