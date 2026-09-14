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
