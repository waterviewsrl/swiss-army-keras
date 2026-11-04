"""Tests for `_model_classifier.py`'s classifier heads, focused on this
session's plan.md §9.1/§9.2/§13 Phase 5 additions: the `pre_pool_attention`
hook wired into `classifier()`/`wise_srnet_classifier()`, and the two new
head variants `multilabel_classifier`/`ordinal_classifier`. No test file for
this module existed before this pass.
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras._model_classifier import (
    OrdinalThresholds,
    classifier,
    multilabel_classifier,
    ordinal_classifier,
    wise_srnet_classifier,
)

INPUT_SIZE = 96
N_CLASSES = 5


def _random_batch(batch_size=2):
    return np.random.rand(batch_size, INPUT_SIZE, INPUT_SIZE, 3).astype("float32")


@pytest.mark.parametrize("pre_pool_attention", [None, "gem", "eca", "coordatt"])
def test_classifier_pre_pool_attention_builds_and_predicts(pre_pool_attention):
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = classifier(
        inp, n_classes=N_CLASSES, backbone="MobileNetV2", weights=None,
        freeze_backbone=False, pre_pool_attention=pre_pool_attention,
    )
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, N_CLASSES)
    assert np.all(np.isfinite(out))
    # softmax output: each row sums to 1
    np.testing.assert_allclose(out.sum(axis=-1), np.ones(2), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("pre_pool_attention", [None, "eca", "coordatt"])
def test_wise_srnet_classifier_pre_pool_attention_builds_and_predicts(pre_pool_attention):
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = wise_srnet_classifier(
        inp, n_classes=N_CLASSES, backbone="MobileNetV2", weights=None,
        freeze_backbone=False, pre_pool_attention=pre_pool_attention,
    )
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, N_CLASSES)
    assert np.all(np.isfinite(out))


def test_wise_srnet_classifier_gem_not_supported():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    with pytest.raises(ValueError, match="gem"):
        wise_srnet_classifier(
            inp, n_classes=N_CLASSES, backbone="MobileNetV2", weights=None,
            pre_pool_attention="gem",
        )


def test_multilabel_classifier_builds_and_predicts():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = multilabel_classifier(
        inp, n_classes=N_CLASSES, backbone="MobileNetV2", weights=None, freeze_backbone=False,
    )
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, N_CLASSES)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0) and np.all(out <= 1)
    # sigmoid, independent per class - unlike softmax, rows need not sum to 1.
    assert not np.allclose(out.sum(axis=-1), np.ones(2), atol=1e-3)


@pytest.mark.parametrize("pre_pool_attention", [None, "gem", "eca", "coordatt"])
def test_multilabel_classifier_pre_pool_attention_builds(pre_pool_attention):
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = multilabel_classifier(
        inp, n_classes=N_CLASSES, backbone="MobileNetV2", weights=None,
        freeze_backbone=False, pre_pool_attention=pre_pool_attention,
    )
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, N_CLASSES)
    assert np.all(np.isfinite(out))


def test_ordinal_classifier_builds_and_predicts():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = ordinal_classifier(
        inp, n_classes=N_CLASSES, backbone="MobileNetV2", weights=None, freeze_backbone=False,
    )
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, N_CLASSES - 1)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0) and np.all(out <= 1)


def test_ordinal_thresholds_shares_one_logit_across_thresholds():
    """The core CORAL property (plan.md §9.2): every threshold's pre-sigmoid
    logit is the *same* shared value, shifted only by its own learned bias -
    not `n_thresholds` independently-learned projections. Confirmed by
    checking the *differences* between thresholds equal the differences
    between their biases exactly, for an arbitrary (non-zero) shared logit.
    """
    layer = OrdinalThresholds(4)
    layer.build((None, 1))
    biases = np.array([0.5, -0.2, -1.0, -2.3], dtype="float32")
    layer.set_weights([biases])

    shared_logit = tf.constant([[2.0], [-1.5]])
    out = layer(shared_logit).numpy()

    for row in out:
        np.testing.assert_allclose(np.diff(row), np.diff(biases), rtol=1e-5, atol=1e-5)


def test_ordinal_thresholds_config_roundtrip():
    layer = OrdinalThresholds(6)
    restored = OrdinalThresholds.from_config(layer.get_config())
    assert restored.n_thresholds == 6


# `return_logits` (plan.md §13.2, Phase 10) - splits the fused
# `Dense(n_classes, activation='softmax')` into two separate layer calls
# so raw logits become available for `layer_utils.add_energy_score_output`
# (softmax alone can't recover them - it normalizes away the scale the
# Energy Score formula needs).


def test_classifier_return_logits_false_keeps_single_output_unchanged():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = classifier(
        inp, n_classes=N_CLASSES, backbone="MobileNetV2", weights=None,
        freeze_backbone=False, return_logits=False,
    )
    assert not isinstance(model.output, list)
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, N_CLASSES)
    np.testing.assert_allclose(out.sum(axis=-1), np.ones(2), rtol=1e-4, atol=1e-4)


def test_classifier_return_logits_true_adds_logits_output_matching_softmax():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = classifier(
        inp, n_classes=N_CLASSES, backbone="MobileNetV2", weights=None,
        freeze_backbone=False, return_logits=True,
    )
    assert isinstance(model.output, list)
    assert len(model.outputs) == 2

    probs, logits = model.predict(_random_batch(), verbose=0)
    assert probs.shape == (2, N_CLASSES)
    assert logits.shape == (2, N_CLASSES)
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones(2), rtol=1e-4, atol=1e-4)
    # Defining property: the two outputs are the same Dense's pre/post-
    # softmax values, not independent computations.
    np.testing.assert_allclose(probs, tf.nn.softmax(logits, axis=-1).numpy(), atol=1e-5)


# `pose_regression_head` / `reid_embedding_head` (plan.md §13.3/§13.4,
# Phase 10) - lightweight single-person keypoint regression and ReID
# embedding heads, both reusing `classifier()`'s own trunk-building
# pattern almost verbatim. Numeric-property assertions (exact value
# ranges, L2-norm) use `MobileOne-S0` rather than this file's usual
# `MobileNetV2` - a real, narrowed-down pre-existing finding (plan.md
# §10) is that `backbone='MobileNetV2'` combined with `weights=None`
# can make a QAT-annotated final unactivated layer collapse to exactly
# zero output; `MobileOne-S0` (this repo's own backbone) doesn't exhibit
# it, confirmed empirically. `MobileNetV2` is still used below for broad
# build/predict/shape coverage, matching this file's existing convention,
# just not for assertions that need genuinely non-degenerate values.

from swiss_army_keras._model_classifier import pose_regression_head, reid_embedding_head
from swiss_army_keras.losses import triplet_1d

POSE_INPUT_SIZE = 96
N_KEYPOINTS = 7
EMBEDDING_SIZE = 32


def test_pose_regression_head_builds_and_predicts_with_mobilenetv2():
    inp = tf.keras.Input((POSE_INPUT_SIZE, POSE_INPUT_SIZE, 3))
    model = pose_regression_head(
        inp, n_keypoints=N_KEYPOINTS, backbone="MobileNetV2", weights=None, freeze_backbone=False,
    )
    out = model.predict(np.random.rand(2, POSE_INPUT_SIZE, POSE_INPUT_SIZE, 3).astype("float32"), verbose=0)
    assert out.shape == (2, N_KEYPOINTS, 3)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_pose_regression_head_outputs_vary_across_keypoints_with_mobileone():
    """Stronger than a bounds check: with a backbone unaffected by the
    §10 all-zero finding, the (x, y, visibility) values must actually
    vary across keypoints, not collapse to a single constant (e.g.
    `sigmoid(0) == 0.5` everywhere, which would still trivially pass a
    `[0, 1]` bounds check).
    """
    inp = tf.keras.Input((POSE_INPUT_SIZE, POSE_INPUT_SIZE, 3))
    model = pose_regression_head(
        inp, n_keypoints=N_KEYPOINTS, backbone="MobileOne-S0", weights=None, freeze_backbone=False,
    )
    out = model.predict(np.random.rand(2, POSE_INPUT_SIZE, POSE_INPUT_SIZE, 3).astype("float32"), verbose=0)
    assert out.shape == (2, N_KEYPOINTS, 3)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    assert not np.allclose(out, 0.5), "output degenerated to a constant - see plan.md §10's finding"
    assert out.std() > 1e-3


def test_reid_embedding_head_builds_and_predicts_with_mobilenetv2():
    inp = tf.keras.Input((POSE_INPUT_SIZE, POSE_INPUT_SIZE, 3))
    model = reid_embedding_head(
        inp, embedding_size=EMBEDDING_SIZE, backbone="MobileNetV2", weights=None, freeze_backbone=False,
    )
    out = model.predict(np.random.rand(2, POSE_INPUT_SIZE, POSE_INPUT_SIZE, 3).astype("float32"), verbose=0)
    assert out.shape == (2, EMBEDDING_SIZE)
    assert np.all(np.isfinite(out))


def test_reid_embedding_head_output_is_l2_normalized_with_mobileone():
    inp = tf.keras.Input((POSE_INPUT_SIZE, POSE_INPUT_SIZE, 3))
    model = reid_embedding_head(
        inp, embedding_size=EMBEDDING_SIZE, backbone="MobileOne-S0", weights=None, freeze_backbone=False,
    )
    out = model.predict(np.random.rand(2, POSE_INPUT_SIZE, POSE_INPUT_SIZE, 3).astype("float32"), verbose=0)
    norms = np.linalg.norm(out, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)


def test_reid_embedding_head_defaults_to_gem_pooling():
    """`pre_pool_attention` defaults to `'gem'` here, unlike every other
    classifier variant in this file (which default to `None`) - a
    deliberate, documented choice (Luo et al. 2019's own ReID-baseline
    convention), confirmed by introspecting the built layer graph the
    same way `test_apply_pre_pool_attention_dispatches_to_correct_layer`
    already does for the shared dispatch helper.
    """
    inp = tf.keras.Input((POSE_INPUT_SIZE, POSE_INPUT_SIZE, 3))
    model = reid_embedding_head(
        inp, embedding_size=EMBEDDING_SIZE, backbone="MobileNetV2", weights=None, freeze_backbone=False,
    )
    assert any("gem" in layer.name.lower() for layer in model.layers)


def test_reid_embedding_head_composes_with_triplet_1d_loss():
    """End-to-end training shape check: calling this same model 3x
    (anchor/positive/negative) and concatenating the embeddings must
    produce a valid `triplet_1d` input, per this head's own documented
    training convention.
    """
    inp = tf.keras.Input((POSE_INPUT_SIZE, POSE_INPUT_SIZE, 3))
    model = reid_embedding_head(
        inp, embedding_size=EMBEDDING_SIZE, backbone="MobileOne-S0", weights=None, freeze_backbone=False,
    )

    def _batch():
        return np.random.rand(3, POSE_INPUT_SIZE, POSE_INPUT_SIZE, 3).astype("float32")

    anchor = model.predict(_batch(), verbose=0)
    positive = model.predict(_batch(), verbose=0)
    negative = model.predict(_batch(), verbose=0)
    concatenated = np.concatenate([anchor, positive, negative], axis=-1)

    loss = triplet_1d(None, tf.constant(concatenated), N=EMBEDDING_SIZE)
    assert np.isfinite(loss.numpy())


# `iqa_head` (plan.md §13.7, Phase 11) - a small IQA side-head tapped off
# the same shared backbone, trained simultaneously with the main task.


def test_classifier_iqa_head_false_keeps_output_unchanged():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = classifier(
        inp, n_classes=N_CLASSES, backbone="MobileNetV2", weights=None,
        freeze_backbone=False, iqa_head=False,
    )
    assert not isinstance(model.output, list)
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, N_CLASSES)


def test_classifier_iqa_head_true_adds_bounded_second_output():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = classifier(
        inp, n_classes=N_CLASSES, backbone="MobileOne-S0", weights=None,
        freeze_backbone=False, iqa_head=True,
    )
    assert len(model.outputs) == 2

    probs, iqa = model.predict(_random_batch(), verbose=0)
    assert probs.shape == (2, N_CLASSES)
    assert iqa.shape == (2, 1)
    assert iqa.min() >= 0.0 and iqa.max() <= 1.0


def test_classifier_iqa_head_composes_with_return_logits():
    """`iqa_head` and `return_logits` compose - the IQA score is always
    the *last* output, per `classifier()`'s own documented order.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = classifier(
        inp, n_classes=N_CLASSES, backbone="MobileOne-S0", weights=None,
        freeze_backbone=False, return_logits=True, iqa_head=True,
    )
    assert len(model.outputs) == 3

    probs, logits, iqa = model.predict(_random_batch(), verbose=0)
    assert probs.shape == (2, N_CLASSES)
    assert logits.shape == (2, N_CLASSES)
    assert iqa.shape == (2, 1)
    assert iqa.min() >= 0.0 and iqa.max() <= 1.0
