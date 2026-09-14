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
