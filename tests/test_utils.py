"""Tests for `swiss_army_keras/utils.py` (plan.md §10/§13 Phase 6)."""

import pytest
import tensorflow as tf

from swiss_army_keras.utils import dummy_loader, freeze_model, unfreeze_model


def test_dummy_loader_no_longer_raises_nameerror(tmp_path):
    """Regression test for the pre-existing bug documented in plan.md §10:
    `dummy_loader` called `keras.models.load_model(...)` with no `keras`
    import anywhere in the file, raising `NameError` unconditionally -
    fixed by adding `from tensorflow import keras`. A nonexistent path
    should now fail with a file-not-found-style error from Keras itself,
    not a `NameError` from this module's own missing import.
    """
    with pytest.raises(Exception) as exc_info:
        dummy_loader(str(tmp_path / "does_not_exist.h5"))
    assert not isinstance(exc_info.value, NameError)


def test_dummy_loader_loads_real_model_weights(tmp_path):
    model = tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=(3,))])
    model_path = tmp_path / "model.h5"
    model.save(str(model_path))

    weights = dummy_loader(str(model_path))
    assert len(weights) == len(model.get_weights())
    for got, expected in zip(weights, model.get_weights()):
        assert got.shape == expected.shape


def _toy_model_with_bn():
    inp = tf.keras.Input((4,))
    x = tf.keras.layers.Dense(4)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.Model(inp, x)


def test_freeze_model_keeps_batch_norm_trainable_by_default():
    """`freeze_batch_norm=False` is the default - despite the name reading
    like "don't touch BN", it means "freeze everything *except* BN" (BN
    stays trainable so its running statistics keep adapting even while the
    rest of a pretrained backbone is frozen - a common, deliberate pattern,
    confirmed by reading the implementation rather than assumed from the
    parameter name alone).
    """
    model = _toy_model_with_bn()
    freeze_model(model)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            assert layer.trainable
        else:
            assert not layer.trainable


def test_freeze_model_freezes_batch_norm_too_when_requested():
    model = _toy_model_with_bn()
    freeze_model(model, freeze_batch_norm=True)
    assert all(not layer.trainable for layer in model.layers)


def test_unfreeze_model_unfreezes_everything_when_requested():
    model = _toy_model_with_bn()
    freeze_model(model, freeze_batch_norm=True)
    unfreeze_model(model, unfreeze_batch_norm=True)
    assert all(layer.trainable for layer in model.layers)
