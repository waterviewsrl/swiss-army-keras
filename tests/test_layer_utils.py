"""Tests for the lightweight attention/pooling layers added to
`swiss_army_keras.layer_utils` under plan.md §9.1/§13 Phase 5:
`GeMPooling2D`, `ECABlock`, `CoordAttLite`, and the `apply_pre_pool_attention`
dispatch helper they're wired in through in `_model_classifier.py`.
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras.layer_utils import (
    CoordAttLite,
    ECABlock,
    GeMPooling2D,
    apply_pre_pool_attention,
)

BATCH, HEIGHT, WIDTH, CHANNELS = 2, 16, 16, 8


@pytest.fixture
def random_features():
    rng = np.random.default_rng(0)
    return tf.constant(rng.random((BATCH, HEIGHT, WIDTH, CHANNELS)).astype("float32"))


def test_gem_pooling_output_shape_and_gradient(random_features):
    x = tf.Variable(random_features)
    with tf.GradientTape() as tape:
        y = GeMPooling2D()(x)
        loss = tf.reduce_sum(y)
    grad = tape.gradient(loss, x)

    assert y.shape == (BATCH, CHANNELS)
    assert grad is not None
    assert np.all(np.isfinite(grad.numpy()))


def test_gem_pooling_p_is_trainable_by_default():
    layer = GeMPooling2D()
    layer.build((None, HEIGHT, WIDTH, CHANNELS))
    assert layer.p.trainable


def test_gem_pooling_p_fixed_when_not_trainable():
    layer = GeMPooling2D(trainable_p=False)
    layer.build((None, HEIGHT, WIDTH, CHANNELS))
    assert not layer.p.trainable


def test_gem_pooling_reduces_to_average_pooling_at_p_equals_one(random_features):
    layer = GeMPooling2D(p_init=1.0, trainable_p=False)
    gem_out = layer(random_features)
    avg_out = tf.reduce_mean(random_features, axis=[1, 2])
    np.testing.assert_allclose(gem_out.numpy(), avg_out.numpy(), rtol=1e-5, atol=1e-5)


def test_gem_pooling_config_roundtrip():
    layer = GeMPooling2D(p_init=2.5, epsilon=1e-5, trainable_p=False)
    restored = GeMPooling2D.from_config(layer.get_config())
    assert restored.p_init == 2.5
    assert restored.epsilon == 1e-5
    assert restored.trainable_p is False


def test_eca_block_output_shape_and_gradient(random_features):
    x = tf.Variable(random_features)
    with tf.GradientTape() as tape:
        y = ECABlock()(x)
        loss = tf.reduce_sum(y)
    grad = tape.gradient(loss, x)

    assert y.shape == random_features.shape
    assert grad is not None
    assert np.all(np.isfinite(grad.numpy()))


@pytest.mark.parametrize("channels,expected_min_kernel", [(8, 3), (2048, 3)])
def test_eca_block_adaptive_kernel_size_is_odd_and_at_least_3(channels, expected_min_kernel):
    layer = ECABlock()
    layer.build((None, HEIGHT, WIDTH, channels))
    assert layer._resolved_kernel_size % 2 == 1
    assert layer._resolved_kernel_size >= expected_min_kernel


def test_eca_block_explicit_kernel_size_is_respected():
    layer = ECABlock(kernel_size=5)
    layer.build((None, HEIGHT, WIDTH, CHANNELS))
    assert layer._resolved_kernel_size == 5


def test_coordatt_lite_output_shape_and_gradient(random_features):
    x = tf.Variable(random_features)
    with tf.GradientTape() as tape:
        y = CoordAttLite()(x)
        loss = tf.reduce_sum(y)
    grad = tape.gradient(loss, x)

    assert y.shape == random_features.shape
    assert grad is not None
    assert np.all(np.isfinite(grad.numpy()))


def test_coordatt_lite_gives_different_rows_different_attention_unlike_eca():
    '''Regression/documentation test for CoordAtt's whole reason for
    existing (plan.md §9.1): plain global-average-pool-based channel
    attention (ECA/SE) reduces the entire spatial extent to *one*
    per-channel scalar gate, broadcast identically over every spatial
    position by construction — so within a single image, its gate at a
    "top" row and a "bottom" row are necessarily equal, no matter how
    differently those two rows' content behaves. CoordAtt pools along H and
    W *separately* before its gate, so a row with different content gets a
    genuinely different attention weight within that same image.

    (This is deliberately a *within-one-image* comparison, not a
    "permute the rows of two separate images" one: with only 1x1 convs, as
    implemented here, CoordAtt's row-attention is itself
    position-*independent* along H — it responds to *what value* a row's
    pooled descriptor holds, not *which index* that row is at, so comparing
    across two images whose row values are just a permutation of each
    other would show no difference either, for reasons unrelated to
    CoordAtt losing positional information. Different content within the
    *same* image is the property that actually distinguishes it from ECA.)
    '''
    top_value, bottom_value = 3.0, 1.0
    x = np.zeros((1, HEIGHT, WIDTH, CHANNELS), dtype="float32")
    x[:, : HEIGHT // 2, :, :] = top_value
    x[:, HEIGHT // 2:, :, :] = bottom_value

    eca = ECABlock()
    eca_out = eca(tf.constant(x)).numpy()
    eca_gate_top = eca_out[:, : HEIGHT // 2, :, :] / top_value
    eca_gate_bottom = eca_out[:, HEIGHT // 2:, :, :] / bottom_value
    # Holds by construction (ECA has exactly one gate value per channel for
    # the whole image) - included to make the contrast with CoordAtt below
    # concrete, not as a meaningful check of ECA's own correctness.
    np.testing.assert_allclose(eca_gate_top.mean(), eca_gate_bottom.mean(), rtol=1e-5)

    coordatt = CoordAttLite()
    ca_out = coordatt(tf.constant(x)).numpy()
    ca_gate_top = ca_out[:, : HEIGHT // 2, :, :] / top_value
    ca_gate_bottom = ca_out[:, HEIGHT // 2:, :, :] / bottom_value
    assert not np.isclose(ca_gate_top.mean(), ca_gate_bottom.mean(), rtol=1e-3), (
        "CoordAttLite should give different rows (with different content) a "
        "different attention weight within the same image - if this fails, "
        "it's degenerated into plain (position-blind) channel attention"
    )


def test_apply_pre_pool_attention_none_is_identity(random_features):
    out = apply_pre_pool_attention(random_features, None)
    assert out is random_features


@pytest.mark.parametrize("mode,layer_cls", [("eca", ECABlock), ("coordatt", CoordAttLite)])
def test_apply_pre_pool_attention_dispatches_to_correct_layer(mode, layer_cls):
    """Uses a symbolic `Input` (not an eager tensor, unlike this file's
    other tests) specifically so `_keras_history` - only populated inside a
    Functional-API graph - is available to introspect which layer type
    `apply_pre_pool_attention` actually built.
    """
    inp = tf.keras.Input((HEIGHT, WIDTH, CHANNELS))
    out = apply_pre_pool_attention(inp, mode, name="test")
    assert out.shape[1:] == (HEIGHT, WIDTH, CHANNELS)
    assert out._keras_history.layer.__class__ is layer_cls
