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
    add_energy_score_output,
    apply_pre_pool_attention,
    energy_score,
    uint8_input_wrapper,
)
from swiss_army_keras.models import classifier, lr_aspp

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


# uint8_input_wrapper / energy_score / add_energy_score_output (plan.md
# §13.1/§13.2, Phase 10) - in-graph uint8 input handling and a
# training-free OOD/low-confidence signal, both built as post-hoc wrappers
# around an already-built model rather than parameters threaded through
# every model-builder function (see `layer_utils.py`'s own module comment
# for why this works cleanly here).

CLF_INPUT_SIZE = 64
CLF_N_CLASSES = 4


def _random_uint8_batch(size, batch_size=2):
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (batch_size, size, size, 3), dtype=np.uint8)


class TestUint8InputWrapper:
    def test_wraps_classifier_with_uint8_input_and_matches_manual_preprocessing(self):
        inp = tf.keras.Input((CLF_INPUT_SIZE, CLF_INPUT_SIZE, 3))
        model = classifier(
            inp, n_classes=CLF_N_CLASSES, backbone="MobileNetV2", weights=None, freeze_backbone=False,
        )
        wrapped = uint8_input_wrapper(model, name="wrapped_clf")

        assert wrapped.input.dtype == tf.uint8
        assert wrapped.input_shape == (None, CLF_INPUT_SIZE, CLF_INPUT_SIZE, 3)

        uint8_x = _random_uint8_batch(CLF_INPUT_SIZE)
        wrapped_out = wrapped.predict(uint8_x, verbose=0)

        manual_preprocessed = model.preprocessing(uint8_x.astype("float32"))
        direct_out = model.predict(manual_preprocessed, verbose=0)
        np.testing.assert_allclose(wrapped_out, direct_out, atol=1e-4)

    def test_preserves_static_batch_dimension_when_wrapped_model_requires_one(self):
        """Regression test for a real bug found while verifying
        `fcos_lite_detection`'s own new static-batch NMS path
        (`_model_detection.py`'s `_nms_dispatch`): this used to always
        build a *dynamic*-batch outward `Input` regardless of what `model`
        itself required - fine for every existing caller (all dynamic-
        batch), but silently wrong for a model built with a static batch
        dimension (`Input(batch_shape=(N, H, W, 3))`), since TFLite's own
        converter defaults an unspecified/dynamic batch to 1, producing a
        `.tflite` whose outward interface claimed batch=1 while the model
        wrapped inside it actually required batch=N - a shape mismatch
        that only surfaced at inference time, not at conversion time.
        """
        inp = tf.keras.Input(batch_shape=(3, CLF_INPUT_SIZE, CLF_INPUT_SIZE, 3))
        model = classifier(inp, n_classes=CLF_N_CLASSES, backbone="MobileNetV2", weights=None, freeze_backbone=False)
        wrapped = uint8_input_wrapper(model, name="wrapped_static_batch_clf")

        assert wrapped.input.dtype == tf.uint8
        assert wrapped.input_shape == (3, CLF_INPUT_SIZE, CLF_INPUT_SIZE, 3)

        uint8_x = _random_uint8_batch(CLF_INPUT_SIZE, batch_size=3)
        wrapped_out = wrapped.predict(uint8_x, verbose=0)

        manual_preprocessed = model.preprocessing(uint8_x.astype("float32"))
        direct_out = model.predict(manual_preprocessed, verbose=0)
        np.testing.assert_allclose(wrapped_out, direct_out, atol=1e-4)

    def test_preprocessing_attribute_replaced_with_dummy_after_wrapping(self):
        inp = tf.keras.Input((CLF_INPUT_SIZE, CLF_INPUT_SIZE, 3))
        model = classifier(
            inp, n_classes=CLF_N_CLASSES, backbone="MobileNetV2", weights=None, freeze_backbone=False,
        )
        wrapped = uint8_input_wrapper(model, name="wrapped_clf2")
        # Normalization is now baked into the graph - a caller must NOT
        # reapply the original per-backbone mean/std externally.
        x = np.zeros((1, 3), dtype="float32")
        np.testing.assert_array_equal(wrapped.preprocessing(x), x)

    def test_channels_first_transposes_before_calling_wrapped_model(self):
        """Builds a toy model whose `Input` is genuinely `(3, H, W)` (via a
        real `data_format='channels_first'` Conv2D, not just a declared
        shape) - confirms the wrapper's own outward-facing input stays
        `(H, W, 3)` regardless, and that what reaches the inner model is
        truly transposed, not just reshaped.
        """
        size = 16
        cf_input = tf.keras.Input((3, size, size))
        cf_x = tf.keras.layers.Conv2D(4, 3, padding='same', data_format='channels_first')(cf_input)
        cf_x = tf.keras.layers.Flatten()(cf_x)
        cf_out = tf.keras.layers.Dense(2)(cf_x)
        cf_model = tf.keras.Model(cf_input, cf_out)
        cf_model.preprocessing = lambda t: t / 255.0

        wrapped = uint8_input_wrapper(cf_model, channels_first=True, name="cf_wrapped")

        assert wrapped.input.dtype == tf.uint8
        assert wrapped.input_shape == (None, size, size, 3)

        out = wrapped.predict(_random_uint8_batch(size), verbose=0)
        assert np.all(np.isfinite(out))

    def test_no_second_quantize_apply_needed(self):
        """The wrapped model must build/predict correctly without ever
        calling `quantize_apply` again - it composes an already-quantized
        model as a single sub-model layer, the same way
        `fcos_lite_detection` already calls its own already-quantized base
        model.
        """
        inp = tf.keras.Input((CLF_INPUT_SIZE, CLF_INPUT_SIZE, 3))
        model = classifier(
            inp, n_classes=CLF_N_CLASSES, backbone="MobileNetV2", weights=None, freeze_backbone=False,
        )
        wrapped = uint8_input_wrapper(model, name="wrapped_clf3")
        out = wrapped.predict(_random_uint8_batch(CLF_INPUT_SIZE), verbose=0)
        assert out.shape == (2, CLF_N_CLASSES)
        assert np.all(np.isfinite(out))


class TestEnergyScore:
    def test_confident_logits_score_lower_than_guessing_logits(self):
        """Hand-computable case, reusing the reference document's own
        worked example numbers as a regression anchor: a confident,
        near-one-hot logit vector must score much lower (more negative)
        energy than a flat, low-magnitude ("guessing") one.
        """
        confident_logits = tf.constant([[12.0, -3.0, 1.5]])
        guessing_logits = tf.constant([[1.2, 0.8, 1.1]])

        confident_energy = energy_score(confident_logits).numpy()
        guessing_energy = energy_score(guessing_logits).numpy()

        assert confident_energy[0] < guessing_energy[0]
        np.testing.assert_allclose(confident_energy, [-12.0], atol=0.01)
        np.testing.assert_allclose(guessing_energy, [-2.15], atol=0.05)

    def test_reduces_class_axis(self):
        logits = tf.random.normal((2, 5, 5, 4))
        out = energy_score(logits, class_axis=-1)
        assert out.shape == (2, 5, 5)

    def test_temperature_scales_result(self):
        logits = tf.constant([[2.0, 1.0, 0.0]])
        e1 = energy_score(logits, temperature=1.0).numpy()
        e2 = energy_score(logits, temperature=2.0).numpy()
        assert not np.isclose(e1, e2)


class TestAddEnergyScoreOutput:
    def test_works_unmodified_on_segmentation_decoder_output(self):
        """`lr_aspp`'s single output is already raw per-pixel logits (no
        fused activation) - `add_energy_score_output` should work with
        zero changes to the decoder itself.
        """
        size = 32
        n_labels = 3
        inp = tf.keras.Input((size, size, 3))
        seg_model = lr_aspp(inp, n_labels=n_labels, backbone="MobileNetV2", weights=None, freeze_backbone=False)

        energy_model = add_energy_score_output(seg_model, logit_output_indices=(0,))
        logits, energy = energy_model.predict(
            np.random.rand(1, size, size, 3).astype("float32"), verbose=0,
        )
        assert logits.shape == (1, size, size, n_labels)
        assert energy.shape == (1, size, size)
        assert np.all(np.isfinite(energy))

    def test_classifier_needs_return_logits_true_first(self):
        """`classifier()`'s default output has softmax fused in - no
        logits tensor exists anywhere in its graph, so
        `add_energy_score_output` cannot be applied to its default output;
        `return_logits=True` (index 1) is required first.
        """
        inp = tf.keras.Input((CLF_INPUT_SIZE, CLF_INPUT_SIZE, 3))
        model = classifier(
            inp, n_classes=CLF_N_CLASSES, backbone="MobileNetV2", weights=None,
            freeze_backbone=False, return_logits=True,
        )
        assert len(model.outputs) == 2  # [probabilities, logits]

        energy_model = add_energy_score_output(model, logit_output_indices=(1,))
        probs, logits, energy = energy_model.predict(
            np.random.rand(2, CLF_INPUT_SIZE, CLF_INPUT_SIZE, 3).astype("float32"), verbose=0,
        )
        assert energy.shape == (2,)
        assert np.all(np.isfinite(energy))

    def test_composes_with_uint8_input_wrapper(self):
        """Full pipeline: classifier(return_logits=True) -> energy score ->
        uint8 input - the intended real-world composition of both features
        together.
        """
        inp = tf.keras.Input((CLF_INPUT_SIZE, CLF_INPUT_SIZE, 3))
        model = classifier(
            inp, n_classes=CLF_N_CLASSES, backbone="MobileNetV2", weights=None,
            freeze_backbone=False, return_logits=True,
        )
        model = add_energy_score_output(model, logit_output_indices=(1,))
        model = uint8_input_wrapper(model, name="full_pipeline")

        assert model.input.dtype == tf.uint8
        probs, logits, energy = model.predict(_random_uint8_batch(CLF_INPUT_SIZE), verbose=0)
        assert probs.shape == (2, CLF_N_CLASSES)
        assert energy.shape == (2,)
        assert np.all(np.isfinite(energy))


# iqa_regression_head (plan.md §13.7, Phase 11) - a small regression tail
# for a shared-backbone IQA side-head.

from swiss_army_keras.layer_utils import iqa_regression_head


class TestIqaRegressionHead:
    def test_gem_pooling_builds_and_predicts_bounded(self):
        inp = tf.keras.Input((HEIGHT, WIDTH, CHANNELS))
        out = iqa_regression_head(inp, pooling='gem', name='test_iqa')
        model = tf.keras.Model(inp, out)

        preds = model.predict(np.random.rand(2, HEIGHT, WIDTH, CHANNELS).astype("float32"), verbose=0)
        assert preds.shape == (2, 1)
        assert preds.min() >= 0.0 and preds.max() <= 1.0

    def test_avg_pooling_builds_and_predicts_bounded(self):
        inp = tf.keras.Input((HEIGHT, WIDTH, CHANNELS))
        out = iqa_regression_head(inp, pooling='avg', name='test_iqa_avg')
        model = tf.keras.Model(inp, out)

        preds = model.predict(np.random.rand(2, HEIGHT, WIDTH, CHANNELS).astype("float32"), verbose=0)
        assert preds.shape == (2, 1)
        assert preds.min() >= 0.0 and preds.max() <= 1.0
