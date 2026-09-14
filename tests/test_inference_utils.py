"""Tests for `swiss_army_keras.inference_utils.tta_predict` (plan.md §4.3/§13
Phase 4). Kept in its own module rather than `quantization_utils.py`/
`training_utils.py` (both options plan.md allowed) since both of those
currently fail to import at all in a clean environment - `ModuleNotFoundError:
No module named 'matplotlib'`, a real pre-existing bug documented in plan.md
§10, not something to build new code on top of.
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras.inference_utils import tta_predict
from swiss_army_keras._model_lr_aspp import lr_aspp

INPUT_SIZE = 64
N_LABELS = 3


def _dynamic_conv_model():
    """A purely convolutional, resolution-agnostic toy model (`Input((None,
    None, 3))`, no fixed-size pooling) - used to test the `scales` (multi-scale)
    path, which none of this repo's own decoders support today (see
    `tta_predict`'s docstring: they all bake a concrete spatial size into a
    "global pool" layer at build time).
    """
    inp = tf.keras.Input((None, None, 3))
    h = tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu')(inp)
    out = tf.keras.layers.Conv2D(N_LABELS, 1, padding='same')(h)
    return tf.keras.Model(inp, out)


@pytest.fixture
def random_image():
    rng = np.random.default_rng(0)
    return rng.random((INPUT_SIZE, INPUT_SIZE, 3)).astype("float32")


def test_tta_predict_no_augmentation_matches_plain_forward_pass(random_image):
    """scales=(1.0,), hflip=False, vflip=False should reduce to exactly one
    plain forward pass - the degenerate case every other case builds on.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = lr_aspp(inp, n_labels=N_LABELS, backbone=None)

    plain = model(random_image[tf.newaxis, ...], training=False)[0].numpy()
    tta = tta_predict(model, random_image, scales=(1.0,), hflip=False, vflip=False)

    np.testing.assert_allclose(plain, tta, rtol=1e-5, atol=1e-5)


def test_tta_predict_flip_only_against_real_decoder_builds_and_predicts(random_image):
    """Flip-based TTA (no rescaling) works against a real, fixed-input-shape
    decoder from this repo, since a flip never changes spatial size - the
    combination the docstring promises actually works today.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = lr_aspp(inp, n_labels=N_LABELS, backbone=None)

    out = tta_predict(model, random_image, scales=(1.0,), hflip=True, vflip=True)
    assert out.shape == (INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_tta_predict_multiscale_requires_dynamic_shape_model(random_image):
    """Regression/documentation test for the real limitation found while
    building this helper (plan.md §4.3): `scales != (1.0,)` against a
    fixed-`Input`-shape model (every decoder in this repo, as built by
    default) raises, since those decoders bake a concrete spatial size into
    a "global pool" layer at build time. Multi-scale TTA only works against
    a genuinely resolution-agnostic model.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = lr_aspp(inp, n_labels=N_LABELS, backbone=None)

    with pytest.raises(Exception):
        tta_predict(model, random_image, scales=(1.0, 0.75), hflip=False, vflip=False)


def test_tta_predict_multiscale_against_dynamic_shape_model(random_image):
    model = _dynamic_conv_model()
    out = tta_predict(model, random_image, scales=(0.75, 1.0, 1.25), hflip=True, vflip=True)
    assert out.shape == (INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_tta_predict_averaging_reduces_variance_on_noisy_model(random_image):
    """A cheap sanity check that TTA is actually *averaging*, not just
    running extra forward passes and discarding them: for a model whose
    output has injected per-call noise, more augmentation variants should
    pull the averaged prediction closer to the (noise-free) single-pass
    baseline than a single noisy pass would be, on average.
    """
    rng = tf.random.Generator.from_seed(0)

    inp = tf.keras.Input((None, None, 3))
    h = tf.keras.layers.Conv2D(4, 3, padding='same', activation='relu')(inp)
    out = tf.keras.layers.Conv2D(N_LABELS, 1, padding='same')(h)
    base_model = tf.keras.Model(inp, out)

    class NoisyWrapper:
        def __call__(self, x, training=False):
            return base_model(x, training=training) + rng.normal(tf.shape(x)[:1].numpy().tolist() + base_model(x).shape[1:], stddev=1.0)

    # Compare the *deterministic* base model's output (ground truth) against
    # a single noisy call vs. an averaged (4-variant) TTA call, both built
    # from the same noisy wrapper.
    baseline = base_model(random_image[tf.newaxis, ...], training=False)[0].numpy()

    noisy = NoisyWrapper()
    single_noisy = noisy(random_image[tf.newaxis, ...])[0].numpy()
    averaged = tta_predict(noisy, random_image, scales=(1.0,), hflip=True, vflip=True)

    single_error = np.mean((single_noisy - baseline) ** 2)
    averaged_error = np.mean((averaged - baseline) ** 2)
    assert averaged_error < single_error
