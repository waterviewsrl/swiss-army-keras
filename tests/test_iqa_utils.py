"""Tests for `swiss_army_keras.iqa_utils` (plan.md §13.7, Phase 11) -
the training-pipeline companion to `layer_utils.iqa_regression_head`:
wrapping a `(image, label)` dataset with a teacher-distilled pseudo-label,
and a small dependency-free reference teacher implementation.
"""

import numpy as np
import tensorflow as tf

from swiss_army_keras.iqa_utils import add_teacher_iqa_labels, laplacian_variance_quality

IMG_SIZE = 16


def _checkerboard(size=32):
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[::2, ::2] = 255.0
    img[1::2, 1::2] = 255.0
    return img


def _heavily_blurred(size=32):
    checkerboard = _checkerboard(size)[None]
    small = tf.image.resize(checkerboard, (4, 4), method='gaussian')
    return tf.image.resize(small, (size, size), method='gaussian')[0].numpy()


class TestLaplacianVarianceQuality:
    def test_sharp_image_scores_higher_than_blurred(self):
        sharp_score = laplacian_variance_quality(tf.constant(_checkerboard())).numpy()
        blurred_score = laplacian_variance_quality(tf.constant(_heavily_blurred())).numpy()
        assert sharp_score > blurred_score

    def test_output_is_bounded(self):
        score = laplacian_variance_quality(tf.constant(_checkerboard()))
        assert 0.0 <= float(score) <= 1.0

    def test_flat_image_has_zero_underlying_variance(self):
        """Regression test for a real bug found while building this:
        `'SAME'` padding's implicit zero-border made even a perfectly flat
        (zero-Laplacian-everywhere) image register a nonzero variance from
        border artifacts alone - fixed by using `'VALID'` padding instead.
        A flat image's score should sit at the low end (`< 0.5`, the
        `blur_threshold`'s own midpoint), not be inflated by border noise.
        """
        flat = np.full((16, 16, 3), 128.0, dtype=np.float32)
        score = laplacian_variance_quality(tf.constant(flat))
        assert float(score) < 0.5

    def test_grayscale_input_accepted(self):
        gray = np.random.rand(16, 16, 1).astype("float32") * 255.0
        score = laplacian_variance_quality(tf.constant(gray))
        assert 0.0 <= float(score) <= 1.0


class TestAddTeacherIqaLabels:
    def test_wraps_dataset_with_dict_labels_at_correct_keys(self):
        rng = np.random.default_rng(0)
        images = rng.random((6, IMG_SIZE, IMG_SIZE, 3)).astype("float32")
        labels = rng.integers(0, 3, (6,)).astype("float32")
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        def stub_teacher(image):
            return np.float32(0.5)

        wrapped = add_teacher_iqa_labels(dataset, stub_teacher, main_label_key='task', iqa_label_key='quality')

        for image, label_dict in wrapped.take(1):
            assert set(label_dict.keys()) == {'task', 'quality'}
            assert image.shape == (IMG_SIZE, IMG_SIZE, 3)
            np.testing.assert_allclose(float(label_dict['quality']), 0.5)

    def test_teacher_scores_reach_the_correct_examples(self):
        """Not just "the right shape" - the *value* `teacher_fn` computes
        for each specific image must land on that same image's own label,
        not get shuffled/misaligned across the `tf.data` pipeline.
        """
        images = np.stack([
            np.full((IMG_SIZE, IMG_SIZE, 3), 0.0, dtype="float32"),
            np.full((IMG_SIZE, IMG_SIZE, 3), 1.0, dtype="float32"),
            np.full((IMG_SIZE, IMG_SIZE, 3), 2.0, dtype="float32"),
        ])
        labels = np.array([0, 1, 2], dtype="float32")
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        def mean_as_score(image):
            return np.float32(tf.reduce_mean(image).numpy() / 10.0)

        wrapped = add_teacher_iqa_labels(dataset, mean_as_score)

        scores = [float(label_dict['iqa']) for _, label_dict in wrapped]
        np.testing.assert_allclose(scores, [0.0, 0.1, 0.2], atol=1e-5)

    def test_composes_with_batching_and_dict_loss_compile(self):
        """End-to-end shape check: batching the wrapped dataset and using
        its dict labels with `model.compile(loss={...})`'s own dict
        convention must work together without special-casing.
        """
        rng = np.random.default_rng(1)
        images = rng.random((8, IMG_SIZE, IMG_SIZE, 3)).astype("float32")
        labels = rng.integers(0, 2, (8,)).astype("float32")
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        wrapped = add_teacher_iqa_labels(
            dataset, lambda image: np.float32(0.7), main_label_key='task', iqa_label_key='quality',
        ).batch(4)

        for _, label_dict in wrapped.take(1):
            assert label_dict['task'].shape == (4,)
            assert label_dict['quality'].shape == (4,)
