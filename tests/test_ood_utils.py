"""Tests for `swiss_army_keras.ood_utils` (plan.md §13.2, Phase 10) -
Mahalanobis-distance OOD detection (Lee et al. 2018, arXiv:1807.03888) and
the connected-component spatial aggregation shared with
`layer_utils.energy_score`'s own dense (segmentation) output. Pure numpy,
standalone - no model building needed to test these, matching
`_model_detection.py`'s `atss_assign_targets` test-file convention.
"""

import numpy as np
import pytest
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis

from swiss_army_keras.ood_utils import (
    aggregate_anomaly_map,
    calibrate_threshold,
    dense_mahalanobis_distance,
    fit_class_statistics,
    mahalanobis_distance,
)


@pytest.fixture
def toy_two_class_gaussian():
    """Two well-separated 3D Gaussian classes sharing a known covariance -
    lets `fit_class_statistics`'s recovered mean/covariance be checked
    against known ground truth, not just "didn't crash".
    """
    rng = np.random.default_rng(0)
    true_cov = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 0.5]])
    mean0 = np.array([0.0, 0.0, 0.0])
    mean1 = np.array([5.0, 5.0, 5.0])

    n = 2000
    x0 = rng.multivariate_normal(mean0, true_cov, n)
    x1 = rng.multivariate_normal(mean1, true_cov, n)
    embeddings = np.concatenate([x0, x1], axis=0)
    labels = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)

    return embeddings, labels, mean0, mean1, true_cov


class TestFitClassStatistics:
    def test_recovers_known_means_and_covariance(self, toy_two_class_gaussian):
        embeddings, labels, mean0, mean1, true_cov = toy_two_class_gaussian
        class_means, covariance_cholesky = fit_class_statistics(embeddings, labels, n_classes=2)

        np.testing.assert_allclose(class_means[0], mean0, atol=0.15)
        np.testing.assert_allclose(class_means[1], mean1, atol=0.15)

        reconstructed_cov = covariance_cholesky @ covariance_cholesky.T
        np.testing.assert_allclose(reconstructed_cov, true_cov, atol=0.2)

    def test_covariance_cholesky_is_lower_triangular(self, toy_two_class_gaussian):
        embeddings, labels, *_ = toy_two_class_gaussian
        _, covariance_cholesky = fit_class_statistics(embeddings, labels, n_classes=2)
        np.testing.assert_allclose(covariance_cholesky, np.tril(covariance_cholesky))


class TestMahalanobisDistance:
    def test_matches_scipy_mahalanobis_for_predicted_class(self, toy_two_class_gaussian):
        embeddings, labels, mean0, mean1, true_cov = toy_two_class_gaussian
        class_means, covariance_cholesky = fit_class_statistics(embeddings, labels, n_classes=2)

        test_point = np.array([1.0, 1.0, 1.0])
        expected = scipy_mahalanobis(test_point, class_means[0], np.linalg.inv(true_cov))
        got = mahalanobis_distance(test_point, class_means, covariance_cholesky, predicted_class=0)

        np.testing.assert_allclose(got, expected, rtol=0.05)

    def test_closest_class_is_minimum_over_all_classes(self, toy_two_class_gaussian):
        embeddings, labels, *_ = toy_two_class_gaussian
        class_means, covariance_cholesky = fit_class_statistics(embeddings, labels, n_classes=2)

        near_class1 = np.array([4.8, 5.1, 4.9])
        d_closest = mahalanobis_distance(near_class1, class_means, covariance_cholesky, predicted_class=None)
        d_to_0 = mahalanobis_distance(near_class1, class_means, covariance_cholesky, predicted_class=0)
        d_to_1 = mahalanobis_distance(near_class1, class_means, covariance_cholesky, predicted_class=1)

        assert d_to_1 < d_to_0, "a point near class 1's mean must be closer to class 1"
        np.testing.assert_allclose(d_closest, min(d_to_0, d_to_1))

    def test_batch_input_matches_per_example_single_calls(self, toy_two_class_gaussian):
        embeddings, labels, *_ = toy_two_class_gaussian
        class_means, covariance_cholesky = fit_class_statistics(embeddings, labels, n_classes=2)

        point_a = np.array([1.0, 1.0, 1.0])
        point_b = np.array([4.8, 5.1, 4.9])
        batch = np.stack([point_a, point_b])

        batch_dist = mahalanobis_distance(batch, class_means, covariance_cholesky, predicted_class=[0, 1])
        single_a = mahalanobis_distance(point_a, class_means, covariance_cholesky, predicted_class=0)
        single_b = mahalanobis_distance(point_b, class_means, covariance_cholesky, predicted_class=1)

        np.testing.assert_allclose(batch_dist, [single_a, single_b], atol=1e-6)


class TestDenseMahalanobisDistance:
    def test_shape_and_matches_single_point_value(self, toy_two_class_gaussian):
        embeddings, labels, *_ = toy_two_class_gaussian
        class_means, covariance_cholesky = fit_class_statistics(embeddings, labels, n_classes=2)

        point = np.array([4.8, 5.1, 4.9])
        feature_map = np.tile(point, (8, 8, 1)).astype(np.float64)

        dense = dense_mahalanobis_distance(feature_map, class_means, covariance_cholesky)
        expected = mahalanobis_distance(point, class_means, covariance_cholesky)

        assert dense.shape == (8, 8)
        np.testing.assert_allclose(dense, expected, atol=1e-6)

    def test_predicted_class_map_selects_per_pixel_class(self, toy_two_class_gaussian):
        embeddings, labels, *_ = toy_two_class_gaussian
        class_means, covariance_cholesky = fit_class_statistics(embeddings, labels, n_classes=2)

        point = np.array([2.5, 2.5, 2.5])  # roughly equidistant from both classes
        feature_map = np.tile(point, (2, 2, 1)).astype(np.float64)
        predicted_class_map = np.array([[0, 1], [0, 1]])

        dense = dense_mahalanobis_distance(feature_map, class_means, covariance_cholesky, predicted_class_map)
        expected_to_0 = mahalanobis_distance(point, class_means, covariance_cholesky, predicted_class=0)
        expected_to_1 = mahalanobis_distance(point, class_means, covariance_cholesky, predicted_class=1)

        np.testing.assert_allclose(dense[:, 0], expected_to_0, atol=1e-6)
        np.testing.assert_allclose(dense[:, 1], expected_to_1, atol=1e-6)


class TestAggregateAnomalyMap:
    def test_large_blob_is_kept_and_flags_the_image(self):
        score_map = np.zeros((20, 20))
        score_map[2:10, 2:10] = 5.0  # 64 pixels = 16% of the 400-pixel image

        flagged, mask = aggregate_anomaly_map(score_map, threshold=1.0, min_blob_fraction=0.01)

        assert flagged is True
        assert mask.sum() == 64
        assert mask[5, 5]

    def test_small_blob_below_min_fraction_is_discarded(self):
        score_map = np.zeros((20, 20))
        score_map[15, 15] = 5.0  # single pixel = 0.25% of the image

        flagged, mask = aggregate_anomaly_map(score_map, threshold=1.0, min_blob_fraction=0.01)

        assert flagged is False
        assert mask.sum() == 0

    def test_no_above_threshold_pixels_yields_no_flag(self):
        flagged, mask = aggregate_anomaly_map(np.zeros((20, 20)), threshold=1.0)
        assert flagged is False
        assert mask.sum() == 0

    def test_two_blobs_only_large_one_kept(self):
        score_map = np.zeros((20, 20))
        score_map[0:8, 0:8] = 5.0    # 64 pixels - large, kept
        score_map[15, 15] = 5.0      # 1 pixel - small, discarded

        flagged, mask = aggregate_anomaly_map(score_map, threshold=1.0, min_blob_fraction=0.01)

        assert flagged is True
        assert mask.sum() == 64
        assert not mask[15, 15]


def test_calibrate_threshold_matches_numpy_percentile():
    scores = np.arange(100)
    threshold = calibrate_threshold(scores, percentile=95.0)
    np.testing.assert_allclose(threshold, np.percentile(scores, 95.0))
