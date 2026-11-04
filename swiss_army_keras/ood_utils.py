'''Mahalanobis-distance out-of-distribution (OOD) detection (plan.md
§13.2, Phase 10) — a lightweight, production-oriented companion to
`swiss_army_keras.layer_utils.energy_score`/`add_energy_score_output`.

----------
Lee, K., Lee, K., Lee, H. and Shin, J., 2018. A simple unified framework
for detecting out-of-distribution samples and adversarial attacks.
NeurIPS. arXiv:1807.03888.

----------
Unlike Energy Score (a training-free function of a model's own already-
computed logits, needing zero offline fitting), Mahalanobis distance needs
an **offline calibration pass** on a trained model's embeddings before it
can be used — this module is deliberately separate from model
construction, matching `_model_detection.py`'s `atss_assign_targets`'s own
"pure numpy, standalone, independently testable, combination documented in
each function's own docstring, not forced into one API" convention.

Typical usage: after training, run the training set through a model built
with `classifier(..., return_logits=True)` (or any model exposing a
pooled/dense pre-classification-head feature tensor), collect
`(embeddings, labels)`, call `fit_class_statistics(...)` once, then
`mahalanobis_distance(...)` (classification/detection) or
`dense_mahalanobis_distance(...)` (segmentation, one embedding per spatial
location) at inference on new embeddings. `aggregate_anomaly_map(...)`
turns a dense per-pixel score map (either this module's own
`dense_mahalanobis_distance` output, or `layer_utils.energy_score`'s dense
output) into a single flagged/not-flagged decision via connected-component
filtering, so naturally-high-score pixels around object boundaries don't
trigger false alarms on their own. `calibrate_threshold(...)` turns a set
of validation scores into the percentile-based threshold both aggregation
paths need — a real, data-dependent calibration step, not something
bakeable into a model's architecture.
'''

from __future__ import absolute_import

import numpy as np
from scipy.linalg import solve_triangular
from scipy.ndimage import label


def fit_class_statistics(embeddings, labels, n_classes):
    '''Offline (once, after training): per-class mean embedding vectors and
    one **shared/tied** covariance matrix across all classes (Lee et al.
    2018's own choice — a single pooled covariance kept tractable, rather
    than a separate, harder-to-estimate covariance per class).

    Input
    ----------
        embeddings: `(N, C)` array — one `C`-dimensional embedding per
                    training example (e.g. a classifier's pooled
                    penultimate feature, or a segmentation decoder's dense
                    feature map flattened to `(H*W, C)` per image and
                    concatenated across images).
        labels: `(N,)` array of integer class indices (`0..n_classes-1`).
        n_classes: total number of classes.

    Output
    ----------
        `(class_means, covariance_cholesky)`:
            class_means: `(n_classes, C)` array.
            covariance_cholesky: `(C, C)` lower-triangular Cholesky factor
                `L` of the shared covariance (`Sigma = L @ L.T`) — returned
                instead of `Sigma` itself, or its inverse, so
                `mahalanobis_distance` can solve a triangular system
                instead of inverting a matrix (faster, numerically
                stable). A small ridge (`1e-6 * I`) is added to the raw
                pooled covariance before decomposing it, guarding against
                a near-singular matrix on small/degenerate calibration sets.
    '''
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels)
    n_features = embeddings.shape[1]

    class_means = np.zeros((n_classes, n_features), dtype=np.float64)
    centered = np.zeros_like(embeddings)
    for c in range(n_classes):
        mask = labels == c
        if not mask.any():
            continue
        class_means[c] = embeddings[mask].mean(axis=0)
        centered[mask] = embeddings[mask] - class_means[c]

    covariance = (centered.T @ centered) / len(embeddings)
    covariance = covariance + np.eye(n_features) * 1e-6
    covariance_cholesky = np.linalg.cholesky(covariance)

    return class_means, covariance_cholesky


def mahalanobis_distance(x, class_means, covariance_cholesky, predicted_class=None):
    '''Inference-time Mahalanobis distance against `fit_class_statistics`'s
    offline statistics — `D_M(x) = sqrt((x - mu)^T Sigma^-1 (x - mu))`,
    computed by forward-solving the Cholesky factor rather than inverting
    `Sigma` explicitly.

    Input
    ----------
        x: `(C,)` (single embedding) or `(N, C)` (batch) array.
        class_means, covariance_cholesky: `fit_class_statistics`'s own output.
        predicted_class: `None` (default) — distance to the **closest**
            class (Lee et al. 2018's own "Option B", `min` over all
            classes). A scalar int, or an `(N,)` array of per-example
            class indices — distance to that **specific** ("Option A")
            class instead, e.g. the model's own argmax prediction for each
            example. Both are real options from the paper; neither is
            silently chosen for the caller.

    Output
    ----------
        `(N,)` array (or a scalar if `x` was a single `(C,)` vector) —
        higher values indicate a more out-of-distribution input.
    '''
    x = np.asarray(x, dtype=np.float64)
    single = x.ndim == 1
    x = np.atleast_2d(x)
    n_classes = class_means.shape[0]

    if predicted_class is not None:
        predicted_class = np.broadcast_to(np.asarray(predicted_class), (x.shape[0],))
        diffs = x - class_means[predicted_class]
        z = solve_triangular(covariance_cholesky, diffs.T, lower=True).T
        distances = np.sqrt(np.sum(z ** 2, axis=-1))
    else:
        all_distances = np.zeros((x.shape[0], n_classes))
        for c in range(n_classes):
            diffs = x - class_means[c]
            z = solve_triangular(covariance_cholesky, diffs.T, lower=True).T
            all_distances[:, c] = np.sqrt(np.sum(z ** 2, axis=-1))
        distances = all_distances.min(axis=-1)

    return distances[0] if single else distances


def dense_mahalanobis_distance(feature_map, class_means, covariance_cholesky, predicted_class_map=None):
    '''Segmentation variant of `mahalanobis_distance` — treats every
    spatial location of a dense feature map as its own embedding against
    the same shared/tied covariance, producing a per-pixel distance map
    (feed to `aggregate_anomaly_map` for an image-level decision, per
    Lee et al. 2018's own segmentation-adjacent recommendation of not
    trusting a raw per-pixel average, which gets washed out by background).

    Input
    ----------
        feature_map: `(H, W, C)` array — one image's dense feature map
                     (e.g. a segmentation decoder's pre-classification-head
                     tensor).
        class_means, covariance_cholesky: `fit_class_statistics`'s own output.
        predicted_class_map: `None` (distance to closest class, per pixel),
            or an `(H, W)` array of per-pixel predicted class indices
            (distance to that pixel's own predicted class instead) —
            same `predicted_class` semantics as `mahalanobis_distance`.

    Output
    ----------
        `(H, W)` distance map.
    '''
    height, width, channels = feature_map.shape
    flat = feature_map.reshape(height * width, channels)
    predicted_flat = None if predicted_class_map is None else np.asarray(predicted_class_map).reshape(-1)

    distances = mahalanobis_distance(flat, class_means, covariance_cholesky, predicted_class=predicted_flat)
    return distances.reshape(height, width)


def aggregate_anomaly_map(score_map, threshold, min_blob_fraction=0.01):
    '''Turns a dense per-pixel anomaly score map (either `layer_utils.
    energy_score`'s dense output or this module's own
    `dense_mahalanobis_distance`) into a single flagged/not-flagged
    decision, via connected-component spatial filtering — the practice the
    user's own reference document recommends: naturally-high-score pixels
    around object boundaries are common and don't by themselves indicate
    an anomaly, so only a **contiguous** blob of above-threshold pixels
    exceeding `min_blob_fraction` of the image triggers a flag, not any
    above-threshold pixel in isolation. Convention: **higher score means
    more anomalous** for both supported score types (Energy Score's own
    formula and Mahalanobis distance both increase with distributional
    unfamiliarity — verified directly from the Energy Score formula's own
    worked example: a confident, in-distribution input scores far more
    negative than an unfamiliar one).

    Input
    ----------
        score_map: `(H, W)` array.
        threshold: score above which a pixel is considered anomalous — see
                   `calibrate_threshold`.
        min_blob_fraction: minimum contiguous-component size, as a
                           fraction of `score_map`'s total pixel count, to
                           trigger a flag (`0.01`, i.e. 1% of the image —
                           the reference document's own suggested starting
                           point).

    Output
    ----------
        `(flagged, mask)` — `flagged` is a bool (whether any qualifying
        blob was found), `mask` is an `(H, W)` boolean array, `True` only
        for pixels belonging to a qualifying (large-enough) blob — smaller
        above-threshold blobs are dropped from the mask entirely, not just
        excluded from the `flagged` decision.
    '''
    score_map = np.asarray(score_map)
    binary = score_map > threshold
    labeled, n_components = label(binary)
    min_pixels = min_blob_fraction * score_map.size

    mask = np.zeros_like(binary)
    flagged = False
    for component_id in range(1, n_components + 1):
        component_mask = labeled == component_id
        if component_mask.sum() >= min_pixels:
            mask |= component_mask
            flagged = True

    return flagged, mask


def calibrate_threshold(scores, percentile=95.0):
    '''A validation-set score distribution -> a single threshold, via
    percentile (`95.0`, the reference document's own suggested default) —
    a one-line wrapper around `np.percentile`, included as a named,
    discoverable function (matching this module's other small, composable
    pieces) rather than leaving every caller to independently land on the
    same percentile convention.

    Input
    ----------
        scores: array of scores from a validation set (Energy Score or
                Mahalanobis distance, on correctly-behaving/in-distribution
                examples).
        percentile: `0-100`.

    Output
    ----------
        A scalar float threshold — pass to `aggregate_anomaly_map` or
        compare a global (non-dense) score against directly.
    '''
    return float(np.percentile(np.asarray(scores), percentile))
