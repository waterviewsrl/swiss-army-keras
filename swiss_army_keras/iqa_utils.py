'''Training-pipeline utilities for the IQA (Image Quality Assessment)
side-head (plan.md §13.7, Phase 11) — `swiss_army_keras.layer_utils.
iqa_regression_head`'s own training-data companion, deliberately kept
separate from `dataset_utils.py` rather than modifying it.

**A real constraint on what a plugged-in teacher function receives, worth
being explicit about rather than assuming away**: the user's own framing
was "run a teacher IQA model at the end of the image aug pipeline". In
`dataset_utils.py`'s `SegmentationAlbumentationsDataLoader`/
`ClassificationAlbumentationsDataLoader`, augmentation (`.map(self.
augment_data, ...)`) runs, *then* normalization (`.map(self.normalize,
...)`) — and `build_datasets()` (the method a training script actually
calls) already returns datasets with *both* applied, i.e. already
backbone-normalized, not raw `[0, 255]`. `add_teacher_iqa_labels` below is
applied to whatever `tf.data.Dataset` you hand it, wherever that is in
your own pipeline — if you call it on `build_datasets()`'s own output
(the common case), your `teacher_fn` receives **already-normalized**
images, and needs to expect that convention (or invert it itself). For
exact "after augment, before normalize" placement instead, build your own
pipeline calling the loader's own already-public `augment_data`/
`normalize` instance methods directly rather than the one-shot
`build_datasets()` convenience wrapper — an existing capability of those
classes, not new plumbing added here.

No teacher model is shipped — `teacher_fn` is any
`callable(image) -> scalar quality score in [0, 1]` you provide (a ported
NIMA/MUSIQ checkpoint, a classical no-reference metric, or your own).
`laplacian_variance_quality` below is a small, dependency-free reference
implementation of that contract, meant for testing/demonstration — not a
serious IQA model.
'''

from __future__ import absolute_import

import tensorflow as tf


def add_teacher_iqa_labels(dataset, teacher_fn, main_label_key='main', iqa_label_key='iqa'):
    '''Wraps a `(image, label)`-yielding `tf.data.Dataset` into
    `(image, {main_label_key: label, iqa_label_key: score})`, where `score`
    is `teacher_fn(image)` computed on the fly — the label shape
    `model.compile(loss={main_label_key: ..., iqa_label_key: ...})` needs
    to train a model built with `iqa_head=True`
    (`swiss_army_keras._model_classifier.classifier`/
    `swiss_army_keras._model_lr_aspp.lr_aspp`) against both its task
    output and its IQA side-head simultaneously.

    Runs `teacher_fn` via `tf.py_function` (not this repo's usual
    `tf.numpy_function`, e.g. `losses.boundary_loss`) — a deliberate
    choice: a teacher is plausibly a real Python callable or a Keras model
    itself, not pure numpy array math, and `tf.py_function` (unlike
    `tf.numpy_function`) supports that.

    Per-example, not batched, in this first pass — simpler and safe as a
    default. Calling `teacher_fn` once per *batch* instead (for real
    throughput on an actual neural teacher model) is a real, worthwhile
    follow-up left to a training script that batches *before* calling this
    function, not solved here.

    Input
    ----------
        dataset: a `tf.data.Dataset` yielding `(image, label)` pairs (e.g.
                 `dataset_utils.py`'s own loader output — see this module's
                 own docstring for what normalization state `image` is in
                 by the time it reaches here).
        teacher_fn: `callable(image) -> scalar float` (a Python function,
                   not required to be a pure TF graph op — this is exactly
                   why `tf.py_function` is used).
        main_label_key, iqa_label_key: dict keys for `model.compile(loss=
            {...})` to key against - **must match `model.output_names`
            exactly (a real gotcha, found while testing this function -
            not the `name=` you passed to `classifier()`/`lr_aspp()`
            itself)**. Keras auto-derives each output's name from its
            *last layer's own name*, not the model-level `name=` kwarg
            those builders take as a prefix - and `quantize_apply` prefixes
            that again with `quant_`. Build the model first, then read
            `main_label_key, iqa_label_key = model.output_names` (a plain
            2-tuple for these two-output models) and pass those exact
            strings here and to `compile(loss={...})` - don't guess them
            from the `name=` argument you passed in.

    Output
    ----------
        A new `tf.data.Dataset` yielding `(image, {main_label_key: label,
        iqa_label_key: score})`.
    '''
    def _add_score(image, label):
        score = tf.py_function(func=teacher_fn, inp=[image], Tout=tf.float32)
        score.set_shape([])
        return image, {main_label_key: label, iqa_label_key: score}

    return dataset.map(_add_score)


def laplacian_variance_quality(image, blur_threshold=100.0):
    '''A classical, dependency-free no-reference blur heuristic — the
    variance of the image's Laplacian (a well-known blur indicator: a
    sharp image has strong, varied high-frequency edge response; a blurred
    one doesn't), squashed to `[0, 1]` via a sigmoid centered on
    `blur_threshold`.

    **A toy/demonstration implementation of `add_teacher_iqa_labels`'s own
    `callable(image) -> [0, 1]` contract, not a serious IQA model** — real
    no-reference IQA (NIMA, MUSIQ) captures far more than blur (exposure,
    noise, compression artifacts, aesthetic quality); this function exists
    so the contract has at least one concrete, runnable example, not as a
    recommended production teacher.

    Input
    ----------
        image: a single `(H, W, C)` image tensor, any float range (only
               used for a grayscale conversion + a 3x3 Laplacian
               convolution, insensitive to the exact normalization
               convention - unlike a real IQA model, which likely *is*
               convention-sensitive, see this module's own docstring).
        blur_threshold: the Laplacian-variance value mapped to a `0.5`
                        score - values well above it approach `1.0`
                        (sharp), well below approach `0.0` (blurred).

    Output
    ----------
        A scalar `[0, 1]` quality score.
    '''
    image = tf.convert_to_tensor(image)
    if image.dtype != tf.float32:
        image = tf.cast(image, tf.float32)

    gray = tf.image.rgb_to_grayscale(image) if image.shape[-1] == 3 else image
    gray = gray[tf.newaxis, ...]

    laplacian_kernel = tf.constant(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=tf.float32,
    )[:, :, tf.newaxis, tf.newaxis]

    # `'VALID'`, not `'SAME'`: `'SAME'` zero-pads the border, and a
    # zero-padded neighbor looks like an edge to the Laplacian kernel even
    # for a perfectly flat image - inflating the variance with border
    # artifacts that have nothing to do with actual sharpness (confirmed
    # empirically: a uniform, zero-Laplacian-everywhere test image gave a
    # nonzero variance under `'SAME'`). `'VALID'` simply skips border
    # pixels lacking a full 3x3 neighborhood instead.
    edges = tf.nn.conv2d(gray, laplacian_kernel, strides=1, padding='VALID')
    variance = tf.math.reduce_variance(edges)

    # Log-ratio, not a raw linear `(variance - threshold) / threshold` -
    # Laplacian variance spans several orders of magnitude between
    # low-detail and high-contrast/high-frequency images (confirmed
    # empirically: a real difference of ~500x between a blurred and a
    # sharp synthetic test image), which saturates a linearly-scaled
    # sigmoid to 0 or 1 almost everywhere; the log-ratio compresses that
    # range sensibly instead.
    log_ratio = tf.math.log(variance + 1.0) - tf.math.log(blur_threshold + 1.0)
    return tf.sigmoid(log_ratio)
