
'''Inference-time (not training-time) helpers — kept in their own module,
not `quantization_utils.py`/`training_utils.py` (both options plan.md §4.3
allowed), since both of those are currently unimportable in a clean
environment (`ModuleNotFoundError: No module named 'matplotlib'` — neither
declares `matplotlib` as a dependency despite importing it unconditionally
at module level; see plan.md §10 for the full finding). This module has no
such baggage: only `tensorflow` is required.
'''

from __future__ import absolute_import

import tensorflow as tf


def tta_predict(model, image, scales=(1.0,), hflip=True, vflip=False):
    '''
    Test-time augmentation (TTA) for segmentation-style models: predicts on
    a set of horizontally/vertically-flipped and/or rescaled copies of
    `image`, undoes each transform on the corresponding output map, and
    averages — a cheap accuracy win for ambiguous/soft boundaries at eval
    time (no training cost, no architecture change), at the cost of
    `len(scales) * (1 + hflip + vflip + (hflip and vflip))` forward passes
    per image.

    tta_predict(model, image, scales=(1.0,), hflip=True, vflip=False)

    Input
    ----------
        model: a `tf.keras.Model` (or anything with a `model(x, training=False)`
               `__call__`) mapping an image to a same-spatial-size output —
               true of every decoder in this repo (`deeplab_v3_plus`/
               `lr_aspp`/`pp_liteseg`/`ddrnet`'s main output all upsample
               back to input resolution internally). For a multi-output
               model (e.g. `ddrnet(..., augment=True)`), pass a wrapper that
               selects the output you want averaged (e.g. `lambda x, **kw:
               model(x, **kw)[0]` for DDRNet's main head) — this helper
               only handles a single output tensor per call.
        image: a single `(H, W, C)` numpy array or tensor (channels-last),
               already preprocessed the way `model` expects (apply
               `model.preprocessing` first, if it has one, before calling
               this).
        scales: resize factors to average over. Default `(1.0,)` — no
                multi-scale averaging, just the flip variants below;
                e.g. `(0.75, 1.0, 1.25)` adds two more scales. **Any scale
                other than `1.0` requires `model` to accept a spatial size
                different from the one it was originally built/called with**
                — i.e. built with `Input((None, None, C))`, not a fixed
                `Input((H, W, C))`, *and* free of any layer that bakes a
                concrete spatial size into itself at build time (a fixed-size
                `AveragePooling2D`/`Conv2D` used as a "global" pool sized to
                the exact deepest-feature-map shape, the pattern every
                decoder in this repo — `deeplab_v3_plus`/`lr_aspp`/
                `pp_liteseg`/`ddrnet`'s DAPPM — currently uses instead of
                `GlobalAveragePooling2D`). None of this repo's own decoders
                satisfy that today, confirmed directly (a `scales != (1.0,)`
                call against any of them raises at the first resized forward
                pass) — multi-scale TTA works, but only against a model
                that's genuinely resolution-agnostic; `hflip`/`vflip` alone
                (`scales=(1.0,)`) work against any of this repo's decoders
                as-is, since a flip never changes spatial size.
        hflip: also average a horizontally-flipped copy at each scale.
        vflip: also average a vertically-flipped copy at each scale (and,
               if `hflip` is also `True`, a copy flipped both ways).

    Output
    ----------
        An `(H, W, n_labels)` numpy array — the averaged prediction, at the
        original input resolution.

    * Flipping and resizing commute (a horizontal/vertical flip doesn't
      change which resize target size is correct, and resizing doesn't
      care what order pixels are in) — so undoing them on the output in
      either order gives the same result; this always un-resizes first,
      then un-flips, for no particular reason beyond consistency.
    '''
    image = tf.convert_to_tensor(image)
    if image.shape.rank == 3:
        image = image[tf.newaxis, ...]

    height, width = int(image.shape[1]), int(image.shape[2])

    flip_variants = [(False, False)]
    if hflip:
        flip_variants.append((True, False))
    if vflip:
        flip_variants.append((False, True))
        if hflip:
            flip_variants.append((True, True))

    accum = None
    n = 0
    for scale in scales:
        if scale != 1.0:
            scaled_h, scaled_w = int(round(height * scale)), int(round(width * scale))
            scaled_image = tf.image.resize(image, (scaled_h, scaled_w))
        else:
            scaled_image = image

        for do_hflip, do_vflip in flip_variants:
            variant = scaled_image
            if do_hflip:
                variant = tf.image.flip_left_right(variant)
            if do_vflip:
                variant = tf.image.flip_up_down(variant)

            pred = model(variant, training=False)

            if scale != 1.0:
                pred = tf.image.resize(pred, (height, width))
            if do_hflip:
                pred = tf.image.flip_left_right(pred)
            if do_vflip:
                pred = tf.image.flip_up_down(pred)

            accum = pred if accum is None else accum + pred
            n += 1

    return (accum / n)[0].numpy()
