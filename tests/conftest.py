"""Must run before any test module (or its imports) touches `tensorflow`.

`swiss_army_keras/__init__.py` sets `TF_USE_LEGACY_KERAS=1` so that
`tensorflow_model_optimization`'s QAT primitives (which only understand
legacy Keras 2 layers) keep working under TF's Keras-3-by-default versions -
but that only has an effect if it runs before TensorFlow's own import-time
backend selection. pytest imports `conftest.py` before collecting/importing
any test module, so setting it here (rather than relying on every test
module importing `swiss_army_keras` before `tensorflow`) is the one place
that's guaranteed to run first regardless of any individual test file's
import order.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# When PyTorch is installed (albumentations imports it if it is there) and
# gets loaded after TensorFlow but before TensorFlow's first convolution,
# float32 convolutions on this stack (TF 2.21, torch 2.14) come out with a
# relative error of about 2e-4 instead of 3e-7. That is harmless for
# training, but it made five tests with tight tolerances depend on whether
# a test file importing `dataset_utils` had been collected first. Loading
# PyTorch first gives the exact results in every order.
try:
    import torch  # noqa: F401
except ImportError:
    pass
