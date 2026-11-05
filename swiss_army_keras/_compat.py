'''Runtime compatibility guards.

`swiss_army_keras/__init__.py` sets `TF_USE_LEGACY_KERAS=1` so that `tf.keras`
resolves to the legacy Keras 2 API (`tf-keras`), which is the only API
`tensorflow_model_optimization`'s QAT layers understand. That environment
variable is read once, when TensorFlow is first imported - so if a caller
imports `tensorflow` *before* `swiss_army_keras`, Keras 3 stays active and
every QAT-wrapped decoder in this package fails later with an opaque
`ValueError: to_annotate can only be a keras.layers.Layer ...` from tfmot.
`ensure_legacy_keras()` turns that into an immediate, explicit error at
import time instead (ASSESSMENT.md C3).
'''

import os


def ensure_legacy_keras():
    import tensorflow as tf

    keras_version = str(getattr(tf.keras, '__version__', ''))
    if not keras_version.startswith('3'):
        return
    raise ImportError(
        "swiss_army_keras requires the legacy Keras 2 API (`tf.keras` is Keras "
        "{} here) because tensorflow_model_optimization's quantization-aware "
        "training layers do not support Keras 3. Fix: `import swiss_army_keras` "
        "before `import tensorflow` (the package sets TF_USE_LEGACY_KERAS=1 "
        "itself), or set TF_USE_LEGACY_KERAS=1 in the environment before "
        "starting Python. Current value: TF_USE_LEGACY_KERAS={!r}; `tf-keras` "
        "must also be installed.".format(keras_version, os.environ.get('TF_USE_LEGACY_KERAS'))
    )
