import os as _os

# tensorflow_model_optimization's QAT primitives (quantize_annotate_layer /
# quantize_apply, used throughout this package's decoders for the QAT
# support added alongside this shim) only understand legacy Keras 2 layers,
# not Keras 3 (the default backing `tf.keras` since TF 2.16). Force TF to
# expose the legacy Keras 2 API via `tf.keras` unless the environment already
# set this explicitly. This only has an effect if it runs before `tensorflow`
# is imported anywhere in the process — if a caller does `import tensorflow`
# before `import swiss_army_keras` (or any of its submodules), this has no
# effect and Keras 3 stays active, which breaks QAT.
_os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

__version__ = '0.11.0'
