from setuptools import setup, find_packages
from codecs import open
from os import path
import re

here = path.abspath(path.dirname(__file__))

# Single source of truth for the version: swiss_army_keras/__init__.py
# (`swiss_army_keras.__version__`), read textually so setup.py does not have
# to import the package (and therefore TensorFlow) at build time.
with open(path.join(here, 'swiss_army_keras', '__init__.py'), encoding='utf-8') as f:
    _version = re.search(r"^__version__ = '([^']+)'", f.read(), re.M).group(1)

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='swiss_army_keras',
      version=_version,
      description='A collection of models and utilities for the development of edge deployable Keras models',
      long_description=long_description,
      keywords=['keras', 'segmentation', 'classification', 'finetuning', 'edge',
                'quantization', 'augmentation'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'],
      url='https://github.com/waterviewsrl/swiss-army-keras',
      # tensorflow>=2.16 itself needs >=3.9; dataclasses/f-strings rule out 3.6.
      python_requires='>=3.9',
      author='Matteo Ferrabone',
      author_email='matteo.ferrabone@gmail.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      package_data={
          # If any package contains *.txt files, include them:
          "": ["*.ttf"],
      },
      install_requires=[
          'numpy',
          'scipy',
          'tensorflow>=2.16',
          # QAT (tensorflow_model_optimization) only supports legacy Keras 2
          # layers; swiss_army_keras/__init__.py forces `tf.keras` back to
          # this legacy API via TF_USE_LEGACY_KERAS. Listed explicitly even
          # though tensorflow_model_optimization also pulls it in, since this
          # package depends on it directly, not just transitively.
          'tf-keras>=2.16',
          # `_backbone_zoo.py` imports it unconditionally (the `hub://`
          # EfficientNet-Lite feature-vector backbones) - was never declared
          # (ASSESSMENT.md C1).
          'tensorflow_hub',
          'irondomo',
          'pillow',
          'tensorflow_model_optimization',
          # Previously used unconditionally (`dataset_utils.py`,
          # `quantization_utils.py`, `training_utils.py`) but never
          # declared here or in requirements.txt - a clean `pip install
          # swiss-army-keras` could not actually import those modules
          # (plan.md §10). Declared explicitly now, alongside every other
          # direct import this package makes.
          'opencv-python',
          'albumentations',
          'matplotlib',
          'scikit-image',
          'tqdm',
          'dill',
          # `TrainingDriver` (training_utils.py) uses a
          # `tf.keras.callbacks.TensorBoard` callback unconditionally -
          # found the same way as the deps above, running `TrainingDriver`
          # for real for the first time (plan.md §10/§13 Phase 8):
          # `tensorflow.python.summary.tb_summary.TBNotInstalledError`.
          'tensorboard',
      ],
      extras_require={
          # EfficientNet-Lite B0-B4 backbones. Kept out of install_requires
          # because (a) PyPI rejects metadata with direct VCS URLs, which made
          # `pip install swiss-army-keras` undeliverable for 0.11.0, and (b)
          # the fork caps Python at <=3.11; `_backbone_zoo.py` degrades
          # gracefully (with a warning) when it is absent (ASSESSMENT.md C1).
          'efficientnet-lite': [
              'efficientnet-lite-keras @ git+https://github.com/waterviewsrl/efficientnet-lite-keras.git',
          ],
          # One-time PyTorch -> Keras weight conversion (scripts/port_*_weights.py).
          'porting': ['torch', 'timm', 'safetensors'],
          'docs': ['sphinx', 'sphinx_rtd_theme', 'nbsphinx'],
          'test': ['pytest'],
      },
      zip_safe=False)
