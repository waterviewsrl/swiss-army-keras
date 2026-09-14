from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='swiss_army_keras',
      version='0.11.0',
      description='A collection of models and utilities for the development of edge deployable Keras models',
      long_description=long_description,
      keywords=['keras', 'segmentation', 'classification', 'finetuning', 'edge',
                'quantization', 'augmentation'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.6',
          'Topic :: Multimedia :: Graphics'],
      url='http://github.com/desmoteo/swiss-army-keras',
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
          'efficientnet-lite-keras @ git+https://github.com/waterviewsrl/efficientnet-lite-keras.git',
          'irondomo',
          'pillow',
          'typeguard',
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
      zip_safe=False)
