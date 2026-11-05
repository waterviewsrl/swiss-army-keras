Getting started
===============

Installation
************

Requirements: Python 3.9 or newer and TensorFlow 2.16 or newer (CPU or GPU).

.. code-block:: bash

   pip install swiss-army-keras                       # from PyPI
   pip install -e .                                   # from a checkout
   pip install "swiss-army-keras[efficientnet-lite]"  # optional EfficientNet-Lite B0-B4 backbones (Python <= 3.11)

.. _import-order:

Import order
************

The library runs on the legacy Keras 2 API (the ``tf-keras`` package,
installed as a dependency), because the quantization-aware-training layers of
``tensorflow_model_optimization`` do not support Keras 3. The switch is an
environment variable that TensorFlow reads when it is first imported, so:

.. code-block:: python

   import swiss_army_keras   # sets TF_USE_LEGACY_KERAS=1
   import tensorflow as tf   # must come after

If TensorFlow was imported first, importing any model module of the library
raises an ``ImportError`` that explains the fix. Setting
``TF_USE_LEGACY_KERAS=1`` in the environment before starting Python works as
well.

Segmentation in five steps
**************************

1. **Build a model.** Every builder takes an input tensor and returns a
   ``tf.keras.Model``. The model carries a ``preprocessing`` attribute with the
   input normalization its backbone expects.

   .. code-block:: python

      import swiss_army_keras
      import tensorflow as tf
      from swiss_army_keras.models import lr_aspp

      IMG_SIZE, N_LABELS = 256, 3
      input_tensor = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 3))
      model = lr_aspp(
          input_tensor,
          n_labels=N_LABELS,
          backbone="MobileOne-S1",   # any name from the model zoo
          weights=None,              # or the path of a converted weights file
          batch_norm=True,
          freeze_backbone=True,      # train the decoder only at first
      )

2. **Build the datasets.** The loader expects ``images/`` and a mask
   directory with one PNG of class ids per image.

   .. code-block:: python

      import albumentations as albu
      from swiss_army_keras.dataset_utils import SegmentationAlbumentationsDataLoader

      loader = SegmentationAlbumentationsDataLoader(
          dataset_path="/data/pets",
          masks_dir="annotations/trimaps",
          num_classes=N_LABELS,
          label_shift=1,                 # class ids in the files start at 1
          width=IMG_SIZE, height=IMG_SIZE,
          batch_size=16,
          train_augmentations=albu.Compose([albu.HorizontalFlip(p=0.5)]),
          normalization=model.preprocessing,
          seed=0,                        # reproducible train/val/test split
      )
      train_set, val_set, test_set = loader.build_datasets()

3. **Choose loss, optimizer and metrics.** The segmentation models return raw
   logits, so wrap probability-based losses with ``loss_adapter``.

   .. code-block:: python

      from swiss_army_keras import losses, metrics
      from swiss_army_keras.optimizers import AdaBeliefOptimizer

      loss = losses.loss_adapter(losses.symmetrical_unified_focal_loss(), from_logits=True)
      model_metrics = [tf.keras.metrics.CategoricalAccuracy(), metrics.MeanIOU(num_classes=N_LABELS)]

4. **Train.** ``TrainingDriver`` runs a frozen-backbone phase, an optional
   unfrozen phase, and the export. Pass ``export_profile="artpec8"`` (or
   another profile, see :doc:`deployment`) to export for a specific target.

   .. code-block:: python

      from swiss_army_keras.training_utils import TrainingDriver

      TrainingDriver(
          model, "pets_mobileone_lraspp",
          AdaBeliefOptimizer(learning_rate=1e-3), loss, model_metrics,
          train_set, val_set, test_set,
          epochs=20, unfreezed_epochs=20, patience=10,
      ).run()

5. **Collect the outputs**, written to the current directory:

   ============================================ =========================================
   File                                         Content
   ============================================ =========================================
   ``<name>_<timestamp>.h5``                    best full model (Keras HDF5)
   ``<name>_<timestamp>.weights.*``             best weights checkpoint
   ``<name>_<timestamp>_quant_ui8.tflite``      full-integer TFLite model, uint8 I/O
   ``<name>_<timestamp>_quant_f16.tflite``      float16 TFLite model
   ``<name>_<timestamp>_{loss,acc,iou}_*.pdf``  training curves
   ``tblogs/logs_<name>_<timestamp>/``          TensorBoard logs
   ============================================ =========================================

Classification
**************

The same flow with the classification loader (one sub-directory per class)
and a classifier head, which returns softmax probabilities:

.. code-block:: python

   from swiss_army_keras.models import classifier
   from swiss_army_keras.dataset_utils import ClassificationAlbumentationsDataLoader

   input_tensor = tf.keras.layers.Input((224, 224, 3))
   model = classifier(input_tensor, n_classes=5, backbone="MobileNetV4-Conv-S", weights=None)

   loader = ClassificationAlbumentationsDataLoader(
       dataset_path="/data/flower_photos", width=224, height=224, batch_size=32,
       normalization=model.preprocessing, seed=0,
   )
   train_set, val_set, test_set = loader.build_datasets()

Use ``"categorical_crossentropy"`` (labels are one-hot) or one of the focal
losses, then ``TrainingDriver`` as above.

Things to know
**************

* **Fixed input size.** Builders compute upsampling factors from static
  shapes; pass an ``Input`` with a concrete height and width, both multiples
  of 32. PP-LiteSeg needs at least 128 x 128.
* **Square inputs with the data loaders.** The loaders do not treat ``width``
  and ``height`` consistently; use equal values.
* **Feature-pyramid depth differs by backbone.** Most backbones expose five
  taps (strides 2 to 32). RepViT and FastViT expose four (strides 4 to 32):
  pass ``deep_layer=4`` (and ``shallow_layer=1`` or ``2``) to
  ``lr_aspp`` / ``deeplab_v3_plus`` / the classifiers, and
  ``feature_levels=(2, 3, 4)`` to ``pp_liteseg`` and the detectors.
  StrideFormer exposes three (strides 8 to 32): ``feature_levels=(1, 2, 3)``,
  ``deep_layer=3``.
* **No heavy defaults.** ``weights='imagenet'`` downloads weights for the
  ``tf.keras.applications`` backbones only; see :ref:`pretrained-weights` for
  the others.
* **Worker processes.** Each data loader starts ``cpu_count() / 2``
  augmentation processes that live until the interpreter exits; create one
  loader per experiment, not one per cell re-run.

Running the tests
*****************

.. code-block:: bash

   pip install -r requirements-test.txt
   pytest -m "not slow"     # about 15-20 minutes on a laptop CPU
   pytest -m slow           # TFLite full-integer export checks

Do not edit files under ``swiss_army_keras/`` while the suite is running:
Keras re-reads a layer's source when it first traces it, and a concurrent
edit produces spurious failures.
