Introduction
============

``swiss_army_keras`` helps with the development of compact vision models in
Keras, with a focus on edge / IoT deployment. It started as a fork of
`keras-unet-collection <https://github.com/yingkaisha/keras-unet-collection>`_
and has grown into a toolkit that covers the whole path from a folder of
images to a quantized TFLite file.

Motivation
**********

Building experiments with flexible augmentation pipelines, and being able to
combine different encoders and decoders, requires a large initial effort. This
is even more true for edge models, which must respect tight latency and memory
budgets and usually have to be quantized to 8-bit integers.

The library lets you concentrate on the dataset, the architecture choice and
the hyper-parameters, and takes care of the rest.

Features
********

1. **Dataset pipelines.** ``tf.data`` loaders for segmentation and
   classification with `albumentations <https://albumentations.ai/>`_
   augmentation running in a pool of worker processes.
2. **Training driver.** Frozen-backbone training, optional unfrozen
   fine-tuning, built-in callbacks, and TFLite export at the end.
3. **Backbones.** One registry for the ``tf.keras.applications`` families
   (MobileNet, ResNet, DenseNet, EfficientNet, VGG), EfficientNet-Lite, and
   five edge-oriented families implemented in this library: MobileOne,
   MobileNetV4, FastViT, RepViT and StrideFormer.
4. **Segmentation.** DeepLabV3+, LR-ASPP, PP-LiteSeg, DDRNet, PIDNet and
   PP-MobileSeg, plus the U-Net family inherited from keras-unet-collection.
5. **Detection and instance segmentation.** Anchor-free FCOS-lite and
   GFL-lite detectors and a YOLACT-style instance-segmentation head.
6. **Classification and regression heads.** Single-label, multi-label,
   ordinal, keypoint regression, re-identification embeddings, monocular
   depth, and an optional image-quality side head.
7. **Losses, optimizers, metrics.** Region, boundary and focal losses for
   imbalanced segmentation, detection losses, depth losses; Lion, SAM,
   AdaBelief and others; boundary F-score and mean average precision.
8. **Deployment utilities.** Target profiles for Axis ARTPEC-8/9 and CPU
   runtimes, full-integer and float16 TFLite export with automatic
   reparameterization of MobileOne / FastViT / RepViT backbones, a check of
   the exported operators against the target, an in-graph uint8 input
   wrapper, out-of-distribution scores and test-time augmentation.

The complete inventory, with registry names and limitations, is in
:doc:`model_zoo`; export details are in :doc:`deployment`.

.. note::
   This short list mirrors the "What it provides" table of the repository's
   ``README.md``. The detailed inventory lives only in :doc:`model_zoo`, so
   there is a single place to update when a component is added.
