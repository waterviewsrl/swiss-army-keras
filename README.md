# swiss-army-keras

A library to help with the development of AI models with Keras, with a focus on edge / IoT applications. Based originally on https://github.com/yingkaisha/keras-unet-collection (see the [README.md](https://github.com/waterviewsrl/swiss-army-keras/blob/main/README-keras-unet-collection.md))

## Summary

This library wants you to focus on dataset, model architecture and hyperparameters tuning, without worring about the rest. 

It provides several helper classes which help in the development of CNN based AI models for edge IoT applications, where resources are limited and model quantization is reccomended. 

The main features of the library are the following:

1) Flexible, efficient and scalable Dataset management with augmentation pipelines leveraging albumentations (https://albumentations.ai/) and td-data (https://www.tensorflow.org/guide/data)
2) Training driver with builtin callbacks, configurable backbone unfreezing, and quantized model generation
3) Helper classes to easiliy combine pretrained backbones for Edge AI applications with the desired segmentation and classification architectures — beyond the standard `tf.keras.applications` backbones (VGG, ResNet, DenseNet, EfficientNet, MobileNet, ...), also includes 4 modern edge-oriented backbone families built from scratch for this library: **MobileOne**, **MobileNetV4** (incl. the Hybrid attention variants), **FastViT**, and **RepViT** (all published sizes each)
4) Additional loss functions and optimizers which are not part of the Keras distribution, as for now — including Lovász-Softmax, boundary loss, structure loss (PraNet), log-cosh Dice, OHEM, Quality Focal / Distribution Focal / VariFocal loss, CIoU/SIoU box regression, Lion, and a SAM (Sharpness-Aware Minimization) optimizer wrapper
5) Lightweight semantic segmentation decoders — **LR-ASPP**, **PP-LiteSeg** (with UAFM), and **DDRNet** (with DAPPM) — alongside the existing DeepLabV3+ family, all pluggable with any registered backbone
6) A lightweight anchor-free object detection head (**FCOS-lite**, with a PAFPN-lite neck and ATSS label assignment) and additional classification heads (multi-label, CORAL-style ordinal) with optional GeM/ECA/CoordAtt attention pooling 

> This list is kept in sync with `docs/source/intro.rst`'s "Features" section by hand (they were found to have drifted apart once already) — update both together.

## Installation

### Dependencies

Python >= 3.9 and tensorflow >= 2.16 (cpu or gpu). The library runs on the
legacy Keras 2 API (`tf-keras`, installed automatically) because
`tensorflow_model_optimization`'s quantization-aware-training layers do not
support Keras 3 yet; `import swiss_army_keras` **before** `import tensorflow`
so the `TF_USE_LEGACY_KERAS=1` switch takes effect (the library raises a clear
error at import if it did not).

The optional EfficientNet-Lite B0-B4 backbones need the
[waterviewsrl/efficientnet-lite-keras](https://github.com/waterviewsrl/efficientnet-lite-keras)
fork (Python <= 3.11 only):

    pip3 install "swiss-army-keras[efficientnet-lite]"

### Install with pip

You can install directly by pypi with pip:

    pip3 install swiss-army-keras


## Documentation

You can find documentation and examples [here](https://swiss-army-keras.readthedocs.io/en/latest/index.html)

Example jupyter notebooks can also be found in the example folder here.
