Introduction
============

A library to help with the development of AI models with Keras, with a focus on edge / IoT applications. 

Motivation
**********

Building CNN experiments with flexible and scalable augmentation pipelines, with the ability to cobine different encoder and decoder architectures, 
can require a huge initial effort. 
This is even more true in the case of edge-AI models, where models need to be build taking into account more severe constraints and quantization is usually required.

This library wants you to focus on dataset, model architecture and hyperparameters tuning, without worring about the rest.

It provides several helper classes which help in the development of CNN based AI models for edge IoT applications, where resources are limited and model quantization is reccomended.

Features
********

The main features of the library are the following:

1. Flexible, efficient and scalable Dataset management with augmentation pipelines leveraging albumentations (https://albumentations.ai/) and td-data (https://www.tensorflow.org/guide/data)
2. Training driver with builtin callbacks, configurable backbone unfreezing, and quantized model generation
3. Helper classes to easiliy combine pretrained backbones for Edge AI applications with the desired segmentation and classification architectures -- beyond the standard ``tf.keras.applications`` backbones (VGG, ResNet, DenseNet, EfficientNet, MobileNet, ...), also includes 4 modern edge-oriented backbone families built from scratch for this library: **MobileOne**, **MobileNetV4** (incl. the Hybrid attention variants), **FastViT**, and **RepViT** (all published sizes each)
4. Additional loss functions and optimizers which are not part of the Keras distribution, as for now -- including Lovász-Softmax, boundary loss, structure loss (PraNet), log-cosh Dice, OHEM, Quality Focal / Distribution Focal / VariFocal loss, CIoU/SIoU box regression, Lion, and a SAM (Sharpness-Aware Minimization) optimizer wrapper
5. Lightweight semantic segmentation decoders -- **LR-ASPP**, **PP-LiteSeg** (with UAFM), and **DDRNet** (with DAPPM) -- alongside the existing DeepLabV3+ family, all pluggable with any registered backbone
6. A lightweight anchor-free object detection head (**FCOS-lite**, with a PAFPN-lite neck and ATSS label assignment) and additional classification heads (multi-label, CORAL-style ordinal) with optional GeM/ECA/CoordAtt attention pooling

.. note::
   This list is kept in sync with the repository's ``README.md`` feature
   list by hand (plan.md §10/§11 flagged the two as near-duplicates that
   had drifted apart once already) -- update both together.

