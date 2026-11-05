API Reference
=============

Generated from the docstrings. The public entry points are
``swiss_army_keras.models`` (full models) and ``swiss_army_keras.base``
(feature builders without input layer and head); both only re-export the
functions documented below, grouped here by the module that defines them.

Docstrings follow the package's own plain-text convention (a summary, then
``Input`` and ``Output`` sections), rendered as literal blocks.

Backbones
---------

.. automodule:: swiss_army_keras._backbone_zoo
   :members:

.. automodule:: swiss_army_keras._backbone_mobileone
   :members:

.. automodule:: swiss_army_keras._backbone_mobilenetv4
   :members:

.. automodule:: swiss_army_keras._backbone_fastvit
   :members:

.. automodule:: swiss_army_keras._backbone_repvit
   :members:

.. automodule:: swiss_army_keras._backbone_strideformer
   :members:

.. automodule:: swiss_army_keras.reparam_utils
   :members:

Segmentation decoders
---------------------

.. automodule:: swiss_army_keras._model_deeplab_v3_plus
   :members:

.. automodule:: swiss_army_keras._model_lr_aspp
   :members:

.. automodule:: swiss_army_keras._model_pp_liteseg
   :members:

.. automodule:: swiss_army_keras._model_ddrnet
   :members:

.. automodule:: swiss_army_keras._model_pidnet
   :members:

.. automodule:: swiss_army_keras._model_pp_mobileseg
   :members:

Classification, regression and embedding heads
----------------------------------------------

.. automodule:: swiss_army_keras._model_classifier
   :members:

Monocular depth
---------------

.. automodule:: swiss_army_keras._model_depth
   :members:

Several heads on one backbone
-----------------------------

.. automodule:: swiss_army_keras._model_multi_task
   :members: multi_task

Object detection and instance segmentation
------------------------------------------

.. automodule:: swiss_army_keras._model_detection
   :members:

Losses
------

.. automodule:: swiss_army_keras.losses
   :members:

Optimizers
----------

.. automodule:: swiss_army_keras.optimizers
   :members:

Metrics
-------

.. automodule:: swiss_army_keras.metrics
   :members:

Data loading
------------

.. automodule:: swiss_army_keras.dataset_utils
   :members: SegmentationAlbumentationsDataLoader, DepthAlbumentationsDataLoader, ClassificationAlbumentationsDataLoader

Training
--------

.. automodule:: swiss_army_keras.training_utils
   :members:

Export and inference
--------------------

.. automodule:: swiss_army_keras.quantization_utils
   :members:

.. automodule:: swiss_army_keras.deploy_utils
   :members:

.. automodule:: swiss_army_keras.inference_utils
   :members:

Out-of-distribution and image-quality utilities
-----------------------------------------------

.. automodule:: swiss_army_keras.ood_utils
   :members:

.. automodule:: swiss_army_keras.iqa_utils
   :members:

Layers and shared building blocks
---------------------------------

.. automodule:: swiss_army_keras.layer_utils
   :members:

.. automodule:: swiss_army_keras.activations
   :members:

.. automodule:: swiss_army_keras.transformer_layers
   :members:

General utilities
-----------------

.. automodule:: swiss_army_keras.utils
   :members:

U-Net family (inherited from keras-unet-collection)
---------------------------------------------------

.. automodule:: swiss_army_keras._model_unet_2d
   :members:

.. automodule:: swiss_army_keras._model_vnet_2d
   :members:

.. automodule:: swiss_army_keras._model_unet_plus_2d
   :members:

.. automodule:: swiss_army_keras._model_r2_unet_2d
   :members:

.. automodule:: swiss_army_keras._model_att_unet_2d
   :members:

.. automodule:: swiss_army_keras._model_resunet_a_2d
   :members:

.. automodule:: swiss_army_keras._model_u2net_2d
   :members:

.. automodule:: swiss_army_keras._model_unet_3plus_2d
   :members:

.. automodule:: swiss_army_keras._model_transunet_2d
   :members:

.. automodule:: swiss_army_keras._model_swin_unet_2d
   :members:
