API Reference
=============

Autodoc reference for every public module in ``swiss_army_keras`` — added
to close a real, pre-existing gap (plan.md §11): ``sphinx.ext.autodoc`` and
``sphinx.ext.napoleon`` were already configured in ``conf.py``, but no
``.rst`` file anywhere in ``docs/source/`` contained a single
``automodule::``/``autoclass::``/``autofunction::`` directive, so building
the docs produced zero reference documentation for any function, class, or
module — regardless of docstring quality — for the existing backbones/
decoders/losses, let alone the ones added since.

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

.. automodule:: swiss_army_keras.reparam_utils
   :members:

Decoders
--------

.. automodule:: swiss_army_keras._model_deeplab_v3_plus
   :members:

.. automodule:: swiss_army_keras._model_lr_aspp
   :members:

.. automodule:: swiss_army_keras._model_pp_liteseg
   :members:

.. automodule:: swiss_army_keras._model_ddrnet
   :members:

Classification
---------------

.. automodule:: swiss_army_keras._model_classifier
   :members:

Object detection
-----------------

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

Layers and shared building blocks
-----------------------------------

.. automodule:: swiss_army_keras.layer_utils
   :members:

Inference utilities
--------------------

.. automodule:: swiss_army_keras.inference_utils
   :members:

Base (encoder/feature-builder) and full-model entry points
--------------------------------------------------------------

.. automodule:: swiss_army_keras.base
   :members:

.. automodule:: swiss_army_keras.models
   :members:

General utilities
------------------

.. automodule:: swiss_army_keras.utils
   :members:
