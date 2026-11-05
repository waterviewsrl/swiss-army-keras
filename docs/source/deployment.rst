Deployment
==========

How a trained model becomes a file that runs on an edge device, what the
library does for you, and what it does not do yet.

From training to TFLite
***********************

Every head built by this library (except the inherited U-Net family) is
wrapped for **quantization-aware training (QAT)**: its layers are annotated
with ``tensorflow_model_optimization`` and ``quantize_apply`` is called before
the model is returned. The backbone itself is not QAT-wrapped. Training
therefore already simulates 8-bit arithmetic in the head, and no separate QAT
step is needed.

``TrainingDriver.run()`` ends by calling ``Quantizer``, which writes:

* ``<name>_quant_ui8.tflite``: full-integer model. Only
  ``TFLITE_BUILTINS_INT8`` operators are allowed, so conversion fails if any
  operator cannot be quantized. Input and output tensors are ``uint8``.
* ``<name>_quant_f16.tflite``: float16 weights with float32 input and output,
  for GPU delegates and as an accuracy reference.

``Quantizer`` can also be used on its own:

.. code-block:: python

   from swiss_army_keras.quantization_utils import Quantizer

   q = Quantizer(test_set, model, "my_model", batches=4)
   q.quantize()
   q.vizualize_ui8_results(4)      # segmentation models

Mixed precision
---------------

``tf.keras.mixed_precision.set_global_policy('mixed_float16')`` can be used
to speed up training on GPU: backbones follow the policy, while QAT-wrapped
layers are forced to float32 (the fake-quantization operator requires it).

Input handling
**************

A model expects its input already normalized by ``model.preprocessing``. On a
device it is usually more convenient to feed raw camera frames:

.. code-block:: python

   from swiss_army_keras.layer_utils import uint8_input_wrapper

   deploy_model = uint8_input_wrapper(model)   # uint8 (H, W, 3) in, normalization in the graph

Structural reparameterization
*****************************

MobileOne, FastViT and RepViT train with several parallel branches per block
and are meant to be deployed with the branches fused into single
convolutions. The fusion functions operate on a **backbone model built
directly from its constructor**:

.. code-block:: python

   from swiss_army_keras._backbone_mobileone import MobileOne_S0, reparameterize_mobileone

   train_model = MobileOne_S0(include_top=False, weights=None, input_shape=(128, 128, 3))
   # ... train ...
   deploy_model = reparameterize_mobileone(train_model)

The effect on a MobileOne-S0 feature extractor at 128 x 128, converted to
per-tensor full-integer TFLite (measured with this repository, random
weights; outputs of the two Keras models agree to 4e-4):

================== ============== ============== ===================================
Model              TFLite ops     File size      Operators
================== ============== ============== ===================================
training-time      416            4.2 MB         183 ADD, 105 DEPTHWISE_CONV_2D,
                                                 86 CONV_2D, 35 MUL, 5 PAD
reparameterized    50             1.0 MB         22 CONV_2D, 21 DEPTHWISE_CONV_2D,
                                                 5 PAD
================== ============== ============== ===================================

.. warning::
   Reparameterization is **not applied automatically**. A decoder or
   classifier built with ``backbone='MobileOne-S0'`` and exported through
   ``TrainingDriver`` / ``Quantizer`` contains the training-time branches,
   and there is no helper yet that fuses a backbone nested inside a full
   model. Until that exists, such exports are several times larger and slower
   than they need to be. This is the first item of ``ROADMAP.md``.

Per-tensor quantization
***********************

TFLite quantizes convolution weights per output channel by default. Some
accelerators require or strongly prefer one scale per tensor. ``Quantizer``
does not expose this yet; convert manually:

.. code-block:: python

   import tensorflow as tf

   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   converter.representative_dataset = representative_data_gen   # yields [float32 batch of 1]
   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
   converter.inference_input_type = tf.uint8
   converter.inference_output_type = tf.uint8
   converter._experimental_disable_per_channel = True           # per-tensor weights
   tflite_bytes = converter.convert()

Per-tensor quantization is harsher than per-channel, especially on depthwise
convolutions and on reparameterized blocks whose fused kernels have very
different ranges per channel. Always compare the accuracy of the per-tensor
model against the float model before deploying.

Inspecting an exported model
****************************

.. code-block:: python

   tf.lite.experimental.Analyzer.analyze(model_content=tflite_bytes)

prints every operator of the flatbuffer. Compare the list with what the
target accelerator supports: anything unsupported runs on the CPU, usually
with a large latency penalty.

Operators produced by typical models of this library (128 x 128 input,
per-tensor full-integer export, random weights):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model
     - Operators beyond CONV_2D / DEPTHWISE_CONV_2D / ADD / CONCATENATION
   * - ``lr_aspp`` + MobileOne-S0
     - RESIZE_BILINEAR, AVERAGE_POOL_2D, LOGISTIC, MUL, PAD, and two
       DEQUANTIZE / QUANTIZE islands
   * - ``pp_liteseg`` + MobileOne-S0
     - RESIZE_BILINEAR, AVERAGE_POOL_2D, MEAN, REDUCE_MAX, LOGISTIC, MUL,
       SUB, PAD
   * - ``deeplab_v3_plus`` + MobileNetV2
     - RESIZE_BILINEAR, AVERAGE_POOL_2D, MUL, and SPACE_TO_BATCH_ND /
       BATCH_TO_SPACE_ND pairs from the dilated convolutions, one DEQUANTIZE
   * - ``classifier`` + MobileNetV4-Conv-S
     - MEAN, FULLY_CONNECTED, LOGISTIC, MUL, SOFTMAX, PAD
   * - ``classifier`` + RepViT-M0.9
     - GELU (27), LOGISTIC and MEAN (squeeze-excite, 11 each), MUL,
       FULLY_CONNECTED, SOFTMAX, PAD

Target notes
************

The library's reference target is the Axis camera DLPU family, reached
through ``.tflite`` files; other aarch64 runtimes consume the same files or
an ONNX conversion of them.

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Target
     - Format
     - What to keep in mind
   * - Axis ARTPEC-8 DLPU
     - int8 / uint8 ``.tflite``
     - Per-tensor quantization (Axis: "performs much better than
       per-channel"). Axis recommends regular convolutions over depthwise,
       3 x 3 kernels, stride 2, filter counts in multiples of 6, and ReLU
       over leaky ReLU.
   * - Axis ARTPEC-9 DLPU
     - int8 / uint8 ``.tflite``
     - Per-channel weights accepted for a limited set of operators. Axis
       recommends ReLU6, channel counts in multiples of 16, no explicit
       padding, batch size 1, tensors of at most 4 dimensions. The published
       operator list is restrictive (for example element-wise MUL only by a
       constant, global mean only on 7 x 7 or 8 x 8 maps, no HARD_SWISH, GELU
       or SOFTMAX), so squeeze-excite style gates and GELU backbones are
       likely to fall back to the CPU.
   * - Axis ARTPEC-7
     - int8 ``.tflite`` compiled for the Edge TPU
     - Needs the Edge TPU compiler after export.
   * - Axis CV25 / CV75
     - ``.tflite`` or ``.onnx`` converted to a native binary
     - Requires the Ambarella toolchain, which is not public.
   * - aarch64 CPU (LiteRT / XNNPACK)
     - any ``.tflite``
     - Per-channel int8 is the best default; float16 for GPU delegates.
   * - NVIDIA Jetson (TensorRT)
     - ONNX
     - Not covered by this library yet: convert the Keras model with
       ``tf2onnx`` and build an FP16 or INT8 engine.

The Axis statements are taken from the Axis developer documentation
(optimization tips and supported-operator pages for each chip) as read in
November 2026; check the live pages for your AXIS OS version, and always
confirm on the device with ``larod-client``. ``EDGE_AI_CANDIDATES.md`` in the
repository discusses these targets, and what the library should add for
them, in depth.

Known limitations
*****************

* No automatic reparameterization at export (see the warning above).
* ``Quantizer`` always produces per-channel weights and ``uint8`` I/O.
* Dilated convolutions (DeepLabV3+ ASPP) export as SPACE_TO_BATCH_ND /
  BATCH_TO_SPACE_ND, which accelerators rarely support.
* Some exports contain DEQUANTIZE / QUANTIZE islands, that is, a few
  operators left in float; they are not yet tracked down layer by layer.
* The detectors run box decoding and NMS inside the graph; on an accelerator
  it is usually better to export the ``*_base`` model and decode on the CPU.
* ``tf.lite.Interpreter`` is deprecated upstream in favour of the LiteRT
  package (``ai_edge_litert``); the library still uses the TensorFlow one.
* A known issue affects QAT-wrapped output layers without activation on a
  randomly initialized ``MobileNetV2`` backbone (all-zero outputs); it does
  not appear with pretrained weights or with the library's own backbones.

Export tests
************

``pytest -m slow`` converts QAT-wrapped decoders to full-integer TFLite with
``TFLITE_BUILTINS_INT8`` only and runs them in the interpreter. Add a case to
``tests/test_tflite_export.py`` for every new head or backbone family.
