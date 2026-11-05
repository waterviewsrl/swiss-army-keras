Deployment
==========

How a trained model becomes a file that runs on an edge device, what the
library does for you, and what it does not do yet.

The export pipeline
*******************

``TrainingDriver.run()`` ends by calling ``Quantizer``, which can also be used
on its own:

.. code-block:: python

   from swiss_army_keras.quantization_utils import Quantizer

   q = Quantizer(test_set, model, "my_model", batches=4, profile="artpec8")
   q.quantize()
   print(q.lint_report)

``quantize()`` performs four steps:

1. **Reparameterize.** The training-time branches of a MobileOne, RepViT or
   FastViT backbone are fused into plain convolutions.
2. **Strip the QAT wrappers** (for integer-only targets, see below).
3. **Convert** for the chosen deployment profile, and write

   * ``<name>_quant_ui8.tflite``: full-integer model. Only
     ``TFLITE_BUILTINS_INT8`` operators are allowed, so conversion fails if
     an operator cannot be quantized. Input and output are ``uint8``.
   * ``<name>_quant_f16.tflite``: float16 weights with float32 input and
     output, for GPU delegates and as an accuracy reference.

4. **Check** the full-integer file against the profile. The report is logged
   and kept in ``q.lint_report``; pass ``strict=True`` to make a failed check
   raise.

With ``TrainingDriver``, choose the target with ``export_profile``:

.. code-block:: python

   TrainingDriver(model, "my_model", optimizer, loss, metrics,
                  train_set, val_set, test_set, epochs=20,
                  export_profile="artpec8").run()

The individual steps are available in ``swiss_army_keras.deploy_utils``:

.. code-block:: python

   from swiss_army_keras import deploy_utils

   export_model = deploy_utils.strip_qat(deploy_utils.reparameterize_model(model))
   tflite_bytes = deploy_utils.convert_to_tflite(export_model, representative_data_gen, profile="artpec8")
   print(deploy_utils.lint_tflite(tflite_bytes, profile="artpec8"))

Deployment profiles
*******************

A profile states how a target wants its model: weight quantization
granularity, input and output type, and, where published, the operators the
accelerator runs.

.. list-table::
   :header-rows: 1
   :widths: 14 14 12 14 46

   * - Profile
     - Weights
     - I/O
     - Integer only
     - Checked against
   * - ``default``
     - per channel
     - uint8
     - no
     - nothing beyond types (the historical behaviour)
   * - ``litert-cpu``
     - per channel
     - uint8
     - no
     - nothing beyond types
   * - ``artpec7``
     - per channel
     - uint8
     - yes
     - nothing; run the Edge TPU compiler on the result
   * - ``artpec8``
     - per tensor
     - uint8
     - yes
     - Axis's ARTPEC-8 operator list; channel counts in multiples of 6
   * - ``artpec9``
     - per channel
     - uint8
     - yes
     - Axis's ARTPEC-9 operator list and three of its per-operator
       constraints; channel counts in multiples of 16

``convert_to_tflite`` and ``lint_tflite`` accept ``per_channel=`` and
``io_dtype=`` overrides, and a custom ``DeploymentProfile`` can be passed
anywhere a profile name is accepted.

.. warning::
   The Axis operator lists were transcribed from the Axis developer
   documentation in November 2026 by an automated reader and have not been
   confirmed on a device. Treat "not on the target operator list" as a
   prompt to test with ``larod-client``, not as proof of a CPU fallback.

Structural reparameterization
*****************************

MobileOne, FastViT and RepViT train with several parallel branches per block
and are meant to be deployed with the branches fused into single
convolutions. ``deploy_utils.reparameterize_model`` does this for a backbone
built directly from its constructor and for any model of the library that
embeds one (decoders, classifiers, detectors, also behind
``uint8_input_wrapper``). Models with other backbones pass through
unchanged. Layers outside the backbone are shared with the original model,
so keep training the original and treat the result as an export artifact.

``Quantizer`` applies it by default (``reparameterize=True``). The effect on
``lr_aspp`` with a MobileOne-S0 backbone at 128 x 128 (random weights,
measured with this repository):

============================================= ============ ===========
Export                                        TFLite ops   File size
============================================= ============ ===========
as built (QAT wrappers, training branches)    432          5.5 MB
fused and stripped, ``artpec8`` profile       61           1.4 MB
============================================= ============ ===========

QAT wrappers and why they are stripped for accelerators
*******************************************************

Every head built by this library (except the inherited U-Net family) is
wrapped for quantization-aware training with
``tensorflow_model_optimization``; the backbone is not. Training therefore
simulates 8-bit arithmetic in the head, which makes its weights robust to
quantization noise.

Exporting the wrapped model directly has two problems on an accelerator that
can only run integer operators:

* **Float operators.** The head's learned fake-quantization ranges and the
  backbone's calibrated ranges can disagree on a tensor they share. The
  converter then leaves an operator in floating point: in LR-ASPP and
  DeepLabV3+ it is the last convolution of the backbone, the most expensive
  one. ``tests/test_tflite_export.py`` keeps a test that documents this.
* **Per-channel weights.** The wrappers quantize convolution weights per
  channel whatever the converter is told, so a per-tensor target such as
  ARTPEC-8 does not get what it asks for.

``deploy_utils.strip_qat`` returns a plain float copy of the model with the
trained weights; converting that copy quantizes the whole graph
post-training from the calibration data, with one consistent set of ranges.
``Quantizer`` does this automatically for profiles marked integer-only
(``strip_qat=None``); pass ``strip_qat=True`` or ``False`` to force either
behaviour. The ``default`` and ``litert-cpu`` profiles keep the wrappers, as
before; a CPU runtime executes the occasional float operator, only more
slowly.

The consequence to be aware of: for accelerator profiles, activation ranges
come from the calibration batches, not from training. Use enough
representative batches (``batches=`` / ``quant_batches=``), and compare the
exported model's accuracy with the float model before deploying.

Checking an exported model
**************************

``deploy_utils.lint_tflite`` checks a flatbuffer against a profile: input
and output types, operators left in floating point, weight quantization
granularity, operators missing from the target's list, per-operator
constraints, and channel counts. Example (the fused LR-ASPP model above,
ARTPEC-8 profile)::

   TFLite lint, profile 'artpec8': 61 operators, ISSUES FOUND
     operators: AVERAGE_POOL_2D x1, CONCATENATION x1, CONV_2D x26, DEPTHWISE_CONV_2D x21, LOGISTIC x1, MUL x1, PAD x5, QUANTIZE x2, RESIZE_BILINEAR x3
     NOT ON THE TARGET OPERATOR LIST (expected to run on the CPU): LOGISTIC x1, PAD x5
       list source: Axis developer documentation, ARTPEC-8 "Supported Ops", read 2026-11-05 (unverified on device)
     warning: 22 convolution(s) have an output channel count that is not a multiple of 6
     note: Axis recommends regular Conv2D over DepthwiseConv2D, 3x3 kernels, stride 2, ReLU over leaky ReLU.

``report.errors`` lists certain problems (the file does not match the
profile), ``report.unsupported_ops`` the operators expected to run on the
CPU, ``report.warnings`` inefficiencies; ``report.ok`` is true when the
first two are empty. ``deploy_utils.list_tflite_ops`` returns just the
operator counts, and ``tf.lite.experimental.Analyzer.analyze`` prints the
full graph.

Operators produced by typical models, before fusion and stripping
(128 x 128 input, per-tensor export, random weights):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model
     - Operators beyond CONV_2D / DEPTHWISE_CONV_2D / ADD / CONCATENATION
   * - ``lr_aspp`` + MobileOne-S0
     - RESIZE_BILINEAR, AVERAGE_POOL_2D, LOGISTIC, MUL, PAD
   * - ``pp_liteseg`` + MobileOne-S0
     - RESIZE_BILINEAR, AVERAGE_POOL_2D, MEAN, REDUCE_MAX, LOGISTIC, MUL,
       SUB, PAD
   * - ``deeplab_v3_plus`` + MobileNetV2
     - RESIZE_BILINEAR, AVERAGE_POOL_2D, MUL, and SPACE_TO_BATCH_ND /
       BATCH_TO_SPACE_ND pairs from the dilated convolutions
   * - ``classifier`` + MobileNetV4-Conv-S
     - MEAN, FULLY_CONNECTED, LOGISTIC, MUL, SOFTMAX, PAD
   * - ``classifier`` + RepViT-M0.9
     - GELU (27), LOGISTIC and MEAN (squeeze-excite, 11 each), MUL,
       FULLY_CONNECTED, SOFTMAX, PAD

Input handling
**************

A model expects its input already normalized by ``model.preprocessing``. On a
device it is usually more convenient to feed raw camera frames:

.. code-block:: python

   from swiss_army_keras.layer_utils import uint8_input_wrapper

   deploy_model = uint8_input_wrapper(model)   # uint8 (H, W, 3) in, normalization in the graph

Mixed precision
***************

``tf.keras.mixed_precision.set_global_policy('mixed_float16')`` can be used
to speed up training on GPU: backbones follow the policy, while QAT-wrapped
layers are forced to float32 (the fake-quantization operator requires it).

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

* For accelerator profiles the QAT ranges are discarded at export (see
  above); per-tensor quantization of depthwise and fused convolutions can
  cost accuracy, and the library has no cross-layer equalization yet.
* Several architectures use operators outside the published Axis lists
  (sigmoid gates, explicit padding, GELU, squeeze-excite, dilated
  convolutions). The lint reports them; it does not rewrite the model.
* The lint checks only part of the published per-operator constraints (for
  example not the pooling window limits of ARTPEC-9).
* The detectors run box decoding and NMS inside the graph; on an accelerator
  it is usually better to export the ``*_base`` model and decode on the CPU.
* ``tf.lite.Interpreter`` is deprecated upstream in favour of the LiteRT
  package (``ai_edge_litert``); the library still uses the TensorFlow one.
* A known issue affects QAT-wrapped output layers without activation on a
  randomly initialized ``MobileNetV2`` backbone (all-zero outputs); it does
  not appear with pretrained weights or with the library's own backbones.

Export tests
************

``tests/test_deploy_utils.py`` covers the profiles, the lint, QAT stripping
and reparameterization on small models. ``pytest -m slow`` runs the full
pipeline on real decoders for the ``artpec8`` and ``artpec9`` profiles and
executes the result in the interpreter. Add a case to
``tests/test_tflite_export.py`` for every new head or backbone family.
