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

``quantize()`` performs these steps:

1. **Reparameterize.** The training-time branches of a MobileOne, RepViT or
   FastViT backbone are fused into plain convolutions.
2. **Strip the QAT wrappers** (for integer-only targets, see below).
3. **Equalize** consecutive layers (for per-tensor targets, see below).
4. **Convert** for the chosen deployment profile, and write

   * ``<name>_quant_ui8.tflite``: full-integer model. Only
     ``TFLITE_BUILTINS_INT8`` operators are allowed, so conversion fails if
     an operator cannot be quantized. Input and output are ``uint8``.
   * ``<name>_quant_f16.tflite``: float16 weights with float32 input and
     output, for GPU delegates and as an accuracy reference.

5. **Check** the full-integer file against the profile. The report is logged
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
   deploy_utils.equalize_cross_layer(export_model)        # per-tensor targets
   tflite_bytes = deploy_utils.convert_to_tflite(export_model, representative_data_gen, profile="artpec8")
   print(deploy_utils.lint_tflite(tflite_bytes, profile="artpec8"))
   print(deploy_utils.quantization_error(export_model, tflite_bytes, some_normalized_images))

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

Per-tensor quantization and cross-layer equalization
****************************************************

With one scale per weight tensor (ARTPEC-8), a single channel with large
weights costs every other channel of that tensor its resolution. Depthwise
and fused convolutions are the usual victims.
``deploy_utils.equalize_cross_layer`` implements cross-layer equalization
(Nagel et al., 2019): for two consecutive convolutions separated only by a
ReLU, an output channel of the first is divided by a factor and the matching
input channel of the second multiplied by it, chosen so both layers end up
with the same weight range in that channel. The network computes exactly the
same function; only the distribution of the weights changes.

It applies to chains
``conv -> [batch norm] -> [ReLU / padding] -> conv`` whose intermediate
tensors have a single consumer, inside the model and its nested backbone.
Feature taps, residual branches and gates are left alone. Chains through
**ReLU6 are skipped**: ReLU6 does not commute with rescaling, and replacing
it by ReLU changed the features of the Keras MobileNetV2 weights by 50 to
110% when it was tried.

``Quantizer`` equalizes automatically when the profile quantizes weights per
tensor and the QAT wrappers were stripped (``equalize=None``), and restores
the original weights after conversion, since layers may be shared with the
model being trained.

Measured quantization quality
-----------------------------

Signal-to-quantization-noise ratio (SQNR, in dB, higher is better; 6 dB is
roughly one bit) of the backbone feature maps, quantized TFLite against the
float model. ImageNet-pretrained weights, 224 x 224, 32 calibration and 16
test images from Oxford-IIIT Pets, measured with this repository.

.. list-table::
   :header-rows: 1
   :widths: 28 12 14 14 16 16

   * - Backbone
     - Tap
     - Per channel
     - Per tensor
     - Per tensor, equalized
     - Pairs equalized
   * - MobileNetV4-Conv-S
     - stride 16
     - 14.9
     - 1.4
     - 7.7
     - 29
   * - MobileNetV4-Conv-S
     - stride 32
     - 19.2
     - 6.7
     - 14.6
     -
   * - MobileNetV2
     - stride 16
     - 16.9
     - 3.0
     - 2.9
     - 2 (ReLU6)
   * - MobileNetV2
     - stride 32
     - 10.9
     - -3.3
     - -3.2
     -
   * - MobileOne-S0, fused
     - stride 16
     - 10.3
     - 5.7
     - 7.8
     - 38
   * - MobileOne-S0, fused
     - stride 32
     - 3.5
     - -3.6
     - 1.2
     -
   * - MobileOne-S0, not fused
     - stride 32
     - -5.1
     - -5.3
     - n/a
     -
   * - MobileOne-S1, fused
     - stride 32
     - -0.9
     - -4.0
     - -3.0
     - 38
   * - RepViT-M0.9, fused
     - stride 32
     - 3.1
     - 1.5
     - not measured
     -

What this says:

* **Fusing helps quantization** as well as speed (MobileOne-S0: -5.1 dB
  unfused against 3.5 dB fused at the deepest tap).
* **Equalization recovers a large part of the per-tensor loss** where it can
  act: MobileNetV4-Conv-S gains 8 dB at the deepest tap and ends above
  per-channel MobileNetV2.
* **MobileNetV2 collapses under per-tensor post-training quantization** and
  cannot be equalized because of ReLU6. On a per-tensor target use a ReLU
  backbone.
* **Fused MobileOne and RepViT quantize poorly post-training even per
  channel**: at the deepest tap the noise is as large as the signal. This is
  the known weakness of reparameterized networks, and nothing in the library
  addresses it yet.
* For a per-tensor accelerator, **MobileNetV4-Conv with equalization is the
  best-measured choice** among the library's backbones today.

These are feature-level measurements on a small image set, a proxy for task
accuracy, not a substitute: no trained head was evaluated. Measure your own
model with ``deploy_utils.quantization_error`` before deploying.

Measuring quantization error
****************************

.. code-block:: python

   errors = deploy_utils.quantization_error(export_model, tflite_bytes, images)   # images: float, normalized
   # one dict per model output: sqnr_db, rmse, max_abs_error, float_std

Run it on a multi-output model (for example the nested backbone, whose
outputs are the feature taps) to see where along the network the error
grows.

Architectures without gates
***************************

Several decoders fuse features with a sigmoid gate (a tensor multiplied by
a tensor), which the published ARTPEC-9 operator list does not cover. Two
builders offer a plain variant, to be chosen before training:

* ``lr_aspp(..., gate=False)``: no global pooling, sigmoid or
  multiplication; the exported head is convolutions, two resizes and a
  concatenation.
* ``pp_liteseg(..., fusion='add')``: the two features are added instead of
  blended by the UAFM attention.

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
  above).
* Post-training quantization of fused MobileOne and RepViT backbones is
  poor (see the measurements above); equalization does not cover ReLU6
  networks; there is no bias correction and no per-layer error report yet.
* Several architectures use operators outside the published Axis lists
  (sigmoid gates, explicit padding, GELU, squeeze-excite, dilated
  convolutions). The lint reports them; it does not rewrite the model.
* The lint checks only part of the published per-operator constraints (for
  example not the pooling window limits of ARTPEC-9).
* The detectors run box decoding and NMS inside the graph; on an accelerator
  it is usually better to export the ``*_base`` model and decode on the CPU.
* ``tf.lite.Interpreter`` is deprecated upstream in favour of the LiteRT
  package (``ai_edge_litert``); the library still uses the TensorFlow one.
* An untrained QAT-wrapped model can output exact zeros. The output
  quantizer starts with a range of -6 to 6, so it rounds to steps of 12/255;
  a randomly initialized ``MobileNetV2`` produces logits around 0.001, which
  all round to zero. It is not a defect and disappears once the outputs have
  a normal magnitude (pretrained weights, or training). To inspect an
  untrained model, look at ``deploy_utils.strip_qat(model)``.

Export tests
************

``tests/test_deploy_utils.py`` covers the profiles, the lint, QAT stripping
and reparameterization on small models. ``pytest -m slow`` runs the full
pipeline on real decoders for the ``artpec8`` and ``artpec9`` profiles and
executes the result in the interpreter. Add a case to
``tests/test_tflite_export.py`` for every new head or backbone family.
