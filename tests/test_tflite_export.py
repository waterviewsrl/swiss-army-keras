"""End-to-end "edge deployable" check: QAT model -> full-integer TFLite.

The library's stated purpose is int8 TFLite models for edge devices, yet until
this file nothing in the suite converted a single model (the only
`TFLiteConverter` call lived inside `Quantizer.quantize()`, which only
`TrainingDriver.run()` reaches). These tests mirror `Quantizer`'s converter
settings exactly - `TFLITE_BUILTINS_INT8` only, uint8 input/output, a
representative dataset - so a backbone or decoder that introduces an op the
int8 converter cannot handle fails here instead of at the end of a training
run (ASSESSMENT.md C4 / missing feature 2).

Marked `slow`: each case builds a model and runs the converter (tens of
seconds to a few minutes on CPU). Run with `pytest -m slow` or deselect with
`-m 'not slow'`.
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras.models import lr_aspp, pp_liteseg

# 128, not 64: PP-LiteSeg's SPPM (bins 1/2/4 over the stride-32 map) needs
# an input of at least 128 - see `sppm_module`'s own ValueError.
INPUT_SIZE = 128
N_LABELS = 3


def _convert_int8(model, n_calibration_batches=2):
    def representative_data_gen():
        for _ in range(n_calibration_batches):
            yield [np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3).astype("float32")]

    model.trainable = False
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    return converter.convert()


def _run_tflite(tflite_bytes, image_uint8):
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    assert inp["dtype"] == np.uint8
    assert out["dtype"] == np.uint8
    interpreter.set_tensor(inp["index"], image_uint8)
    interpreter.invoke()
    return interpreter.get_tensor(out["index"])


@pytest.mark.slow
@pytest.mark.parametrize(
    "decoder, backbone",
    [
        (lr_aspp, "MobileOne-S0"),
        (pp_liteseg, "MobileOne-S0"),
    ],
    ids=lambda v: getattr(v, "__name__", v),
)
def test_qat_decoder_converts_to_full_int8_tflite(decoder, backbone):
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = decoder(inp, n_labels=N_LABELS, backbone=backbone, weights=None, freeze_backbone=False)

    tflite_bytes = _convert_int8(model)
    assert len(tflite_bytes) > 0

    image = (np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3) * 255).astype(np.uint8)
    out = _run_tflite(tflite_bytes, image)
    assert out.shape == (1, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    # A fully-degenerate (constant) output would still have the right shape -
    # see the all-zero QAT-output bug recorded in plan.md §10 / ASSESSMENT.md C5.
    assert out.min() != out.max()


# --------------------------------------------------------------------------- #
# Deployment pipeline: reparameterize -> strip QAT -> convert for a profile -> lint


def _ptq_representative():
    for _ in range(2):
        yield [np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3).astype("float32")]


@pytest.mark.slow
@pytest.mark.parametrize("profile", ["artpec8", "artpec9"])
@pytest.mark.parametrize(
    "decoder, backbone",
    [
        (lr_aspp, "MobileOne-S0"),
        (pp_liteseg, "MobileOne-S0"),
        (lr_aspp, "MobileNetV4-Conv-S"),
    ],
    ids=lambda v: getattr(v, "__name__", v),
)
def test_deployment_pipeline_produces_an_integer_only_model(decoder, backbone, profile):
    """The export `Quantizer` performs for accelerator profiles: no operator
    may be left in floating point, and the weight granularity must match the
    profile. (Operators missing from the target's published list are reported
    by the lint but are a property of the architecture, not of the export,
    so they are not asserted here.)
    """
    from swiss_army_keras import deploy_utils as du

    # Seeded: with some random initializations a sigmoid gate of PP-LiteSeg
    # saturates, a tensor is then constant during calibration, gets a
    # near-zero quantization scale, and XNNPACK intermittently refuses the
    # file (seen in about 1 run in 9 before seeding; the lint now warns).
    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = decoder(inp, n_labels=N_LABELS, backbone=backbone, weights=None, freeze_backbone=False)

    export_model = du.strip_qat(du.reparameterize_model(model))
    assert not du.has_qat_layers(export_model)

    tflite_bytes = du.convert_to_tflite(export_model, _ptq_representative, profile=profile)
    report = du.lint_tflite(tflite_bytes, profile=profile)
    assert report.errors == [], str(report)
    assert "DEQUANTIZE" not in report.ops, str(report)
    if backbone.startswith("MobileOne"):
        # fused: tens of operators, not hundreds
        assert report.ops["ADD"] < 20, str(report)

    image = (np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3) * 255).astype(np.uint8)
    out = _run_tflite(tflite_bytes, image)
    assert out.shape == (1, INPUT_SIZE, INPUT_SIZE, N_LABELS)


@pytest.mark.slow
def test_qat_export_of_lr_aspp_leaves_a_float_operator():
    """Documents WHY accelerator profiles strip the QAT wrappers: exported
    with its wrappers, LR-ASPP keeps the backbone's last convolution in
    floating point (the head's fake-quantization range and the backbone's
    calibrated range disagree on the shared tensor). If this test starts
    failing because the float operator is gone, the stripping default can be
    reconsidered.
    """
    from swiss_army_keras import deploy_utils as du

    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = lr_aspp(inp, n_labels=N_LABELS, backbone="MobileNetV4-Conv-S", weights=None, freeze_backbone=False)
    report = du.lint_tflite(du.convert_to_tflite(model, _ptq_representative, profile="default"), profile="default")
    assert any("floating-point" in e for e in report.errors), str(report)


@pytest.mark.slow
def test_quantizer_end_to_end_for_an_accelerator_profile(tmp_path):
    from swiss_army_keras.quantization_utils import Quantizer

    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = lr_aspp(inp, n_labels=N_LABELS, backbone="MobileOne-S0", weights=None, freeze_backbone=False)
    images = np.random.rand(4, INPUT_SIZE, INPUT_SIZE, 3).astype("float32")
    labels = np.zeros((4, INPUT_SIZE, INPUT_SIZE, N_LABELS), dtype="float32")
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(2)

    quantizer = Quantizer(dataset, model, str(tmp_path / "toy"), append_datetime=False, batches=1,
                          profile="artpec8", strict=False)
    assert quantizer.strip_qat is True          # integer-only profile
    quantizer.quantize()

    assert (tmp_path / "toy_quant_ui8.tflite").stat().st_size > 0
    assert (tmp_path / "toy_quant_f16.tflite").stat().st_size > 0
    assert quantizer.lint_report.errors == [], str(quantizer.lint_report)
    assert quantizer.lint_report.ops["ADD"] < 20      # the MobileOne backbone was fused
    out = _run_tflite(quantizer.tflite_ui8_model, (images[:1] * 255).astype(np.uint8))
    assert out.shape == (1, INPUT_SIZE, INPUT_SIZE, N_LABELS)

    # default profile keeps the historical behaviour (QAT wrappers kept)
    assert Quantizer(dataset, model, str(tmp_path / "d"), append_datetime=False).strip_qat is False


@pytest.mark.slow
def test_gate_free_lr_aspp_export_has_no_gate_operators():
    """`lr_aspp(gate=False)` exists for accelerators that do not run global
    pooling, sigmoid or tensor-by-tensor multiplication."""
    from swiss_army_keras import deploy_utils as du

    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = lr_aspp(inp, n_labels=N_LABELS, backbone="MobileNetV4-Conv-S", weights=None, freeze_backbone=False, gate=False)
    export_model = du.strip_qat(model)
    report = du.lint_tflite(du.convert_to_tflite(export_model, _ptq_representative, profile="artpec9"), profile="artpec9")
    assert report.errors == [], str(report)
    for op in ("LOGISTIC", "MUL", "AVERAGE_POOL_2D", "MEAN"):
        assert op not in report.ops, str(report)
    assert not any("MUL" in w for w in report.warnings), str(report)


@pytest.mark.slow
def test_quantizer_equalizes_for_per_tensor_profiles_and_restores_weights(tmp_path):
    from swiss_army_keras.quantization_utils import Quantizer

    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = lr_aspp(inp, n_labels=N_LABELS, backbone="MobileNetV4-Conv-S", weights=None, freeze_backbone=False)
    weights_before = [w.copy() for w in model.get_weights()]
    images = np.random.rand(4, INPUT_SIZE, INPUT_SIZE, 3).astype("float32")
    labels = np.zeros((4, INPUT_SIZE, INPUT_SIZE, N_LABELS), dtype="float32")
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(2)

    per_tensor = Quantizer(dataset, model, str(tmp_path / "a8"), append_datetime=False, profile="artpec8")
    assert per_tensor.equalize is True
    per_tensor.quantize()
    assert per_tensor.equalization is not None and per_tensor.equalization.pairs > 10
    assert per_tensor.lint_report.errors == [], str(per_tensor.lint_report)
    # MobileNetV4's backbone is shared with the training model: its weights must be back
    for a, b in zip(weights_before, model.get_weights()):
        np.testing.assert_array_equal(a, b)

    assert Quantizer(dataset, model, str(tmp_path / "a9"), append_datetime=False, profile="artpec9").equalize is False
    assert Quantizer(dataset, model, str(tmp_path / "d"), append_datetime=False).equalize is False



@pytest.mark.slow
def test_depth_export_for_artpec8_drops_the_sigmoid(tmp_path):
    """The sigmoid operator is not on the ARTPEC-8 list. `Quantizer` can
    export the depth model without it: the file then passes the check and
    returns the value the sigmoid would have been applied to.
    """
    from swiss_army_keras.models import fastdepth_head
    from swiss_army_keras.quantization_utils import Quantizer

    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = fastdepth_head(inp, backbone="MobileNetV2", weights=None, freeze_backbone=False)
    images = np.random.rand(4, INPUT_SIZE, INPUT_SIZE, 3).astype("float32")
    dataset = tf.data.Dataset.from_tensor_slices((images, np.zeros((4, 1), dtype="float32"))).batch(2)

    kept = Quantizer(dataset, model, str(tmp_path / "kept"), append_datetime=False, profile="artpec8")
    kept.quantize()
    assert "LOGISTIC" in kept.lint_report.unsupported_ops

    dropped = Quantizer(dataset, model, str(tmp_path / "dropped"), append_datetime=False, profile="artpec8",
                        drop_output_activations=True)
    dropped.quantize()
    assert dropped.export_model.dropped_output_activations == ["sigmoid"]
    assert dropped.lint_report.ok, str(dropped.lint_report)
    out = _run_tflite(dropped.tflite_ui8_model, (images[:1] * 255).astype(np.uint8))
    assert out.shape == (1, INPUT_SIZE, INPUT_SIZE, 1)


@pytest.mark.slow
def test_multi_task_exports_one_integer_file_with_named_outputs():
    """One backbone, three heads, one file. The converter reorders the
    outputs, so `tflite_outputs` is what tells the application which
    position holds which head.
    """
    from swiss_army_keras import deploy_utils
    from swiss_army_keras.models import multi_task

    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = multi_task(inp, {
        "seg": dict(type="lr_aspp", n_labels=N_LABELS, gate=False),
        "depth": dict(type="fastdepth"),
        "cls": dict(type="classifier", n_classes=4, size=32, activation="relu"),
    }, backbone="MobileNetV4-Conv-S", weights=None, freeze_backbone=False)

    plain = deploy_utils.drop_output_activations(deploy_utils.strip_qat(deploy_utils.reparameterize_model(model)))
    tflite = deploy_utils.convert_to_tflite(plain, _ptq_representative, profile="artpec9")
    report = deploy_utils.lint_tflite(tflite, profile="artpec9")
    assert report.errors == [], str(report)
    assert not {"LOGISTIC", "SOFTMAX"} & set(report.ops)

    outputs = {d["name"]: d for d in deploy_utils.tflite_outputs(tflite)}
    assert set(outputs) == {"seg", "depth", "cls"}
    assert outputs["seg"]["shape"] == (1, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert outputs["depth"]["shape"] == (1, INPUT_SIZE, INPUT_SIZE, 1)
    assert outputs["cls"]["shape"] == (1, 4)
    assert sorted(d["position"] for d in outputs.values()) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# Additions of 2026-11-10

def _dataset(n_outputs=1, size=INPUT_SIZE):
    images = np.random.rand(4, size, size, 3).astype("float32")
    return tf.data.Dataset.from_tensor_slices((images, np.zeros((4, 1), dtype="float32"))).batch(2), images


@pytest.mark.slow
def test_plain_backbone_with_pyramid_head_is_clean_for_both_axis_profiles(tmp_path):
    """The first model of the library with nothing outside either published
    Axis operator list: ResNet-18 with the plain pyramid head.
    """
    from swiss_army_keras.models import fpn_seg
    from swiss_army_keras.quantization_utils import Quantizer

    model = fpn_seg(tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3)), N_LABELS, backbone="ResNet18", weights=None,
                    freeze_backbone=False)
    dataset, images = _dataset()
    for profile in ("artpec8", "artpec9"):
        quantizer = Quantizer(dataset, model, str(tmp_path / profile), append_datetime=False, batches=2,
                              profile=profile)
        quantizer.quantize()
        report = quantizer.lint_report
        assert report.errors == [] and not report.unsupported_ops, str(report)
        assert set(report.ops) <= {"QUANTIZE", "CONV_2D", "MAX_POOL_2D", "ADD", "RESIZE_BILINEAR"}
        out = _run_tflite(quantizer.tflite_ui8_model, (images[:1] * 255).astype(np.uint8))
        assert out.shape == (1, INPUT_SIZE, INPUT_SIZE, N_LABELS)


@pytest.mark.slow
@pytest.mark.parametrize("name, main_output", [("ddrnet", 0), ("pidnet", 1)])
def test_dual_resolution_networks_export_their_main_output_only(tmp_path, name, main_output):
    from swiss_army_keras.models import ddrnet, pidnet
    from swiss_army_keras.quantization_utils import Quantizer

    builder = {"ddrnet": ddrnet, "pidnet": pidnet}[name]
    model = builder(tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3)), n_labels=N_LABELS, augment=True)
    dataset, images = _dataset()
    quantizer = Quantizer(dataset, model, str(tmp_path / name), append_datetime=False, batches=2,
                          profile="artpec9", outputs=main_output)
    quantizer.quantize()

    from swiss_army_keras import deploy_utils
    outputs = deploy_utils.tflite_outputs(quantizer.tflite_ui8_model)
    assert len(outputs) == 1 and outputs[0]["shape"] == (1, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert quantizer.lint_report.errors == [], str(quantizer.lint_report)
    if name == "ddrnet":
        assert not quantizer.lint_report.unsupported_ops, str(quantizer.lint_report)   # nothing off the ARTPEC-9 list
    _run_tflite(quantizer.tflite_ui8_model, (images[:1] * 255).astype(np.uint8))


@pytest.mark.slow
def test_per_tensor_export_folds_equalizes_and_corrects_bias_then_restores(tmp_path):
    from swiss_army_keras.quantization_utils import Quantizer

    model = lr_aspp(tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3)), n_labels=N_LABELS, gate=False,
                    backbone="MobileNetV4-Conv-S", weights=None, freeze_backbone=False)
    before = [w.copy() for w in model.get_weights()]
    dataset, _ = _dataset()
    quantizer = Quantizer(dataset, model, str(tmp_path / "pt"), append_datetime=False, batches=2, profile="artpec8")
    quantizer.quantize()

    assert quantizer.bias_corrected is not None and quantizer.bias_corrected.layers > 20
    assert quantizer.equalization is not None and quantizer.equalization.pairs > 10
    assert not any(isinstance(layer, tf.keras.layers.BatchNormalization) and layer.weights
                   for layer in quantizer.export_model.layers)
    assert quantizer.lint_report.errors == [], str(quantizer.lint_report)
    for old, new in zip(before, model.get_weights()):
        np.testing.assert_array_equal(old, new)                      # the trained model is untouched


@pytest.mark.slow
def test_fixed_batch_conversion_removes_deeplab_space_to_batch_operators():
    """With a dynamic batch the converter exports each dilated convolution
    of the DeepLab pyramid as SPACE_TO_BATCH_ND + convolution +
    BATCH_TO_SPACE_ND, operators that are on neither Axis list. Converting
    with the fixed batch of the accelerator profiles keeps them as single
    convolutions with a dilation factor.
    """
    from swiss_army_keras import deploy_utils
    from swiss_army_keras.models import deeplab_v3_plus

    size = 256      # a deep map large enough for the dilation rates to matter
    def representative():
        for _ in range(2):
            yield [np.random.rand(1, size, size, 3).astype("float32")]

    model = deeplab_v3_plus(tf.keras.Input((size, size, 3)), n_labels=N_LABELS, backbone="MobileNetV2",
                            weights=None, freeze_backbone=False)
    plain = deploy_utils.strip_qat(model)
    fixed = deploy_utils.list_tflite_ops(deploy_utils.convert_to_tflite(plain, representative, profile="artpec9"))
    dynamic = deploy_utils.list_tflite_ops(
        deploy_utils.convert_to_tflite(plain, representative, profile="artpec9", static_batch=0))
    assert {"SPACE_TO_BATCH_ND", "BATCH_TO_SPACE_ND"} <= set(dynamic)
    assert not {"SPACE_TO_BATCH_ND", "BATCH_TO_SPACE_ND"} & set(fixed)


@pytest.mark.slow
def test_repaired_heads_export_without_their_former_off_list_operators():
    """FCOS distances without EXP, GFL sides without STRIDED_SLICE, the pose
    head without shape arithmetic, the embedding head without L2
    normalization, the classifier without MEAN (CANDIDATES_BY_TASK.md
    section 2).
    """
    from swiss_army_keras import deploy_utils
    from swiss_army_keras._model_detection import fcos_lite_detection_base, gfl_lite_detection_base
    from swiss_army_keras.models import classifier, pose_regression_head, reid_embedding_head

    kwargs = dict(backbone="ResNet18", weights=None, freeze_backbone=False)

    def export_ops(model):
        plain = deploy_utils.strip_qat(model)
        plain = deploy_utils.replace_global_pooling(deploy_utils.drop_output_activations(plain))
        return set(deploy_utils.list_tflite_ops(
            deploy_utils.convert_to_tflite(plain, _ptq_representative, profile="artpec8")))

    def inp():
        return tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))

    assert "EXP" not in export_ops(fcos_lite_detection_base(inp(), n_classes=3, distance_activation="relu", **kwargs))
    assert "STRIDED_SLICE" not in export_ops(gfl_lite_detection_base(inp(), n_classes=3, flat_reg=True, **kwargs))
    pose = export_ops(pose_regression_head(inp(), 6, activation="relu", **kwargs))
    assert not {"SHAPE", "PACK", "STRIDED_SLICE", "LOGISTIC", "MEAN"} & pose
    reid = export_ops(reid_embedding_head(inp(), pre_pool_attention=None, activation="relu", **kwargs))
    assert not {"L2_NORMALIZATION", "POW", "MEAN"} & reid
    assert not {"MEAN", "SOFTMAX", "LOGISTIC"} & export_ops(classifier(inp(), 4, activation="relu", **kwargs))


@pytest.mark.slow
@pytest.mark.parametrize("build", [
    lambda inp: lr_aspp(inp, n_labels=N_LABELS, backbone="MobileOne-S0", weights=None, freeze_backbone=False),
    lambda inp: __import__("swiss_army_keras._model_detection", fromlist=["x"]).gfl_lite_detection_base(
        inp, n_classes=3, flat_reg=True, backbone="ResNet18", weights=None),
    # its sigmoid gates declare a fixed output range; with a learned one the
    # export had two floating-point requantizations
    lambda inp: __import__("swiss_army_keras.models", fromlist=["x"]).pidnet(inp, N_LABELS),
], ids=["lr_aspp-MobileOne-S0", "gfl_base-ResNet18", "pidnet"])
def test_whole_model_qat_fine_tunes_and_exports_per_tensor(tmp_path, build):
    """`prepare_qat` on real library models: a fused reparameterized
    backbone with a gated head, and a detector whose output convolutions
    are shared across pyramid levels. One optimizer step, then the
    `Quantizer` export for ARTPEC-8 must be integer-only with per-tensor
    weights and match the prepared model's simulation.
    """
    from swiss_army_keras import deploy_utils
    from swiss_army_keras.quantization_utils import Quantizer

    tf.keras.utils.set_random_seed(0)
    model = build(tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3)))
    images = np.random.default_rng(0).random((12, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    prepared = deploy_utils.prepare_qat(model, images[:8], per_channel=False)
    assert prepared.unwrapped_layers == ()

    targets = [np.zeros((len(images),) + tuple(shape[1:]), "float32") for shape in
               (prepared.output_shape if isinstance(prepared.output_shape, list) else [prepared.output_shape])]
    prepared.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="mse")
    prepared.fit(images, targets if len(targets) > 1 else targets[0], batch_size=4, epochs=1, steps_per_epoch=1,
                 verbose=0)

    dataset = tf.data.Dataset.from_tensor_slices((images, np.zeros((len(images), 1), "float32"))).batch(4)
    quantizer = Quantizer(dataset, prepared, str(tmp_path / "qat"), append_datetime=False, batches=2, profile="artpec8")
    quantizer.quantize()
    assert quantizer.lint_report.errors == [], str(quantizer.lint_report)
