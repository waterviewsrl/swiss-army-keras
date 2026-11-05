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

