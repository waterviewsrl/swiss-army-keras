"""Deployment helpers: profiles, TFLite conversion per profile, the lint of
exported files, QAT stripping and reparameterization of full models.

The conversion tests use a tiny hand-built network so they stay fast; the
end-to-end exports of real decoders live in `tests/test_tflite_export.py`
(marked `slow`).
"""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from swiss_army_keras import deploy_utils as du
from swiss_army_keras.layer_utils import qat_annotate_layer
from swiss_army_keras.models import classifier, lr_aspp

L = tf.keras.layers
SIZE = 32


def _representative():
    rng = np.random.default_rng(0)
    for _ in range(8):
        yield [rng.random((1, SIZE, SIZE, 3), dtype=np.float32)]


def _toy_model(activation="relu", gate=False):
    """Conv-BN-activation x2 (+ an optional squeeze-excite style gate)."""
    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input((SIZE, SIZE, 3))
    x = L.Conv2D(12, 3, strides=2, padding="same", use_bias=False)(inp)
    x = L.BatchNormalization()(x)
    x = L.LeakyReLU(0.1)(x) if activation == "leaky" else L.ReLU()(x)
    x = L.Conv2D(16, 3, padding="same")(x)
    x = L.ReLU()(x)
    if gate:
        g = L.Conv2D(16, 1, activation="sigmoid")(x)
        x = L.Multiply()([x, g])
    return tf.keras.Model(inp, x)


def _toy_qat_model():
    """An un-annotated 'backbone' conv feeding a QAT-wrapped head."""
    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input((SIZE, SIZE, 3))
    x = L.Conv2D(8, 3, strides=2, padding="same", name="bb_conv")(inp)
    x = L.ReLU(name="bb_relu")(x)
    x = qat_annotate_layer(L.Conv2D(8, 3, padding="same", use_bias=False, name="h_conv"))(x)
    x = qat_annotate_layer(L.BatchNormalization(name="h_bn"))(x)
    x = qat_annotate_layer(L.Activation("relu", name="h_act"))(x)
    x = L.GlobalAveragePooling2D(name="h_gap")(x)
    x = qat_annotate_layer(L.Dense(6, activation="swish", name="h_dense"))(x)
    out = qat_annotate_layer(L.Dense(3, name="h_out"))(x)
    model = tf.keras.Model(inp, out)
    with tfmot.quantization.keras.quantize_scope():
        return tfmot.quantization.keras.quantize_apply(model)


# --------------------------------------------------------------------------- #
# profiles


def test_profiles_are_consistent():
    assert {"default", "litert-cpu", "artpec7", "artpec8", "artpec9"} <= set(du.PROFILES)
    for name, profile in du.PROFILES.items():
        assert profile.name == name
        assert profile.io_dtype in ("uint8", "int8", "float32")
        for constraint in profile.constraints:
            assert constraint in du._CONSTRAINT_CHECKS
    assert du.PROFILES["artpec8"].per_channel is False
    assert du.PROFILES["artpec8"].integer_only and du.PROFILES["artpec9"].integer_only
    assert not du.PROFILES["default"].integer_only


def test_get_profile_accepts_names_and_objects_and_rejects_unknown():
    assert du.get_profile("artpec8") is du.PROFILES["artpec8"]
    assert du.get_profile(du.PROFILES["artpec9"]) is du.PROFILES["artpec9"]
    with pytest.raises(ValueError, match="Unknown deployment profile"):
        du.get_profile("no-such-target")


# --------------------------------------------------------------------------- #
# conversion + lint on a tiny model


def test_default_profile_exports_per_channel_uint8_and_lints_clean():
    tflite = du.convert_to_tflite(_toy_model(), _representative, profile="default")
    report = du.lint_tflite(tflite, profile="default")
    assert report.ok, str(report)
    assert report.ops["CONV_2D"] == 2
    assert du.list_tflite_ops(tflite) == report.ops
    # the same per-channel file does NOT satisfy a per-tensor target
    report8 = du.lint_tflite(tflite, profile="artpec8")
    assert any("per channel" in e for e in report8.errors), str(report8)
    assert not report8.ok


def test_artpec8_profile_exports_per_tensor_weights():
    tflite = du.convert_to_tflite(_toy_model(), _representative, profile="artpec8")
    report = du.lint_tflite(tflite, profile="artpec8")
    assert report.errors == [], str(report)
    assert report.unsupported_ops == {}, str(report)
    # 12 is a multiple of 6, 16 is not
    assert any("multiple of 6" in w for w in report.warnings), str(report)
    interpreter = tf.lite.Interpreter(model_content=tflite)
    interpreter.allocate_tensors()
    assert interpreter.get_input_details()[0]["dtype"] == np.uint8


def test_lint_reports_operators_missing_from_the_target_list():
    tflite = du.convert_to_tflite(_toy_model(activation="leaky"), _representative, profile="artpec8")
    report = du.lint_tflite(tflite, profile="artpec8")
    assert "LEAKY_RELU" in report.unsupported_ops, str(report)
    assert not report.ok
    assert "NOT ON THE TARGET OPERATOR LIST" in str(report)


def test_artpec9_constraint_flags_tensor_by_tensor_multiplication():
    tflite = du.convert_to_tflite(_toy_model(gate=True), _representative, profile="artpec9")
    report = du.lint_tflite(tflite, profile="artpec9")
    assert any("MUL" in w for w in report.warnings), str(report)


def test_lint_reports_io_type_mismatch_and_overrides_are_honoured():
    tflite = du.convert_to_tflite(_toy_model(), _representative, profile="default", io_dtype="int8")
    assert any("input tensor is int8" in e for e in du.lint_tflite(tflite, profile="default").errors)
    assert du.lint_tflite(tflite, profile="default", io_dtype="int8").ok


def test_float16_conversion_needs_no_representative_dataset():
    tflite = du.convert_to_tflite(_toy_model(), float16=True)
    interpreter = tf.lite.Interpreter(model_content=tflite)
    interpreter.allocate_tensors()
    assert interpreter.get_input_details()[0]["dtype"] == np.float32
    with pytest.raises(ValueError, match="representative_dataset"):
        du.convert_to_tflite(_toy_model(), profile="default")


# --------------------------------------------------------------------------- #
# strip_qat


def test_strip_qat_returns_plain_models_unchanged():
    model = _toy_model()
    assert not du.has_qat_layers(model)
    assert du.strip_qat(model) is model


def test_strip_qat_removes_wrappers_and_keeps_the_trained_weights():
    qat = _toy_qat_model()
    assert du.has_qat_layers(qat)
    plain = du.strip_qat(qat)
    assert not du.has_qat_layers(plain)

    wrapped = {l.layer.name: l for l in qat.layers if hasattr(l, "layer") and hasattr(l, "quantize_config")}
    checked = 0
    for layer in plain.layers:
        if layer.name in wrapped and layer.weights:
            source = {v.name.rsplit("/", 1)[-1]: v.numpy() for v in wrapped[layer.name].weights}
            for variable in layer.weights:
                np.testing.assert_array_equal(variable.numpy(), source[variable.name.rsplit("/", 1)[-1]])
                checked += 1
    assert checked >= 6  # conv kernel, 4 BN variables, 2 dense kernels + biases

    # activations substituted by tfmot are restored
    assert plain.get_layer("h_dense").activation is tf.keras.activations.swish
    assert plain.get_layer("h_conv").activation is tf.keras.activations.linear

    x = np.random.default_rng(1).random((4, SIZE, SIZE, 3), dtype=np.float32)
    a, b = qat.predict(x, verbose=0), plain.predict(x, verbose=0)
    assert a.shape == b.shape
    assert b.std() > 0
    # Fake quantization only perturbs the float computation. An untrained
    # output quantizer spans [-6, 6], i.e. rounds to steps of 12/255, so the
    # two outputs agree to within one such step.
    assert np.abs(a - b).max() <= 12 / 255 + 1e-3


def test_stripped_model_exports_per_tensor_where_the_qat_model_does_not():
    qat = _toy_qat_model()
    report_qat = du.lint_tflite(du.convert_to_tflite(qat, _representative, profile="artpec8"), profile="artpec8")
    assert any("per channel" in e for e in report_qat.errors), str(report_qat)
    plain = du.strip_qat(qat)
    report = du.lint_tflite(du.convert_to_tflite(plain, _representative, profile="artpec8"), profile="artpec8")
    assert report.errors == [], str(report)


# --------------------------------------------------------------------------- #
# reparameterize_model


def _nested_backbone(model):
    return next(l for l in model.layers if isinstance(l, tf.keras.Model))


def _layer_count(model):
    return sum(len(l.layers) if isinstance(l, tf.keras.Model) else 1 for l in model.layers)


@pytest.mark.parametrize(
    "build",
    [
        lambda i: lr_aspp(i, n_labels=3, backbone="MobileOne-S0", weights=None, freeze_backbone=False),
        lambda i: classifier(i, n_classes=4, backbone="RepViT-M0.9", weights=None, freeze_backbone=False, deep_layer=4),
        lambda i: classifier(i, n_classes=4, backbone="FastViT-T8", weights=None, freeze_backbone=False, deep_layer=4),
    ],
    ids=["lr_aspp-MobileOne-S0", "classifier-RepViT-M0.9", "classifier-FastViT-T8"],
)
def test_reparameterize_model_fuses_a_nested_backbone(build):
    tf.keras.utils.set_random_seed(0)
    model = build(tf.keras.Input((64, 64, 3)))
    fused = du.reparameterize_model(model)

    assert fused is not model
    assert _layer_count(fused) < _layer_count(model)
    assert fused.preprocessing is model.preprocessing
    assert fused.output_shape == model.output_shape

    # the fused backbone computes the same feature pyramid
    x = np.random.default_rng(0).random((2, 64, 64, 3), dtype=np.float32)
    before = _nested_backbone(model).predict(x, verbose=0)
    after = _nested_backbone(fused).predict(x, verbose=0)
    assert len(before) == len(after)
    for a, b in zip(before, after):
        assert a.shape == b.shape
        assert np.abs(a).max() > 0
        assert np.abs(a - b).max() <= 2e-3 * np.abs(a).max()

    # the head layers are shared, not copied
    head_names = {l.name for l in model.layers if not isinstance(l, tf.keras.Model)}
    for name in head_names:
        if model.get_layer(name).weights:
            assert fused.get_layer(name) is model.get_layer(name)

    # idempotent
    assert du.reparameterize_model(fused) is fused


def test_reparameterize_model_leaves_other_backbones_alone():
    model = classifier(tf.keras.Input((64, 64, 3)), n_classes=4, backbone="MobileNetV2", weights=None)
    assert du.reparameterize_model(model) is model


def test_reparameterize_model_accepts_a_standalone_backbone():
    from swiss_army_keras._backbone_mobileone import MobileOne_S0

    backbone = MobileOne_S0(include_top=False, weights=None, input_shape=(64, 64, 3))
    fused = du.reparameterize_model(backbone)
    assert len(fused.layers) < len(backbone.layers)
    x = np.random.default_rng(0).random((1, 64, 64, 3), dtype=np.float32)
    a, b = backbone.predict(x, verbose=0), fused.predict(x, verbose=0)
    assert np.abs(a - b).max() <= 2e-3 * np.abs(a).max()
    assert du.reparameterize_model(fused) is fused
