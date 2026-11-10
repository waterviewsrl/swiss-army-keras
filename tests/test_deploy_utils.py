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


# --------------------------------------------------------------------------- #
# cross-layer equalization and quantization error


def _unbalanced_chain(final_relu6=False):
    """conv -> BN -> ReLU -> depthwise -> ReLU -> conv, where some channels
    of the first two layers are deliberately 100x larger than others (and
    the next layer compensates), the situation per-tensor quantization
    cannot represent."""
    tf.keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    inp = tf.keras.Input((SIZE, SIZE, 3))
    x = L.Conv2D(16, 3, padding="same", use_bias=False, name="c1")(inp)
    x = L.BatchNormalization(name="bn1")(x)
    x = L.ReLU(6.0, name="r1")(x) if final_relu6 else L.Activation("relu", name="r1")(x)
    x = L.ZeroPadding2D(1, name="pad")(x)
    x = L.DepthwiseConv2D(3, padding="valid", name="dw")(x)
    x = L.Activation("relu", name="r2")(x)
    out = L.Conv2D(8, 1, name="c3")(x)
    model = tf.keras.Model(inp, out)

    spread = np.where(np.arange(16) % 4 == 0, 30.0, 0.3).astype("float32")
    bn = model.get_layer("bn1")
    bn.set_weights([spread, rng.normal(0, 0.1, 16).astype("float32"),
                    rng.normal(0, 0.1, 16).astype("float32"), np.ones(16, "float32")])
    dw = model.get_layer("dw")
    k, b = dw.get_weights()
    dw.set_weights([k / spread[None, None, :, None] * spread[::-1][None, None, :, None], b])
    c3 = model.get_layer("c3")
    k, b = c3.get_weights()
    c3.set_weights([k / spread[::-1][None, None, :, None], b])
    return model


def test_equalization_preserves_the_function_and_can_be_restored():
    model = _unbalanced_chain()
    x = np.random.default_rng(1).random((4, SIZE, SIZE, 3), dtype=np.float32)
    before = model(x, training=False).numpy()
    original = [w.copy() for w in model.get_weights()]

    result = du.equalize_cross_layer(model)
    assert result.pairs == 2                      # (c1+bn1 -> dw) and (dw -> c3)
    assert result.iterations >= 1 and result.max_scale > 5
    after = model(x, training=False).numpy()
    np.testing.assert_allclose(after, before, rtol=1e-3, atol=1e-4 * np.abs(before).max())
    assert any(not np.allclose(a, b) for a, b in zip(original, model.get_weights()))

    # per-channel weight ranges inside the depthwise kernel are now similar
    dw_range = np.abs(model.get_layer("dw").get_weights()[0]).max(axis=(0, 1, 3))
    assert dw_range.max() / dw_range.min() < 30

    result.restore()
    for a, b in zip(original, model.get_weights()):
        np.testing.assert_array_equal(a, b)


def test_equalization_improves_per_tensor_quantization():
    x = np.random.default_rng(2).random((8, SIZE, SIZE, 3), dtype=np.float32)

    def sqnr(model):
        tflite = du.convert_to_tflite(model, _representative, profile="artpec8")
        return du.quantization_error(model, tflite, x)[0]["sqnr_db"]

    plain = sqnr(_unbalanced_chain())
    model = _unbalanced_chain()
    du.equalize_cross_layer(model)
    equalized = sqnr(model)
    assert equalized > plain + 6, (plain, equalized)     # at least one more bit of precision


def test_equalization_skips_relu6_shared_tensors_and_qat_models():
    # ReLU6 is not positively homogeneous: the first pair must be skipped
    relu6 = _unbalanced_chain(final_relu6=True)
    assert du.equalize_cross_layer(relu6).pairs == 1

    # a tensor with two consumers (here: also a model output) cannot be rescaled
    inp = tf.keras.Input((SIZE, SIZE, 3))
    a = L.Conv2D(8, 3, padding="same", activation="relu")(inp)
    b = L.Conv2D(8, 3, padding="same")(a)
    assert du.equalize_cross_layer(tf.keras.Model(inp, [a, b])).pairs == 0

    with pytest.raises(ValueError, match="strip_qat"):
        du.equalize_cross_layer(_toy_qat_model())


def test_equalization_reaches_into_nested_backbones_but_not_across_their_taps():
    model = du.strip_qat(du.reparameterize_model(
        lr_aspp(tf.keras.Input((64, 64, 3)), n_labels=3, backbone="MobileOne-S0", weights=None, freeze_backbone=False)))
    x = np.random.default_rng(3).random((2, 64, 64, 3), dtype=np.float32)
    backbone = _nested_backbone(model)
    taps_before = [t.numpy() for t in backbone(x, training=False)]
    out_before = model(x, training=False).numpy()

    result = du.equalize_cross_layer(model)
    assert result.pairs > 20
    # the feature taps the head consumes are untouched, hence so is the output
    for a, b in zip(taps_before, backbone(x, training=False)):
        assert np.abs(a - b.numpy()).max() <= 2e-3 * np.abs(a).max()
    assert np.abs(out_before - model(x, training=False).numpy()).max() <= 2e-3 * np.abs(out_before).max()


def test_quantization_error_reports_one_entry_per_output():
    model = _toy_model()
    x = np.random.default_rng(4).random((4, SIZE, SIZE, 3), dtype=np.float32)
    float16 = du.quantization_error(model, du.convert_to_tflite(model, float16=True), x)
    int8 = du.quantization_error(model, du.convert_to_tflite(model, _representative, profile="default"), x)
    assert len(float16) == len(int8) == 1
    assert set(int8[0]) == {"sqnr_db", "rmse", "max_abs_error", "float_std"}
    assert float16[0]["sqnr_db"] > 40           # float16 weights are nearly exact
    assert 5 < int8[0]["sqnr_db"] < float16[0]["sqnr_db"]


# --------------------------------------------------------------------------- #
# Output activations and output description

def _two_output_model():
    """A sigmoid map named by an identity layer, and a fused-softmax vector."""
    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input((SIZE, SIZE, 3))
    x = L.Conv2D(8, 3, strides=2, padding="same", activation="relu")(inp)
    depth = L.Conv2D(1, 1, name="depth_conv")(x)
    depth = L.Activation("sigmoid", name="depth_sigmoid")(depth)
    depth = L.Activation("linear", name="depth")(depth)
    cls = L.Dense(3, activation="softmax", name="cls")(L.GlobalAveragePooling2D()(x))
    model = tf.keras.Model(inp, [depth, cls])
    model.preprocessing = "sentinel"
    return model


def test_drop_output_activations_returns_logits_and_keeps_names():
    model = _two_output_model()
    dropped = du.drop_output_activations(model)

    assert dropped is not model
    assert dropped.dropped_output_activations == ["sigmoid", "softmax"]
    assert dropped.preprocessing == "sentinel"
    assert dropped.output_names[0] == "depth"  # the naming layer is kept

    x = np.random.default_rng(0).random((2, SIZE, SIZE, 3), dtype=np.float32)
    depth, probabilities = model.predict(x, verbose=0)
    depth_logits, logits = dropped.predict(x, verbose=0)
    np.testing.assert_allclose(tf.sigmoid(depth_logits).numpy(), depth, atol=1e-6)
    np.testing.assert_allclose(tf.nn.softmax(logits).numpy(), probabilities, atol=1e-6)


def test_drop_output_activations_is_selective_and_a_no_op_without_activations():
    model = _two_output_model()
    only_sigmoid = du.drop_output_activations(model, activations=("sigmoid",))
    assert only_sigmoid.dropped_output_activations == ["sigmoid", None]

    plain = _toy_model()
    assert du.drop_output_activations(plain) is plain


def test_drop_output_activations_handles_qat_wrapped_and_tf_op_sigmoids():
    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input((SIZE, SIZE, 3))
    x = qat_annotate_layer(L.Conv2D(1, 3, padding="same", name="h_conv"))(inp)
    x = qat_annotate_layer(L.Activation("sigmoid", name="h_sigmoid"))(x)
    with tfmot.quantization.keras.quantize_scope():
        qat = tfmot.quantization.keras.quantize_apply(tf.keras.Model(inp, x))
    assert du.drop_output_activations(qat).dropped_output_activations == ["sigmoid"]

    inp = tf.keras.Input((SIZE, SIZE, 3))
    wrapped = tf.keras.Model(inp, tf.sigmoid(L.Conv2D(1, 3, padding="same")(inp)))
    dropped = du.drop_output_activations(wrapped)
    assert dropped.dropped_output_activations == ["sigmoid"]
    x = np.random.default_rng(0).random((1, SIZE, SIZE, 3), dtype=np.float32)
    np.testing.assert_allclose(tf.sigmoid(dropped.predict(x, verbose=0)).numpy(), wrapped.predict(x, verbose=0), atol=1e-6)


def test_dropping_the_sigmoid_removes_logistic_from_the_export():
    model = _two_output_model()
    with_sigmoid = du.list_tflite_ops(du.convert_to_tflite(model, _representative, profile="artpec8"))
    without = du.list_tflite_ops(du.convert_to_tflite(du.drop_output_activations(model), _representative, profile="artpec8"))
    assert "LOGISTIC" in with_sigmoid and "SOFTMAX" in with_sigmoid
    assert "LOGISTIC" not in without and "SOFTMAX" not in without


def test_tflite_outputs_maps_file_positions_to_keras_output_names():
    model = _two_output_model()
    described = du.tflite_outputs(du.convert_to_tflite(model, _representative, profile="artpec8"))

    assert [d["position"] for d in described] == [0, 1]
    by_name = {d["name"]: d for d in described}
    assert set(by_name) == {"depth", "cls"}
    assert by_name["depth"]["shape"] == (1, SIZE // 2, SIZE // 2, 1)
    assert by_name["cls"]["shape"] == (1, 3)
    assert all(d["dtype"] == "uint8" and d["scale"] > 0 for d in described)


# --------------------------------------------------------------------------- #
# Graph rewrites and accelerator conversion details

def test_select_outputs_by_index_and_name_drops_unused_layers():
    model = _two_output_model()
    only_depth = du.select_outputs(model, "depth")
    assert only_depth.output_names == ["depth"]
    assert not any(layer.name == "cls" for layer in only_depth.layers)
    assert only_depth.preprocessing == "sentinel"

    swapped = du.select_outputs(model, [1, 0])
    assert swapped.output_names == ["cls", "depth"]
    with pytest.raises(ValueError, match="no output named"):
        du.select_outputs(model, "nope")
    with pytest.raises(ValueError, match="out of range"):
        du.select_outputs(model, 5)


def _pooled_classifier(keepdims=False):
    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input((SIZE, SIZE, 3))
    x = L.Conv2D(8, 3, strides=2, padding="same", activation="relu")(inp)
    x = L.GlobalAveragePooling2D(keepdims=keepdims, name="gap")(x)
    if keepdims:
        x = L.Flatten()(x)
    return tf.keras.Model(inp, L.Dense(3, name="logits")(x))


@pytest.mark.parametrize("keepdims", [False, True])
def test_replace_global_pooling_keeps_the_function_and_changes_the_operator(keepdims):
    model = _pooled_classifier(keepdims)
    rewritten = du.replace_global_pooling(model)
    assert not any(isinstance(l, L.GlobalAveragePooling2D) for l in rewritten.layers)

    x = np.random.default_rng(0).random((2, SIZE, SIZE, 3), dtype=np.float32)
    np.testing.assert_allclose(rewritten.predict(x, verbose=0), model.predict(x, verbose=0), atol=1e-6)

    before = du.list_tflite_ops(du.convert_to_tflite(model, _representative, profile="artpec8"))
    after = du.list_tflite_ops(du.convert_to_tflite(rewritten, _representative, profile="artpec8"))
    assert "MEAN" in before and "MEAN" not in after and "AVERAGE_POOL_2D" in after

    assert du.replace_global_pooling(_toy_model()) is not None      # nothing to replace: returned as is
    with pytest.raises(ValueError, match="strip_qat"):
        du.replace_global_pooling(_toy_qat_model())


def test_strip_qat_handles_graphs_keras_cannot_clone():
    """A QAT sub-model called inside a graph of `tf` operations with
    constant arguments (what `uint8_input_wrapper` and the detectors build)
    cannot be cloned from its configuration; `strip_qat` replays it.
    """
    qat = _toy_qat_model()
    inp = tf.keras.Input((SIZE, SIZE, 3))
    scaled = (inp - np.array([0.5, 0.4, 0.3], dtype="float32")) / np.array([0.2, 0.2, 0.2], dtype="float32")
    logits = qat(scaled)
    outer = tf.keras.Model(inp, tf.concat([logits, logits * 2.0], axis=-1))

    stripped = du.strip_qat(outer)
    assert not du.has_qat_layers(stripped)
    x = np.random.default_rng(0).random((2, SIZE, SIZE, 3), dtype=np.float32)
    assert stripped.predict(x, verbose=0).shape == (2, 6)
    # the stripped model computes what the float graph inside the wrappers computes
    plain_inner = du.strip_qat(qat)
    expected = plain_inner.predict((x - [0.5, 0.4, 0.3]) / 0.2, verbose=0)
    np.testing.assert_allclose(stripped.predict(x, verbose=0)[:, :3], expected, atol=1e-5)


def test_drop_output_activations_handles_l2_normalization_and_a_trailing_reshape():
    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input((SIZE, SIZE, 3))
    feature = L.GlobalAveragePooling2D()(L.Conv2D(8, 3, activation="relu")(inp))
    embedding = L.UnitNormalization(name="l2")(L.Dense(4, name="embedding")(feature))
    keypoints = L.Reshape((2, 3), name="reshape")(L.Dense(6, activation="sigmoid", name="keypoints")(feature))
    model = tf.keras.Model(inp, [embedding, keypoints])

    dropped = du.drop_output_activations(model)
    assert dropped.dropped_output_activations == ["l2_normalize", "sigmoid"]
    assert dropped.output_shape == [(None, 4), (None, 2, 3)]

    x = np.random.default_rng(0).random((2, SIZE, SIZE, 3), dtype=np.float32)
    raw_embedding, logits = dropped.predict(x, verbose=0)
    reference_embedding, reference_keypoints = model.predict(x, verbose=0)
    np.testing.assert_allclose(tf.math.l2_normalize(raw_embedding, axis=-1).numpy(), reference_embedding, atol=1e-6)
    np.testing.assert_allclose(tf.sigmoid(logits).numpy(), reference_keypoints, atol=1e-6)


def test_accelerator_profiles_convert_with_a_fixed_batch_of_one():
    """A fixed batch lets the converter fold the shape arithmetic of
    reshapes: no SHAPE / PACK / STRIDED_SLICE operators in the file.
    """
    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input((SIZE, SIZE, 3))
    x = L.GlobalAveragePooling2D()(L.Conv2D(8, 3, activation="relu")(inp))
    out = L.Reshape((2, 3), name="pairs")(L.Dense(6)(x))
    model = tf.keras.Model(inp, out)

    dynamic = du.list_tflite_ops(du.convert_to_tflite(model, _representative, profile="default"))
    fixed_bytes = du.convert_to_tflite(model, _representative, profile="artpec9")
    fixed = du.list_tflite_ops(fixed_bytes)
    assert "SHAPE" in dynamic
    assert not {"SHAPE", "PACK", "STRIDED_SLICE"} & set(fixed)
    assert du.tflite_outputs(fixed_bytes)[0]["name"] == "pairs"
    assert du.tflite_outputs(fixed_bytes)[0]["shape"] == (1, 2, 3)

    still_dynamic = du.list_tflite_ops(du.convert_to_tflite(model, _representative, profile="artpec9", static_batch=0))
    assert "SHAPE" in still_dynamic
    assert du.get_profile("artpec8").static_batch == 1 and du.get_profile("default").static_batch is None


# --------------------------------------------------------------------------- #
# Post-training quantization strategies: folding, bias correction, error report

def _conv_bn_model():
    """conv-BN-ReLU twice, a pre-activation BN that cannot be folded, and a
    nested sub-model with its own conv-BN."""
    tf.keras.utils.set_random_seed(0)
    inner_in = tf.keras.Input((SIZE // 2, SIZE // 2, 8))
    inner = L.BatchNormalization(name="inner_bn")(L.Conv2D(8, 3, padding="same", use_bias=False, name="inner_conv")(inner_in))
    nested = tf.keras.Model(inner_in, L.ReLU()(inner), name="nested")

    inp = tf.keras.Input((SIZE, SIZE, 3))
    x = L.Conv2D(8, 3, strides=2, padding="same", use_bias=False, name="c1")(inp)
    x = L.ReLU()(L.BatchNormalization(name="bn1")(x))
    x = nested(x)
    x = L.DepthwiseConv2D(3, padding="same", use_bias=True, name="dw")(x)
    x = L.ReLU()(L.BatchNormalization(name="bn_dw")(x))
    x = L.ReLU()(L.BatchNormalization(name="pre_bn")(x))          # follows a ReLU: not foldable
    model = tf.keras.Model(inp, L.Conv2D(4, 1, name="out")(x))
    rng = np.random.default_rng(0)
    for layer in list(model.layers) + list(nested.layers):
        if isinstance(layer, L.BatchNormalization):
            shape = layer.get_weights()[0].shape
            layer.set_weights([rng.uniform(0.5, 1.5, shape), rng.normal(0, 0.3, shape),
                               rng.normal(0, 0.3, shape), rng.uniform(0.5, 1.5, shape)])
    return model


def test_fold_batch_norms_preserves_the_function_and_leaves_unfoldable_ones():
    model = _conv_bn_model()
    folded = du.fold_batch_norms(model)

    x = np.random.default_rng(1).random((2, SIZE, SIZE, 3), dtype=np.float32)
    np.testing.assert_allclose(folded.predict(x, verbose=0), model.predict(x, verbose=0), rtol=1e-4, atol=1e-5)

    def batch_norms(m):
        return [l.name for l in m.layers if isinstance(l, L.BatchNormalization)]
    assert batch_norms(folded) == ["pre_bn"]
    assert batch_norms(folded.get_layer("nested")) == []
    assert folded.get_layer("c1").use_bias and folded.get_layer("nested").get_layer("inner_conv").use_bias
    assert not model.get_layer("c1").use_bias                       # the original is untouched
    assert du.fold_batch_norms(_toy_model()) is not None
    with pytest.raises(ValueError, match="strip_qat"):
        du.fold_batch_norms(_toy_qat_model())


def test_correct_bias_restores_the_mean_output_of_each_quantized_layer():
    """After the correction, a layer with its weights rounded to 8 bits has
    the same mean output per channel as the float layer, on the
    calibration data. Checked on the first convolution, per tensor.
    """
    from swiss_army_keras.deploy_utils import _quantized_weights

    model = du.fold_batch_norms(_conv_bn_model())
    calibration = np.random.default_rng(2).random((8, SIZE, SIZE, 3), dtype=np.float32)
    layer = model.get_layer("c1")
    kernel, bias = [w.copy() for w in layer.get_weights()]
    quantized = _quantized_weights(kernel, per_channel=False, channel_axis=-1)

    def mean_output(k, b):
        out = tf.nn.conv2d(calibration, k, strides=2, padding="SAME") + b
        return out.numpy().reshape(-1, out.shape[-1]).mean(axis=0)

    assert np.abs(mean_output(quantized, bias) - mean_output(kernel, bias)).max() > 1e-4      # there is a shift
    result = du.correct_bias(model, calibration, per_channel=False)
    corrected_bias = layer.get_weights()[1]
    np.testing.assert_allclose(mean_output(quantized, corrected_bias), mean_output(kernel, bias), atol=1e-5)
    assert result.layers == 4 and result.skipped == 0 and result.max_shift > 0

    result.restore()
    np.testing.assert_array_equal(layer.get_weights()[1], bias)
    # layers without a bias are skipped, not failed
    unfolded = du.correct_bias(_conv_bn_model(), calibration)
    assert unfolded.skipped == 2 and unfolded.layers == 2
    with pytest.raises(ValueError, match="strip_qat"):
        du.correct_bias(_toy_qat_model(), calibration)


def test_quantized_weight_rounding_matches_the_converter_convention():
    from swiss_army_keras.deploy_utils import _quantized_weights

    kernel = np.random.default_rng(0).normal(size=(3, 3, 4, 6)).astype("float32")
    kernel[..., 0] *= 20.0                                            # one loud channel
    per_tensor = _quantized_weights(kernel, False, -1)
    per_channel = _quantized_weights(kernel, True, -1)
    step = np.abs(kernel).max() / 127.0
    np.testing.assert_allclose(per_tensor / step, np.round(per_tensor / step), atol=1e-4)
    assert np.abs(per_tensor - kernel).max() <= step / 2 + 1e-6
    # per channel, the quiet channels are rounded much more finely
    assert np.abs(per_channel[..., 1:] - kernel[..., 1:]).max() < np.abs(per_tensor[..., 1:] - kernel[..., 1:]).max()


def test_quantization_debug_report_ranks_tensors_by_relative_error():
    report = du.quantization_debug_report(_toy_model(), _representative, profile="artpec8")
    assert len(report) >= 2
    assert {"op_name", "tensor_name", "scale", "rmse", "rmse_over_scale", "max_abs_error"} <= set(report[0])
    ratios = [row["rmse_over_scale"] for row in report]
    assert ratios == sorted(ratios, reverse=True)
    assert all(row["scale"] > 0 and np.isfinite(row["rmse"]) for row in report)
    assert len(du.quantization_debug_report(_toy_model(), _representative, top=1)) == 1


# --------------------------------------------------------------------------- #
# QAT switches

def test_builders_can_skip_qat_and_the_switch_is_restored():
    from swiss_army_keras.layer_utils import qat_enabled, qat_options, set_qat

    assert qat_enabled()
    with qat_options(enabled=False):
        assert not qat_enabled()
        plain = lr_aspp(tf.keras.Input((64, 64, 3)), n_labels=3, backbone="MobileNetV2", weights=None)
    assert qat_enabled()
    wrapped = lr_aspp(tf.keras.Input((64, 64, 3)), n_labels=3, backbone="MobileNetV2", weights=None)

    assert not du.has_qat_layers(plain) and du.has_qat_layers(wrapped)
    assert "lr_aspp_head" in {layer.name for layer in plain.layers}
    assert "quant_lr_aspp_head" in {layer.name for layer in wrapped.layers}
    assert plain.output_shape == wrapped.output_shape and callable(plain.preprocessing)
    assert du.strip_qat(plain) is plain

    previous = set_qat(enabled=False)
    assert previous == {"enabled": True} and not qat_enabled()
    set_qat(**previous)
    assert qat_enabled()


def test_quantizer_strategy_defaults_follow_the_profile():
    from swiss_army_keras.quantization_utils import Quantizer

    def options(profile, **kwargs):
        q = Quantizer(None, _toy_model(), "unused", append_datetime=False, profile=profile, **kwargs)
        return q.strip_qat, q.fold_batch_norms, q.equalize, q.bias_correction, q.replace_global_pooling

    assert options("artpec8") == (True, True, True, True, True)
    assert options("artpec9") == (True, True, False, False, False)
    assert options("default") == (False, False, False, False, False)
    assert options("artpec8", bias_correction=False, fold_batch_norms=False)[1::2] == (False, False)
