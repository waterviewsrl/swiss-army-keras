"""Whole-model quantization-aware training (`deploy_utils.prepare_qat` and
the graph rewrites it is built from).

Everything here runs on small hand-built models with random weights: the
checks are structural (what gets wrapped, how weights are quantized in the
exported file) and numerical (the rewrites preserve the function; the
prepared model's simulation matches its exported file). Whether QAT
improves accuracy is a training question, measured separately (see the
deployment guide).
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras import deploy_utils as du
from swiss_army_keras.layer_utils import qat_annotate_layer, qat_apply

L = tf.keras.layers
SIZE = 32


def _inputs(n=12, seed=1):
    return np.random.default_rng(seed).random((n, SIZE, SIZE, 3), dtype=np.float32)


def _representative(inputs):
    def generator():
        for k in range(min(len(inputs), 8)):
            yield [inputs[k:k + 1]]
    return generator


def _randomize_batch_norms(layers, seed=0):
    rng = np.random.default_rng(seed)
    for layer in layers:
        inner = getattr(layer, "layer", layer)
        if isinstance(inner, L.BatchNormalization):
            shape = inner.get_weights()[0].shape
            inner.set_weights([rng.uniform(0.5, 1.5, shape), rng.normal(0, 0.3, shape),
                               rng.normal(0, 0.3, shape), rng.uniform(0.5, 1.5, shape)])


def _library_like_model():
    """A nested '<name>_backbone' (conv-BN-ReLU6, depthwise-ReLU, a
    pre-activation BN after an addition) and a QAT-wrapped head with a
    sigmoid gate (a tensor product the QAT library has no config for)."""
    tf.keras.utils.set_random_seed(0)
    backbone_in = tf.keras.Input((SIZE, SIZE, 3))
    b = L.Conv2D(8, 3, strides=2, padding="same", use_bias=False, name="bb_conv")(backbone_in)
    b = L.ReLU(6.0, name="bb_relu6")(L.BatchNormalization(name="bb_bn")(b))
    d = L.Activation("relu", name="bb_dw_relu")(L.DepthwiseConv2D(3, padding="same", name="bb_dw")(b))
    b = L.Add(name="bb_add")([b, d])
    b = L.Activation("relu", name="bb_pre_relu")(L.BatchNormalization(name="bb_pre_bn")(b))
    backbone = tf.keras.Model(backbone_in, b, name="Toy_backbone")

    inp = tf.keras.Input((SIZE, SIZE, 3))
    x = backbone(inp)
    h = qat_annotate_layer(L.Conv2D(8, 3, padding="same", use_bias=False, name="h_conv"))(x)
    h = qat_annotate_layer(L.BatchNormalization(name="h_bn"))(h)
    h = qat_annotate_layer(L.Activation("relu", name="h_relu"))(h)
    gate = qat_annotate_layer(L.Activation("sigmoid", name="h_gate"))(
        qat_annotate_layer(L.Conv2D(8, 1, name="h_gate_conv"))(h))
    h = L.Multiply(name="h_mul")([h, gate])
    h = qat_annotate_layer(L.UpSampling2D(2, interpolation="bilinear", name="h_up"))(h)
    model = qat_apply(tf.keras.Model(inp, qat_annotate_layer(L.Conv2D(3, 1, name="h_out"))(h)))
    model.preprocessing = "sentinel"
    _randomize_batch_norms(list(model.layers) + list(backbone.layers))
    return model


def _relative_error(a, b):
    return float(np.abs(a - b).mean() / (np.abs(b).mean() + 1e-12))


# --------------------------------------------------------------------------- #
# graph rewrites

def test_flatten_model_inlines_nested_models_and_keeps_the_function():
    model = du.strip_qat(_library_like_model())
    flat = du.flatten_model(model)
    assert not any(isinstance(layer, tf.keras.Model) for layer in flat.layers)
    assert "bb_conv" in {layer.name for layer in flat.layers}
    assert flat.get_layer("bb_conv") is model.get_layer("Toy_backbone").get_layer("bb_conv")   # shared
    x = _inputs()
    np.testing.assert_allclose(flat.predict(x, verbose=0), model.predict(x, verbose=0), atol=1e-6)
    assert flat.preprocessing == "sentinel"


def test_merge_activations_moves_relu_and_relu6_into_the_convolution():
    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input((SIZE, SIZE, 3))
    a = L.Activation("relu", name="a_relu")(L.Conv2D(4, 3, padding="same", name="a")(inp))
    b = L.ReLU(6.0, name="b_relu6")(L.Conv2D(4, 3, padding="same", name="b")(a))
    shared = L.Conv2D(4, 1, name="c")(b)                          # its output also feeds the model output
    out = L.Concatenate()([L.Activation("relu", name="c_relu")(shared), shared])
    model = tf.keras.Model(inp, out)

    merged = du.merge_activations(model)
    assert getattr(merged.get_layer("a").activation, "__name__") == "relu"
    assert getattr(merged.get_layer("b").activation, "__name__") == "relu6"
    assert getattr(merged.get_layer("c").activation, "__name__") == "linear"   # two consumers: left alone
    x = _inputs()
    np.testing.assert_allclose(merged.predict(x, verbose=0), model.predict(x, verbose=0), atol=1e-6)
    assert getattr(model.get_layer("a").activation, "__name__") == "linear"    # the original is untouched


def test_batch_norms_to_affine_keeps_the_function():
    model = du.flatten_model(du.fold_batch_norms(du.strip_qat(_library_like_model())))
    assert any(isinstance(layer, L.BatchNormalization) for layer in model.layers)       # the pre-activation one
    affine = du.batch_norms_to_affine(model)
    assert not any(isinstance(layer, L.BatchNormalization) for layer in affine.layers)
    assert isinstance(affine.get_layer("bb_pre_bn_affine"), L.DepthwiseConv2D)
    x = _inputs()
    np.testing.assert_allclose(affine.predict(x, verbose=0), model.predict(x, verbose=0), rtol=1e-4, atol=1e-5)


def test_untie_shared_layers_gives_each_call_its_own_copy():
    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input((SIZE, SIZE, 3))
    shared = L.Conv2D(4, 1, name="head")
    coarse = L.Conv2D(8, 3, strides=2, padding="same", name="down")(inp)
    model = tf.keras.Model(inp, [shared(L.Conv2D(8, 3, padding="same", name="fine")(inp)), shared(coarse)])

    untied = du.untie_shared_layers(model, jitter=1e-6)
    names = sorted(layer.name for layer in untied.layers if layer.name.startswith("head"))
    assert names == ["head_untied0", "head_untied1"]
    kernel = model.get_layer("head").get_weights()[0]
    np.testing.assert_array_equal(untied.get_layer("head_untied0").get_weights()[0], kernel)
    np.testing.assert_allclose(untied.get_layer("head_untied1").get_weights()[0], kernel * (1 + 1e-6), rtol=1e-9)
    x = _inputs()
    for a, b in zip(untied.predict(x, verbose=0), model.predict(x, verbose=0)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    assert du.untie_shared_layers(du.strip_qat(_library_like_model())) is not None


# --------------------------------------------------------------------------- #
# prepare_qat

@pytest.mark.parametrize("per_channel", [True, False], ids=["per_channel", "per_tensor"])
def test_prepare_qat_wraps_every_layer_and_matches_its_export(per_channel):
    model = _library_like_model()
    before = [w.copy() for w in model.get_weights()]
    x = _inputs()
    prepared = du.prepare_qat(model, x[:8], per_channel=per_channel)

    # every convolution carries a weight quantizer with the target granularity
    weighted = [layer for layer in prepared.layers
                if type(getattr(layer, "layer", None)).__name__ in ("Conv2D", "DepthwiseConv2D")]
    assert len(weighted) == 6                    # 5 convolutions + the batch norm turned depthwise
    for layer in weighted:
        shapes = [tuple(v.shape) for v in layer.weights if "kernel_min" in v.name]
        assert len(shapes) == 1
        per_axis = per_channel and type(layer.layer).__name__ == "Conv2D"
        assert (shapes[0] != ()) == per_axis, layer.name
    assert prepared.unwrapped_layers == ()
    assert "quant_h_mul" in {layer.name for layer in prepared.layers}       # output-only quantizer
    assert prepared.preprocessing == "sentinel" and prepared._sak_qat_prepared

    # the float copy is independent, and not part of what training updates
    assert all(np.array_equal(a, b) for a, b in zip(before, model.get_weights()))
    float_vars = {v.ref() for v in prepared.float_model.weights}
    assert not float_vars & {v.ref() for v in prepared.trainable_variables}
    np.testing.assert_allclose(prepared.float_model.predict(x, verbose=0),
                               du.strip_qat(model).predict(x, verbose=0), rtol=1e-4, atol=1e-5)

    # what the prepared model simulates is what the file computes
    profile = "artpec9" if per_channel else "artpec8"
    flatbuffer = du.convert_to_tflite(prepared, _representative(x), profile=profile)
    report = du.lint_tflite(flatbuffer, profile=profile)
    assert report.errors == [], str(report)
    interpreter = tf.lite.Interpreter(model_content=flatbuffer)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    exported = []
    for k in range(8, 12):
        scale, zero = inp["quantization"]
        interpreter.set_tensor(inp["index"], np.clip(np.round(x[k:k + 1] / scale + zero), 0, 255).astype(inp["dtype"]))
        interpreter.invoke()
        out_scale, out_zero = out["quantization"]
        exported.append((interpreter.get_tensor(out["index"]).astype(np.float32) - out_zero) * out_scale)
    simulated = prepared.predict(x[8:12], verbose=0)
    assert _relative_error(np.concatenate(exported), simulated) < 0.1


def test_prepare_qat_per_tensor_export_has_per_tensor_weights_even_for_shared_layers():
    """The detection heads call their output convolutions once per pyramid
    level. A shared QAT-wrapped layer is exported per channel whatever the
    converter is told; `prepare_qat` unties it.
    """
    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input((SIZE, SIZE, 3))
    fine = L.Conv2D(8, 3, padding="same", activation="relu", name="fine")(inp)
    coarse = L.Conv2D(8, 3, strides=2, padding="same", activation="relu", name="coarse")(fine)
    head = L.Conv2D(4, 1, name="head")
    model = tf.keras.Model(inp, [head(fine), head(coarse)])
    x = _inputs()

    prepared = du.prepare_qat(model, x[:8], per_channel=False)
    report = du.lint_tflite(du.convert_to_tflite(prepared, _representative(x), profile="artpec8"), profile="artpec8")
    assert report.errors == [], str(report)


def test_calibrated_ranges_have_a_minimum_width():
    """A tensor that is constant on the calibration data (a dead ReLU) would
    get a zero scale, which the LiteRT runtime refuses."""
    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input((SIZE, SIZE, 3))
    dead = L.Conv2D(4, 3, padding="same", activation="relu", kernel_initializer="zeros",
                    bias_initializer=tf.keras.initializers.Constant(-1.0), name="dead")(inp)
    model = tf.keras.Model(inp, L.Conv2D(2, 1, name="out")(L.Concatenate()([dead, inp])))
    x = _inputs()
    prepared = du.prepare_qat(model, x[:8])

    ranges = {v.name.rsplit("/", 1)[-1]: float(v.numpy()) for v in prepared.get_layer("quant_dead").weights
              if "kernel" not in v.name and ("min" in v.name or "max" in v.name)}
    low, high = [value for name, value in sorted(ranges.items()) if "min" in name][0], \
        [value for name, value in sorted(ranges.items()) if "max" in name][0]
    assert high - low >= 1e-3 - 1e-9
    flatbuffer = du.convert_to_tflite(prepared, _representative(x), profile="artpec9")
    assert du.lint_tflite(flatbuffer, profile="artpec9").errors == []
    tf.lite.Interpreter(model_content=flatbuffer).allocate_tensors()          # the runtime accepts it


def test_fine_tuning_a_prepared_model_updates_weights_and_ranges():
    model = _library_like_model()
    x = _inputs(16)
    y = tf.one_hot(np.random.default_rng(2).integers(0, 3, (16, SIZE, SIZE)), 3).numpy()
    prepared = du.prepare_qat(model, x[:8], per_channel=False)
    prepared.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))

    def snapshot(name):
        return {v.name.rsplit("/", 1)[-1]: v.numpy().copy() for v in prepared.get_layer(name).weights}

    conv_before, up_before = snapshot("quant_h_out"), snapshot("quant_h_up")
    prepared.fit(x, y, batch_size=8, epochs=1, verbose=0)
    conv_after, up_after = snapshot("quant_h_out"), snapshot("quant_h_up")
    assert np.abs(conv_after["kernel:0"] - conv_before["kernel:0"]).max() > 0
    assert any(not np.array_equal(up_after[k], up_before[k]) for k in up_before if "min" in k or "max" in k)

    report = du.lint_tflite(du.convert_to_tflite(prepared, _representative(x), profile="artpec8"), profile="artpec8")
    assert report.errors == [], str(report)


def test_prepared_weights_round_trip_and_the_quantizer_keeps_the_wrappers(tmp_path):
    from swiss_army_keras.quantization_utils import Quantizer

    model = _library_like_model()
    x = _inputs()
    prepared = du.prepare_qat(model, x[:8], per_channel=False)
    path = str(tmp_path / "prepared.weights.h5")
    prepared.save_weights(path)
    again = du.prepare_qat(model, x[:8], per_channel=False)
    again.load_weights(path)
    np.testing.assert_allclose(again.predict(x, verbose=0), prepared.predict(x, verbose=0), atol=1e-6)

    dataset = tf.data.Dataset.from_tensor_slices((x, np.zeros((len(x), 1), "float32"))).batch(4)
    assert prepared.qat_per_channel is False
    import logging
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logging.getLogger().addHandler(handler)
    try:
        Quantizer(dataset, prepared, str(tmp_path / "mismatch"), append_datetime=False, profile="artpec9")
    finally:
        logging.getLogger().removeHandler(handler)
    assert any("prepared for per-tensor weights" in record.getMessage() for record in records)

    quantizer = Quantizer(dataset, prepared, str(tmp_path / "qat"), append_datetime=False, batches=2,
                          profile="artpec8")
    assert (quantizer.strip_qat, quantizer.fold_batch_norms, quantizer.equalize, quantizer.bias_correction) == \
        (False, False, False, False)
    quantizer.quantize()
    assert quantizer.lint_report.errors == [], str(quantizer.lint_report)
    assert du.has_qat_layers(quantizer.export_model)
