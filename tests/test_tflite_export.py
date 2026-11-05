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
