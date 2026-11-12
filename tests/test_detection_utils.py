"""The NumPy reference decoder for detectors exported without decoding and
NMS, against the decoding the full detectors do inside their graph."""

import json

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras import deploy_utils, detection_utils
from swiss_army_keras._model_detection import (
    fcos_lite_detection, fcos_lite_detection_base, gfl_lite_detection, gfl_lite_detection_base)
from swiss_army_keras.quantization_utils import Quantizer

SIZE = 128


def _images(count=1, seed=0):
    return np.random.default_rng(seed).random((count, SIZE, SIZE, 3)).astype(np.float32)


def test_nms_keeps_the_best_box_of_each_cluster():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 9]], np.float32)
    scores = np.array([0.6, 0.9, 0.3, 0.8], np.float32)
    assert detection_utils.nms(boxes, scores, iou_threshold=0.5).tolist() == [1, 2]
    assert detection_utils.nms(boxes, scores, iou_threshold=0.5, max_detections=1).tolist() == [1]
    # a threshold above every overlap keeps all boxes, by decreasing score
    assert detection_utils.nms(boxes, scores, iou_threshold=0.95).tolist() == [1, 3, 0, 2]


def test_dequantize_and_anchor_points():
    assert detection_utils.dequantize(np.array([0, 128, 255], np.uint8), 0.5, 128).tolist() == [-64.0, 0.0, 63.5]
    assert detection_utils.dequantize(np.array([1.5], np.float32), 0.0, 0).tolist() == [1.5]
    points = detection_utils.anchor_points(2, 3, 8)
    assert points.tolist() == [[4, 4], [12, 4], [20, 4], [4, 12], [12, 12], [20, 12]]


@pytest.mark.parametrize("kind", ["fcos", "gfl"])
def test_reference_decoder_matches_the_decoding_inside_the_full_detector(kind):
    tf.keras.utils.set_random_seed(0)
    options = dict(backbone="ResNet18", weights=None, freeze_backbone=False, score_threshold=0.3,
                   iou_threshold=0.5, max_detections=50)
    if kind == "fcos":
        full = fcos_lite_detection(tf.keras.Input((SIZE, SIZE, 3)), 3, distance_activation="relu", **options)
    else:
        full = gfl_lite_detection(tf.keras.Input((SIZE, SIZE, 3)), 3, flat_reg=True, **options)
    base = full.base_model
    assert base.detection["kind"] == kind

    image = _images()
    boxes, scores, classes, valid = [np.asarray(v) for v in full.predict(image, verbose=0)]
    count = int(valid[0])
    assert count > 0

    spec = deploy_utils.detection_export_spec(base)
    assert json.loads(json.dumps(spec)) == spec
    assert spec["input_size"] == [SIZE, SIZE] and spec["strides"] == list(base.strides)
    detections = detection_utils.decode_detections(
        base.predict(image, verbose=0), spec, score_threshold=0.3, iou_threshold=0.5, max_detections=50)

    assert len(detections["scores"]) == count
    np.testing.assert_allclose(detections["scores"], scores[0, :count], atol=1e-4)
    np.testing.assert_allclose(detections["boxes"], boxes[0, :count], atol=1e-2)
    assert detections["classes"].tolist() == classes[0, :count].astype(int).tolist()


def test_decoder_refuses_outputs_that_do_not_fit_the_spec():
    base = gfl_lite_detection_base(tf.keras.Input((SIZE, SIZE, 3)), 2, backbone="ResNet18", weights=None,
                                   flat_reg=True)
    spec = deploy_utils.detection_export_spec(base)
    outputs = base.predict(_images(2), verbose=0)
    with pytest.raises(ValueError, match="one image"):
        detection_utils.decode_detections(outputs, spec)
    with pytest.raises(ValueError, match="output tensors"):
        detection_utils.decode_detections(outputs[:-1], spec)
    with pytest.raises(ValueError, match="not a base detector"):
        deploy_utils.detection_export_spec(tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))]))
    empty = detection_utils.decode_detections([o[:1] for o in outputs], spec, score_threshold=1.0)
    assert empty["boxes"].shape == (0, 4) and empty["classes"].shape == (0,)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["fcos", "gfl"])
def test_exported_base_detector_decodes_like_its_keras_model(kind, tmp_path):
    tf.keras.utils.set_random_seed(0)
    if kind == "fcos":
        base = fcos_lite_detection_base(tf.keras.Input((SIZE, SIZE, 3)), 3, backbone="ResNet18", weights=None,
                                        distance_activation="relu")
    else:
        base = gfl_lite_detection_base(tf.keras.Input((SIZE, SIZE, 3)), 3, backbone="ResNet18", weights=None,
                                       flat_reg=True)
    images = _images(8)
    dataset = tf.data.Dataset.from_tensor_slices((images, np.zeros(8, np.float32))).batch(4)
    quantizer = Quantizer(dataset, base, str(tmp_path / kind), append_datetime=False, batches=2, profile="artpec8")
    quantizer.quantize()
    assert not quantizer.lint_report.errors

    spec = detection_utils.load_detection_spec(quantizer.detection_spec_path)
    assert spec == json.loads(json.dumps(quantizer.detection_spec))
    assert sorted((o["role"], o["level"]) for o in spec["outputs"]) == sorted(
        (role, level) for level in range(3) for role in base.detection["roles"])

    interpreter = tf.lite.Interpreter(model_content=quantizer.tflite_ui8_model)
    interpreter.allocate_tensors()
    entry = interpreter.get_input_details()[0]
    scale, zero_point = entry["quantization"]
    interpreter.set_tensor(entry["index"], np.round(images[:1] / scale + zero_point).astype(entry["dtype"]))
    interpreter.invoke()
    raw = [interpreter.get_tensor(d["index"]) for d in interpreter.get_output_details()]
    assert all(r.dtype == np.uint8 for r in raw)

    boxes, scores = detection_utils.decode_candidates(raw, spec)
    reference = quantizer.export_model
    expected_boxes, expected_scores = detection_utils.decode_candidates(
        reference.predict(images[:1], verbose=0), deploy_utils.detection_export_spec(reference))
    assert boxes.shape == expected_boxes.shape == (16 * 16 + 8 * 8 + 4 * 4, 4)
    # 8-bit outputs of an untrained network: close, not equal
    assert np.abs(scores - expected_scores).mean() < 0.02
    assert np.abs(boxes - expected_boxes).mean() < 2.0
