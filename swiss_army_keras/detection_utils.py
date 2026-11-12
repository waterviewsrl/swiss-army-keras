'''Reference decoder for detectors exported without decoding and NMS.

An accelerator runs convolutions; box decoding and non-maximum suppression
are better done by the application on the CPU. The `*_base` detectors
(`fcos_lite_detection_base`, `gfl_lite_detection_base`) export exactly the
convolutional part, and this module turns their raw outputs into boxes.

It needs NumPy only and does not import TensorFlow, so the file can be
copied to a device as it is. The workflow:

1. on the workstation, export the base model and describe the file::

       quantizer = Quantizer(dataset, base_model, 'detector', profile='artpec8')
       quantizer.quantize()            # also writes detector_quant_ui8_detection.json
       spec = quantizer.detection_spec

   (or `deploy_utils.detection_export_spec(exported_keras_model, tflite_bytes)`);

2. on the device, run the file, collect its outputs in the order the
   runtime returns them, and decode::

       spec = load_detection_spec('detector_quant_ui8_detection.json')
       detections = decode_detections(raw_outputs, spec, score_threshold=0.3)

Boxes are `(x_min, y_min, x_max, y_max)` in pixels of the network input.
'''

import json

import numpy as np


def load_detection_spec(path):
    '''Read a detection spec written at export (a JSON file).'''
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def anchor_points(height, width, stride):
    '''Centres of the cells of one pyramid level, `(height * width, 2)` as
    `(x, y)` input pixels, row by row.'''
    xs = (np.arange(width, dtype=np.float32) + 0.5) * stride
    ys = (np.arange(height, dtype=np.float32) + 0.5) * stride
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)


def box_iou(box, boxes):
    '''Intersection over union of one box with an array of boxes.'''
    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])
    intersection = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - intersection
    return np.where(union > 0, intersection / np.where(union > 0, union, 1.0), 0.0)


def nms(boxes, scores, iou_threshold=0.5, max_detections=100):
    '''Greedy non-maximum suppression.

    Input
    ----------
        boxes: `(n, 4)` array, `(x_min, y_min, x_max, y_max)`.
        scores: `(n,)` array.
        iou_threshold: a box is dropped when it overlaps a kept box of
                       higher score by more than this.
        max_detections: stop after this many kept boxes.

    Output
    ----------
        Indices of the kept boxes, by decreasing score.
    '''
    order = np.argsort(-scores, kind='stable')
    kept = []
    while order.size and len(kept) < max_detections:
        best = order[0]
        kept.append(int(best))
        rest = order[1:]
        order = rest[box_iou(boxes[best], boxes[rest]) <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def dequantize(raw, scale, zero_point):
    '''Real values of a raw output tensor; a float tensor (scale 0) is
    returned as it is.'''
    raw = np.asarray(raw)
    if not scale:
        return raw.astype(np.float32)
    return (raw.astype(np.float32) - zero_point) * scale


def _level_outputs(outputs, spec):
    '''`{level: {role: float array (H, W, C)}}` from outputs in file order.'''
    if len(outputs) != len(spec['outputs']):
        raise ValueError('Expected {} output tensors, got {}.'.format(len(spec['outputs']), len(outputs)))
    levels = {}
    for raw, described in zip(outputs, spec['outputs']):
        values = dequantize(raw, described['scale'], described['zero_point'])
        height, width = described['shape'][1], described['shape'][2]
        if values.size != height * width * int(np.prod(described['shape'][3:])):
            raise ValueError("Output at position {} has {} values; the spec says shape {} (decode one image at a "
                             "time).".format(described['position'], values.size, described['shape']))
        levels.setdefault(described['level'], {})[described['role']] = values.reshape(height, width, -1)
    return levels


def decode_candidates(outputs, spec):
    '''Boxes and class scores of every location, before any filtering.

    Input
    ----------
        outputs: the output tensors of the exported file for **one image**,
                 in the order the file stores them (the order of
                 `spec['outputs']`), as raw integers or floats.
        spec: the detection spec of that file.

    Output
    ----------
        `(boxes, scores)`: `(n, 4)` boxes in input pixels and `(n,
        n_classes)` scores in 0..1, for all `n` locations of all levels.
    '''
    levels = _level_outputs(outputs, spec)
    kind = spec['kind']
    all_boxes, all_scores = [], []
    for level, stride in enumerate(spec['strides']):
        tensors = levels[level]
        height, width, _ = tensors['cls'].shape
        points = anchor_points(height, width, stride)
        cls = _sigmoid(tensors['cls'].reshape(height * width, -1))
        if kind == 'fcos':
            # the graph already outputs distances in pixels
            distances = tensors['reg'].reshape(height * width, 4)
            scores = np.sqrt(cls * _sigmoid(tensors['centerness'].reshape(height * width, 1)))
        elif kind == 'gfl':
            # one distribution over `reg_max + 1` bins per side; its mean is the distance in strides
            logits = tensors['reg'].reshape(height * width, 4, spec['reg_max'] + 1)
            logits = logits - logits.max(axis=-1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            distances = (probabilities * np.arange(spec['reg_max'] + 1, dtype=np.float32)).sum(axis=-1) * stride
            scores = cls
        else:
            raise ValueError("Unknown detector kind '{}'.".format(kind))
        all_boxes.append(np.stack([points[:, 0] - distances[:, 0], points[:, 1] - distances[:, 1],
                                   points[:, 0] + distances[:, 2], points[:, 1] + distances[:, 3]], axis=-1))
        all_scores.append(scores)
    return np.concatenate(all_boxes).astype(np.float32), np.concatenate(all_scores).astype(np.float32)


def decode_detections(outputs, spec, score_threshold=0.05, iou_threshold=0.5, max_detections=100, clip=False):
    '''Turn the raw outputs of an exported base detector into detections:
    dequantize, decode the boxes, then suppress overlaps class by class.
    The result matches what the full detector (`fcos_lite_detection`,
    `gfl_lite_detection`) computes inside its graph.

    Input
    ----------
        outputs: the output tensors for one image, in file order (see
                 `decode_candidates`).
        spec: the detection spec of the file.
        score_threshold: locations scoring less are ignored.
        iou_threshold: overlap above which the lower-scoring box of the
                       same class is dropped.
        max_detections: at most this many detections per class, and in
                       total.
        clip: clip the boxes to the image.

    Output
    ----------
        A dict with `boxes` `(n, 4)` as `(x_min, y_min, x_max, y_max)` in
        input pixels, `scores` `(n,)` and `classes` `(n,)` (int), by
        decreasing score.
    '''
    boxes, scores = decode_candidates(outputs, spec)
    if clip:
        height, width = spec['input_size']
        boxes = np.clip(boxes, 0, [width, height, width, height]).astype(np.float32)

    picked_boxes, picked_scores, picked_classes = [], [], []
    for class_index in range(scores.shape[1]):
        candidates = np.flatnonzero(scores[:, class_index] > score_threshold)
        if not candidates.size:
            continue
        kept = candidates[nms(boxes[candidates], scores[candidates, class_index], iou_threshold, max_detections)]
        picked_boxes.append(boxes[kept])
        picked_scores.append(scores[kept, class_index])
        picked_classes.append(np.full(len(kept), class_index, dtype=np.int64))

    if not picked_boxes:
        return {'boxes': np.zeros((0, 4), np.float32), 'scores': np.zeros((0,), np.float32),
                'classes': np.zeros((0,), np.int64)}
    boxes, scores, classes = np.concatenate(picked_boxes), np.concatenate(picked_scores), np.concatenate(picked_classes)
    order = np.argsort(-scores, kind='stable')[:max_detections]
    return {'boxes': boxes[order], 'scores': scores[order], 'classes': classes[order]}
