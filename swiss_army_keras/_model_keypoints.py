'''Keypoint localization by coordinate classification (SimCC).

Li et al., 2022, "SimCC: A Simple Coordinate Classification Perspective for
Human Pose Estimation" (ECCV), the formulation RTMPose (Jiang et al., 2023)
builds on. Each keypoint is predicted as two 1-D classifications, one over
horizontal bins and one over vertical bins, instead of a 2-D heatmap or a
direct regression. For an accelerator this has two advantages: there is no
high-resolution heatmap to compute and quantize, and the outputs are plain
logits whose argmax is taken by the application, so 8-bit quantization only
has to preserve which bin is largest.

The head is top-down: it expects a crop around one subject (from a detector
or a fixed region of the camera view) and returns one location per
keypoint. `pose_regression_head` remains the cheaper, less accurate
alternative.

Structure, all in operators that the Axis accelerator lists contain
(convolutions, reshape, transpose):

1. a `final_kernel` x `final_kernel` convolution turns the deepest backbone
   feature into one map per keypoint;
2. each keypoint map is flattened into a vector and passed through a shared
   hidden layer;
3. two shared linear layers turn that representation into the horizontal
   and the vertical bin logits.

Steps 2 and 3 are 1x1 convolutions on a `(keypoints, 1, features)` tensor,
which is how a layer shared across keypoints is expressed without a
fully-connected operator on 3-D tensors. RTMPose's self-attention block
between steps 2 and 3 is left out.
'''

from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from swiss_army_keras.layer_utils import check_input_size, qat_annotate_layer, qat_apply
from swiss_army_keras._backbone_zoo import backbone_zoo, backbone_num_levels

from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Permute, Reshape
from tensorflow.keras.models import Model

import tensorflow_model_optimization as tfmot


def simcc_head(input_tensor, n_keypoints, split_ratio=2.0, hidden=256, final_kernel=7, feature_level=None,
               backbone='MobileNetV4-Conv-S', weights='imagenet', freeze_backbone=True, freeze_batch_norm=True,
               name='simcc'):
    '''Top-down keypoint head with coordinate classification.

    Input
    ----------
        input_tensor: the input tensor (a crop around one subject), with a
                      static height and width that are multiples of 32,
                      e.g. `keras.layers.Input((256, 192, 3))`.
        n_keypoints: number of keypoints.
        split_ratio: bins per input pixel. 2.0 (the SimCC and RTMPose
                     default) gives half-pixel resolution: `2 * width`
                     horizontal and `2 * height` vertical bins.
        hidden: width of the per-keypoint hidden representation.
        final_kernel: kernel size of the convolution that produces the
                      keypoint maps (7 in RTMPose; 3 is cheaper).
        feature_level: 1-indexed backbone level to use, or `None` for the
                       deepest.
        backbone, weights, freeze_backbone, freeze_batch_norm: as every
            other `backbone_zoo`-based model of the library.
        name: prefix of the created layers.

    Output
    ----------
        A `tf.keras.Model` with a `.preprocessing` attribute and two
        outputs: `x_logits` of shape `(batch, n_keypoints, x_bins)` and
        `y_logits` of shape `(batch, n_keypoints, y_bins)`. The attributes
        `split_ratio`, `x_bins` and `y_bins` are set on the model. Train
        with `losses.simcc_loss` against targets from `simcc_encode`;
        decode with `simcc_decode`.
    '''
    height, width = check_input_size(input_tensor, multiple=32, what='simcc_head')
    n_levels = backbone_num_levels(backbone)
    feature_level = n_levels if feature_level is None else feature_level
    if not 1 <= feature_level <= n_levels:
        raise ValueError(f'feature_level must be between 1 and {n_levels} for this backbone, got {feature_level}.')
    x_bins, y_bins = int(round(width * split_ratio)), int(round(height * split_ratio))

    backbone_ = backbone_zoo(backbone, weights, input_tensor, feature_level, freeze_backbone, freeze_batch_norm)
    features = backbone_([input_tensor, ])
    feature = features[feature_level - 1] if isinstance(features, (list, tuple)) else features

    maps = qat_annotate_layer(Conv2D(
        n_keypoints, final_kernel, padding='same', use_bias=False, name=f'{name}_keypoint_maps'))(feature)
    maps = qat_annotate_layer(BatchNormalization(name=f'{name}_keypoint_maps_bn'))(maps)
    maps = qat_annotate_layer(Activation('relu', name=f'{name}_keypoint_maps_relu'))(maps)

    cells = int(maps.shape[1]) * int(maps.shape[2])
    # (h, w, K) -> (h*w, K) -> (K, h*w) -> (K, 1, h*w): one row per keypoint
    tokens = Reshape((cells, n_keypoints), name=f'{name}_flatten')(maps)
    tokens = Permute((2, 1), name=f'{name}_per_keypoint')(tokens)
    tokens = Reshape((n_keypoints, 1, cells), name=f'{name}_tokens')(tokens)

    tokens = qat_annotate_layer(Conv2D(hidden, 1, use_bias=False, name=f'{name}_hidden'))(tokens)
    tokens = qat_annotate_layer(BatchNormalization(name=f'{name}_hidden_bn'))(tokens)
    tokens = qat_annotate_layer(Activation('relu', name=f'{name}_hidden_relu'))(tokens)

    x_logits = qat_annotate_layer(Conv2D(x_bins, 1, name=f'{name}_x_bins'))(tokens)
    y_logits = qat_annotate_layer(Conv2D(y_bins, 1, name=f'{name}_y_bins'))(tokens)
    x_logits = Reshape((n_keypoints, x_bins), name='x_logits')(x_logits)
    y_logits = Reshape((n_keypoints, y_bins), name='y_logits')(y_logits)

    m = Model([input_tensor, ], [x_logits, y_logits])

    with tfmot.quantization.keras.quantize_scope():
        m = qat_apply(m)

    m.preprocessing = backbone_.preprocessing
    m.split_ratio = split_ratio
    m.x_bins, m.y_bins = x_bins, y_bins

    return m


def simcc_encode(keypoints, visible, input_size, split_ratio=2.0, sigma=6.0):
    '''Training targets for `simcc_head`: one Gaussian-smoothed distribution
    over bins per keypoint and axis.

    Input
    ----------
        keypoints: `(..., n_keypoints, 2)` array of `(x, y)` positions in
                   input pixels.
        visible: `(..., n_keypoints)` array, non-zero for keypoints that
                 are labelled. The distributions of the others are all
                 zero, which `losses.simcc_loss` treats as "no target".
        input_size: `(height, width)` of the network input.
        split_ratio: bins per pixel, as given to `simcc_head`.
        sigma: standard deviation of the target, in bins (6.0 is the value
               RTMPose uses with a split ratio of 2).

    Output
    ----------
        `(x_target, y_target)`: float32 arrays of shape
        `(..., n_keypoints, x_bins)` and `(..., n_keypoints, y_bins)`,
        each labelled row summing to 1.
    '''
    keypoints = np.asarray(keypoints, dtype=np.float32)
    visible = (np.asarray(visible) > 0).astype(np.float32)
    height, width = input_size
    targets = []
    for axis, extent in ((0, width), (1, height)):
        bins = int(round(extent * split_ratio))
        centres = keypoints[..., axis:axis + 1] * split_ratio
        grid = np.arange(bins, dtype=np.float32)
        target = np.exp(-0.5 * ((grid - centres) / sigma) ** 2)
        target = target / np.maximum(target.sum(axis=-1, keepdims=True), 1e-12)
        targets.append((target * visible[..., None]).astype(np.float32))
    return targets[0], targets[1]


def simcc_decode(x_logits, y_logits, split_ratio=2.0):
    '''Keypoint positions from the two outputs of `simcc_head`, in NumPy:
    what the application runs after inference.

    Input
    ----------
        x_logits, y_logits: arrays of shape `(..., n_keypoints, bins)`:
                            the model outputs, as floats or as the integers
                            of a quantized file (only the order of the
                            values matters for the position).
        split_ratio: bins per pixel, as given to `simcc_head`.

    Output
    ----------
        `(keypoints, scores)`: `(..., n_keypoints, 2)` positions `(x, y)`
        in input pixels, and `(..., n_keypoints)` scores, the smaller of
        the two maximum values (use them to rank or threshold keypoints
        within one model; they are logits, not probabilities).
    '''
    x_logits, y_logits = np.asarray(x_logits), np.asarray(y_logits)
    x = x_logits.argmax(axis=-1) / split_ratio
    y = y_logits.argmax(axis=-1) / split_ratio
    scores = np.minimum(x_logits.max(axis=-1), y_logits.max(axis=-1))
    return np.stack([x, y], axis=-1).astype(np.float32), scores
