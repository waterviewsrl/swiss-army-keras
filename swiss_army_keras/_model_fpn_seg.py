'''A plain feature-pyramid head for dense prediction on accelerators.

The other segmentation decoders of the library fuse features with sigmoid
gates (LR-ASPP, PP-LiteSeg, PIDNet, PP-MobileSeg), dilated convolutions
(DeepLabV3+) or large pooling windows (DDRNet). This head uses only what a
camera accelerator runs natively: 1x1 and 3x3 convolutions with batch norm
and ReLU, additions, and resizes by exactly 2. It is the "gate-free decoder"
of `EDGE_AI_CANDIDATES.md` and the "plain pyramid semantic head" of
`CANDIDATES_BY_TASK.md`.

Structure (Lin et al., 2017, "Feature Pyramid Networks", and the semantic
head of Kirillov et al., 2019, "Panoptic Feature Pyramid Networks", reduced
to one output scale): each selected backbone level is projected to
`num_filters` channels by a 1x1 convolution; starting from the deepest, each
level is upsampled by 2 and added to the next shallower one, followed by a
3x3 convolution; the finest merged map goes through one more 3x3 convolution
and the 1x1 classifier, and is brought to the input resolution by repeated
2x bilinear resizes.

The same head serves three tasks:

- `fpn_seg`: semantic segmentation (`n_labels` logits).
- `fpn_seg(..., n_labels=1, output_activation='sigmoid')`: a dense
  regression map in `[0, 1]`.
- `density_head`: object counting by density map, a non-negative map whose
  sum over the image is the count.
'''

from __future__ import absolute_import

from swiss_army_keras.layer_utils import *
from swiss_army_keras._backbone_zoo import backbone_zoo, backbone_num_levels

from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal

import tensorflow_model_optimization as tfmot


def _conv_bn_relu(x, filters, kernel_size, name):
    x = qat_annotate_layer(Conv2D(
        filters, kernel_size, padding='same', use_bias=False, kernel_initializer=HeNormal(), name=f'{name}_conv'))(x)
    x = qat_annotate_layer(BatchNormalization(name=f'{name}_bn'))(x)
    return qat_annotate_layer(Activation('relu', name=f'{name}_relu'))(x)


def _upsample_by_two(x, target_height, target_width, name):
    '''Bring `x` to the target size with 2x bilinear resizes only. A last
    resize by another factor is added when the ratio is not a power of two.
    '''
    step = 0
    while int(x.shape[1]) * 2 <= target_height and int(x.shape[2]) * 2 <= target_width:
        x = qat_annotate_layer(UpSampling2D(size=2, interpolation='bilinear', name=f'{name}_up{step}'))(x)
        step += 1
    return resize_to(x, target_height, target_width, name=f'{name}_resize')


def fpn_head(features, input_tensor, n_labels, levels=None, num_filters=96, output_activation=None,
             name='fpn_seg'):
    '''The plain pyramid head on an existing list of encoder features.

    Exposed so that several heads can share one backbone (see `multi_task`).
    The layers are annotated for quantization-aware training but
    `quantize_apply` is left to the caller.

    Input
    ----------
        features: list of encoder feature tensors, shallowest first.
        input_tensor: the model input, used for the output resolution.
        n_labels: number of output channels.
        levels: 1-indexed feature levels to merge, shallowest first, or
                `None` for the four deepest (all of them when the backbone
                has fewer): strides 4 to 32 on five-level and four-level
                backbones. Consecutive levels must differ by a factor 2.
        num_filters: channel count of the pyramid. A multiple of 16 suits
                     Axis ARTPEC-9, a multiple of 6 ARTPEC-8; 96 is both.
        output_activation: `None` (logits), `'sigmoid'` or `'relu'`,
                           applied as the last layer so that it can be
                           dropped at export.
        name: prefix of the created layers.

    Output
    ----------
        The `(H, W, n_labels)` output at the input resolution.
    '''
    if output_activation not in (None, 'sigmoid', 'relu'):
        raise ValueError(f"Unknown output_activation '{output_activation}' - expected None, 'sigmoid' or 'relu'.")
    features = list(features)
    if levels is None:
        levels = tuple(range(max(1, len(features) - 3), len(features) + 1))
    if not levels or any(not 1 <= level <= len(features) for level in levels):
        raise ValueError(f'levels must be between 1 and {len(features)} for this backbone, got {tuple(levels)}.')
    selected = [features[level - 1] for level in levels]

    x = _conv_bn_relu(selected[-1], num_filters, 1, name=f'{name}_lateral{levels[-1]}')
    for level, feature in zip(reversed(levels[:-1]), reversed(selected[:-1])):
        if int(feature.shape[1]) != int(x.shape[1]) * 2 or int(feature.shape[2]) != int(x.shape[2]) * 2:
            raise ValueError(
                f'{name}: level {level} is {tuple(feature.shape[1:3])}, not twice the size of the level below it '
                f'{tuple(x.shape[1:3])}. Choose consecutive levels.')
        x = qat_annotate_layer(UpSampling2D(size=2, interpolation='bilinear', name=f'{name}_top_down{level}'))(x)
        lateral = _conv_bn_relu(feature, num_filters, 1, name=f'{name}_lateral{level}')
        x = Add(name=f'{name}_merge{level}')([x, lateral])
        x = _conv_bn_relu(x, num_filters, 3, name=f'{name}_smooth{level}')

    x = _conv_bn_relu(x, num_filters, 3, name=f'{name}_pre_head')
    x = qat_annotate_layer(Conv2D(n_labels, 1, padding='same', name=f'{name}_head'))(x)
    x = _upsample_by_two(x, int(input_tensor.shape[1]), int(input_tensor.shape[2]), name=name)
    if output_activation is not None:
        x = qat_annotate_layer(Activation(output_activation, name=f'{name}_{output_activation}'))(x)
    return x


def fpn_seg(input_tensor, n_labels, levels=None, num_filters=96, output_activation=None,
            backbone='MobileNetV4-Conv-S', weights='imagenet', freeze_backbone=True, freeze_batch_norm=True,
            name='fpn_seg'):
    '''Semantic segmentation with the plain pyramid head: convolutions,
    additions and 2x resizes only, for accelerators.

    With a backbone of the plain families (`ResNet18`, `STDC1`,
    `QARepVGG-A0`, `CSP-S`) the whole exported model contains no operator
    outside the published Axis ARTPEC-8 and ARTPEC-9 lists.

    Input
    ----------
        input_tensor: the input tensor, with a static height and width that
                      are multiples of 32.
        n_labels: number of segmentation classes.
        levels, num_filters, output_activation: see `fpn_head`.
        backbone: name of a backbone registered in
                  `swiss_army_keras._backbone_zoo.BACKBONE_REGISTRY`.
        weights: one of `None`, `'imagenet'`, or a path to a weights file.
        freeze_backbone: `True` to freeze the backbone's weights.
        freeze_batch_norm: `True` to also freeze the backbone's BN layers.
        name: prefix of the created layers.

    Output
    ----------
        A `tf.keras.Model` with a `.preprocessing` attribute, producing an
        `(H, W, n_labels)` map at the input resolution: logits by default.
    '''
    check_input_size(input_tensor, multiple=32, what='fpn_seg')
    backbone_ = backbone_zoo(
        backbone, weights, input_tensor, backbone_num_levels(backbone), freeze_backbone, freeze_batch_norm)
    features = backbone_([input_tensor, ])
    if not isinstance(features, (list, tuple)):
        features = [features]

    output = fpn_head(features, input_tensor, n_labels, levels=levels, num_filters=num_filters,
                      output_activation=output_activation, name=name)
    m = Model([input_tensor, ], [output, ])

    with tfmot.quantization.keras.quantize_scope():
        m = qat_apply(m)

    m.preprocessing = backbone_.preprocessing

    return m


def density_head(input_tensor, levels=None, num_filters=96, backbone='MobileNetV4-Conv-S', weights='imagenet',
                 freeze_backbone=True, freeze_batch_norm=True, name='density'):
    '''Object counting by density map: `fpn_seg` with one non-negative
    output channel.

    The network predicts a map whose sum over the image is the number of
    objects. Train it with `losses.density_count_loss` against maps made by
    `losses.points_to_density` from point annotations. At inference the
    count is the sum of the output, taken by the application; local sums
    give the count per region.

    Input
    ----------
        input_tensor, levels, num_filters, backbone, weights,
        freeze_backbone, freeze_batch_norm, name: see `fpn_seg`.

    Output
    ----------
        A `tf.keras.Model` producing an `(H, W, 1)` non-negative density
        map at the input resolution.
    '''
    return fpn_seg(input_tensor, 1, levels=levels, num_filters=num_filters, output_activation='relu',
                   backbone=backbone, weights=weights, freeze_backbone=freeze_backbone,
                   freeze_batch_norm=freeze_batch_norm, name=name)
