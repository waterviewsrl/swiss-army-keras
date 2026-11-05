
'''LR-ASPP (Lite Reduced Atrous Spatial Pyramid Pooling), from the
MobileNetV3 paper (Howard et al., 2019, "Searching for MobileNetV3",
arXiv:1905.02244, §5). The cheapest of the three decoders added under
plan.md §3: one 1x1 branch plus one global-average-pool branch (not a
dilated-conv ASPP stack at all, unlike `deeplab_v3_plus`'s
`DilatedSpatialPyramidPooling`), the pooled branch acting as a
sigmoid-gated *weight* on the 1x1 branch rather than a separate parallel
feature to concatenate.

Follows the `deeplab_v3_plus` precedent (`_model_deeplab_v3_plus.py`) for
encoder wiring via `backbone_zoo` and `tfmot.quantization.keras.quantize_annotate_layer`
wrapping for QAT parity — including its `backbone=None` no-backbone
fallback path (a plain UNET-style stacked-conv encoder), so this decoder
can be exercised and tested the same way without requiring a backbone.
Only needs 2 feature levels (one deep, one shallow), unlike
`deeplab_v3_plus`'s use of the full pyramid depth for its dilated
convolutions.
'''

from __future__ import absolute_import

from swiss_army_keras.layer_utils import *
from swiss_army_keras._backbone_zoo import backbone_zoo, backbone_num_levels
from swiss_army_keras._model_unet_2d import UNET_left

from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Multiply,
    UpSampling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal

from tensorflow.nn import relu

import tensorflow as tf
import tensorflow_model_optimization as tfmot


def _conv_bn_act(x, num_filters, kernel_size=1, use_bias=False, activation=relu, name=''):
    x = qat_annotate_layer(Conv2D(
        num_filters, kernel_size, padding='same', use_bias=use_bias,
        kernel_initializer=HeNormal(), name=f'{name}_conv',
    ))(x)
    x = qat_annotate_layer(BatchNormalization(name=f'{name}_bn'))(x)
    if activation is not None:
        x = qat_annotate_layer(Activation(activation, name=f'{name}_act'))(x)
    return x


def lr_aspp_module(deep_feature, num_filters=128, name='lr_aspp_module', gate=True):
    '''The LR-ASPP branch itself: `1x1 Conv+BN+ReLU` gated by
    `sigmoid(1x1 Conv(GlobalAvgPool))`, upsampled back to the deep
    feature's own spatial size before the elementwise gate (a `Multiply`,
    not a `Concatenate` — the pooled branch reweights the 1x1 branch,
    it doesn't add a new set of channels).
    '''
    dims = deep_feature.shape

    branch_1x1 = _conv_bn_act(deep_feature, num_filters, kernel_size=1, name=f'{name}_branch1x1')
    if not gate:
        # Gate-free variant: no global pooling, sigmoid or tensor-by-tensor
        # multiplication, which several NPUs do not accelerate.
        return branch_1x1

    pooled = qat_annotate_layer(AveragePooling2D(
        pool_size=(dims[-3], dims[-2]), name=f'{name}_gap',
    ))(deep_feature)
    pooled = qat_annotate_layer(Conv2D(
        num_filters, 1, padding='same', use_bias=True, name=f'{name}_gap_conv',
    ))(pooled)
    pooled = qat_annotate_layer(Activation('sigmoid', name=f'{name}_gap_sigmoid'))(pooled)
    pooled = qat_annotate_layer(UpSampling2D(
        size=(dims[-3] // pooled.shape[1], dims[-2] // pooled.shape[2]),
        interpolation='bilinear', name=f'{name}_gap_upsample',
    ))(pooled)

    # `Multiply` (like `Concatenate` elsewhere in this codebase's decoders,
    # e.g. `deeplab_v3_plus`'s `DilatedSpatialPyramidPooling`) is a merge
    # layer `tfmot`'s default QAT scheme doesn't support annotating directly
    # (`quantize_apply` raises `RuntimeError: ... is not supported` for it) -
    # left unannotated/unquantized, same as those `Concatenate` calls.
    return Multiply(name=f'{name}_gate')([branch_1x1, pooled])


def resolve_lr_aspp_levels(n_levels, deep_layer=None, shallow_layer=None, what='the encoder'):
    '''Resolve the deep and shallow feature levels of an LR-ASPP decoder.

    Input
    ----------
        n_levels: number of feature levels the encoder exposes.
        deep_layer: 1-indexed level feeding the LR-ASPP module, or `None`
                    for the deepest one.
        shallow_layer: 1-indexed level of the skip connection, or `None`
                       for three levels above the deep one (never above the
                       first). On five-level and four-level backbones this
                       is the stride-4 feature.
        what: name used in error messages.

    Output
    ----------
        `(deep_layer, shallow_layer)` as ints. Raises `ValueError` when a
        requested level does not exist.
    '''
    if deep_layer is None:
        deep_layer = n_levels
    if shallow_layer is None:
        shallow_layer = max(1, deep_layer - 3)
    if not 1 <= deep_layer <= n_levels:
        raise ValueError(
            f'{what} exposes {n_levels} feature levels, got deep_layer={deep_layer}. '
            f'Pass a value between 1 and {n_levels}, or None for the deepest level.')
    if not 1 <= shallow_layer <= deep_layer:
        raise ValueError(
            f'shallow_layer must be between 1 and deep_layer ({deep_layer}), got {shallow_layer}.')
    return deep_layer, shallow_layer


def lr_aspp_head(features, input_tensor, n_labels, deep_layer=None, shallow_layer=None,
                 num_filters=128, num_filters_shallow=48, name='lr_aspp', gate=True):
    '''The LR-ASPP decoder on an existing list of encoder features.

    This is the part of `lr_aspp` that follows the encoder. It is exposed so
    that several heads can share one backbone (see `multi_task`). The layers
    are annotated for quantization-aware training but `quantize_apply` is
    left to the caller.

    Input
    ----------
        features: list of encoder feature tensors, shallowest first.
        input_tensor: the model input, used for the output resolution.
        n_labels: number of output channels.
        deep_layer, shallow_layer: see `resolve_lr_aspp_levels`.
        num_filters, num_filters_shallow, name, gate: see `lr_aspp`.

    Output
    ----------
        `(logits, deep_feature)`: the `(H, W, n_labels)` output at the input
        resolution and the encoder feature the module was built on.
    '''
    deep_layer, shallow_layer = resolve_lr_aspp_levels(len(features), deep_layer, shallow_layer)

    deep_feature = features[deep_layer - 1]
    shallow_feature = features[shallow_layer - 1]

    x = lr_aspp_module(deep_feature, num_filters=num_filters, name='{}_module'.format(name), gate=gate)

    x = qat_annotate_layer(UpSampling2D(
        size=(shallow_feature.shape[1] // x.shape[1],
              shallow_feature.shape[2] // x.shape[2]),
        interpolation='bilinear', name='{}_upsample_to_shallow'.format(name),
    ))(x)

    shallow_projected = _conv_bn_act(
        shallow_feature, num_filters_shallow, kernel_size=1, activation=None,
        name='{}_shallow_proj'.format(name),
    )

    x = Concatenate(axis=-1, name='{}_concat'.format(name))([x, shallow_projected])

    x = qat_annotate_layer(UpSampling2D(
        size=(input_tensor.shape[1] // x.shape[1],
              input_tensor.shape[2] // x.shape[2]),
        interpolation='bilinear', name='{}_upsample_to_input'.format(name),
    ))(x)

    logits = qat_annotate_layer(
        Conv2D(n_labels, kernel_size=1, padding='same', name='{}_head'.format(name)))(x)

    return logits, deep_feature


def lr_aspp(input_tensor, n_labels, filter_num_down=[64, 128, 256, 512, 1024],
            deep_layer=None, shallow_layer=None, num_filters=128, num_filters_shallow=48,
            stack_num_down=2, activation='ReLU', batch_norm=False, pool=True,
            backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True,
            name='lr_aspp', iqa_head=False, gate=True):
    '''LR-ASPP (Lite Reduced ASPP) segmentation decoder, from the
    MobileNetV3 paper (Howard et al., 2019).

    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Input((None, None, 3))`.
        n_labels: number of segmentation classes.
        filter_num_down: per-level filter counts for the no-backbone encoder
                         fallback (`backbone=None`); ignored otherwise. The
                         network depth for that fallback is `len(filter_num_down)`.
        deep_layer: which encoder feature level (1-indexed) feeds the LR-ASPP
                    module. `None` (default) uses the deepest level the
                    encoder exposes: 5 for most backbones, 4 for RepViT and
                    FastViT, 3 for StrideFormer.
        shallow_layer: which encoder feature level (1-indexed) is the shallow
                       skip connection concatenated back in. `None` (default)
                       uses three levels above the deep one, never above the
                       first: level 2 on five-level backbones, level 1 on
                       the others.
        num_filters: channel count of the LR-ASPP module's 1x1/pooled branches.
        num_filters_shallow: channel count the shallow skip is projected to
                             before concatenation.
        stack_num_down, activation, batch_norm, pool: passed through to the
            no-backbone (`backbone=None`) encoder fallback (`CONV_stack`/
            `UNET_left`) — see `deeplab_v3_plus` for the identical usage.
        backbone: `None` (default; plain stacked-conv encoder), or the name of
                  a backbone registered in `swiss_army_keras._backbone_zoo.BACKBONE_REGISTRY`
                  (`tf.keras.applications` names, or any of MobileOne/
                  MobileNetV4/FastViT/RepViT's custom entries).
        weights: one of `None`, `'imagenet'`, or a path to a weights file.
        freeze_backbone: `True` to freeze the backbone's weights.
        freeze_batch_norm: `True` to also freeze the backbone's BN layers.
        name: prefix of the created Keras model and its layers.
        iqa_head (plan.md §13.7): `False` (default) — unchanged behavior,
            a single `(H, W, n_labels)` output, exactly as before. `True` —
            adds a small `swiss_army_keras.layer_utils.iqa_regression_head`
            side branch tapped off `deep_feature` (this decoder's own
            deepest encoder tap, before the LR-ASPP module consumes it),
            trained simultaneously with the main segmentation task on the
            same shared backbone (e.g. via
            `swiss_army_keras.iqa_utils.add_teacher_iqa_labels`) — the
            exported model carries both heads off one shared encoder.
            Appends its `[0, 1]` score as a second output:
            `[logit_map, iqa_score]`.
        gate: `True` (default) for the original LR-ASPP module, in which
            the 1x1 branch is multiplied by a sigmoid gate computed from
            the globally pooled feature. `False` drops the gate, leaving
            only convolutions, resizes and a concatenation in the exported
            graph: use it for accelerators that do not run global pooling,
            sigmoid or tensor-by-tensor multiplication (see the deployment
            guide). The gate-free model is a different architecture and has
            to be trained as such.

    Output
    ----------
        A `tf.keras.Model` with a `.preprocessing` attribute (the backbone's
        expected input preprocessing function, or `dummy_preprocessing` for
        `backbone=None`), producing an `(H, W, n_labels)` logit map at the
        input's spatial resolution (plus, if `iqa_head=True`, a second
        `(batch, 1)` quality-score output).
    '''

    X_encoder = []

    if backbone is None:
        deep_layer, shallow_layer = resolve_lr_aspp_levels(
            len(filter_num_down), deep_layer, shallow_layer, what='The plain encoder')

        X = input_tensor
        X = CONV_stack(X, filter_num_down[0], kernel_size=3, stack_num=stack_num_down,
                       activation=activation, batch_norm=batch_norm, name='{}_down0'.format(name))
        X_encoder.append(X)

        for i, f in enumerate(filter_num_down[1:]):
            X = UNET_left(X, f, kernel_size=3, stack_num=stack_num_down, activation=activation,
                          pool=pool, batch_norm=batch_norm, name='{}_down{}'.format(name, i + 1))
            X_encoder.append(X)

        preprocessing = dummy_preprocessing
    else:
        deep_layer, shallow_layer = resolve_lr_aspp_levels(
            backbone_num_levels(backbone), deep_layer, shallow_layer, what=f"Backbone '{backbone}'")

        backbone_ = backbone_zoo(
            backbone, weights, input_tensor, deep_layer, freeze_backbone, freeze_batch_norm)
        X_encoder = backbone_([input_tensor, ])
        preprocessing = backbone_.preprocessing

    model_output, deep_feature = lr_aspp_head(
        X_encoder, input_tensor, n_labels, deep_layer=deep_layer, shallow_layer=shallow_layer,
        num_filters=num_filters, num_filters_shallow=num_filters_shallow, name=name, gate=gate)

    outputs = [model_output, ]
    if iqa_head:
        iqa_score = iqa_regression_head(deep_feature, name='{}_iqa'.format(name))
        outputs.append(iqa_score)

    m = Model([input_tensor, ], outputs)

    with tfmot.quantization.keras.quantize_scope():
        m = tfmot.quantization.keras.quantize_apply(m)

    m.preprocessing = preprocessing

    return m
