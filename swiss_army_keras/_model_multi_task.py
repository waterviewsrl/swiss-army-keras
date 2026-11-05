'''Several heads on one shared backbone.

Every other builder of this library creates its own backbone, so two tasks
on the same image cost two backbone passes. On an edge accelerator the
backbone is most of the inference time; `multi_task` builds it once and
attaches any number of heads to its features, so segmentation plus depth,
or classification plus an image-quality score, run in one pass and export
as one TFLite file with one output per head.

Available head types (the `type` entry of a head specification):

* `'lr_aspp'`: LR-ASPP segmentation decoder (`lr_aspp_head`). Options:
  `n_labels` (required), `gate`, `num_filters`, `num_filters_shallow`,
  `deep_layer`, `shallow_layer`.
* `'lr_aspp_depth'`: LR-ASPP decoder with one output channel. Options:
  `output_mode`, and those of `'lr_aspp'`.
* `'fastdepth'`: FastDepth decoder (`fastdepth_decoder`). Options:
  `output_mode`, `decoder_channels`.
* `'classifier'`: pooling and dense tail (`classification_head`). Options:
  `n_classes` (required), `logits` (return logits instead of
  probabilities), `pooling`, `size`, `activation`, `dropout`,
  `pre_pool_attention`, `feature_level`.
* `'iqa'`: image-quality score (`iqa_regression_head`). Options: `size`,
  `pooling`, `feature_level`.
* a callable `fn(features, input_tensor, name=..., **options)` returning one
  tensor, for a head of your own. `features` is the list of backbone
  feature maps, shallowest first.
'''

from __future__ import absolute_import

from swiss_army_keras.layer_utils import *
from swiss_army_keras._backbone_zoo import backbone_zoo, backbone_num_levels
from swiss_army_keras._model_lr_aspp import lr_aspp_head
from swiss_army_keras._model_depth import fastdepth_decoder, _valid_output_mode
from swiss_army_keras._model_classifier import classification_head

from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model

import tensorflow_model_optimization as tfmot


def _feature(features, feature_level):
    '''Pick a 1-indexed feature level; `None` is the deepest.'''
    if feature_level is None:
        return features[-1]
    if not 1 <= feature_level <= len(features):
        raise ValueError(
            f'feature_level must be between 1 and {len(features)} for this backbone, got {feature_level}.')
    return features[feature_level - 1]


def _lr_aspp_segmentation(features, input_tensor, name, n_labels, **options):
    logits, _ = lr_aspp_head(features, input_tensor, n_labels, name=name, **options)
    return logits


def _lr_aspp_depth(features, input_tensor, name, output_mode='sigmoid', **options):
    _valid_output_mode(output_mode)
    depth, _ = lr_aspp_head(features, input_tensor, 1, name=name, **options)
    if output_mode == 'sigmoid':
        depth = qat_annotate_layer(Activation('sigmoid', name=f'{name}_sigmoid'))(depth)
    return depth


def _fastdepth(features, input_tensor, name, **options):
    return fastdepth_decoder(features, input_tensor, name=name, **options)


def _classifier(features, input_tensor, name, n_classes, logits=False, feature_level=None, **options):
    probabilities, raw = classification_head(
        _feature(features, feature_level), n_classes, return_logits=True, name=name, named_layers=True, **options)
    return raw if logits else probabilities


def _iqa(features, input_tensor, name, feature_level=None, **options):
    return iqa_regression_head(_feature(features, feature_level), name=name, **options)


HEAD_TYPES = {
    'lr_aspp': _lr_aspp_segmentation,
    'lr_aspp_depth': _lr_aspp_depth,
    'fastdepth': _fastdepth,
    'classifier': _classifier,
    'iqa': _iqa,
}


def multi_task(input_tensor, heads, backbone='MobileNetV4-Conv-S', weights='imagenet',
               freeze_backbone=True, freeze_batch_norm=True, name='multi_task'):
    '''Build one backbone with several task heads.

    Example::

        model = multi_task(
            tf.keras.Input((384, 384, 3)),
            heads={
                'seg':   dict(type='lr_aspp', n_labels=4, gate=False),
                'depth': dict(type='fastdepth', output_mode='sigmoid'),
                'quality': dict(type='iqa'),
            },
            backbone='MobileNetV4-Conv-S',
        )
        model.compile(
            optimizer='adam',
            loss={'seg': segmentation_loss, 'depth': depth_loss, 'quality': 'mse'},
            loss_weights={'seg': 1.0, 'depth': 0.5, 'quality': 0.1},
        )

    Each output is named after its head, so losses, loss weights, metrics
    and the targets of a `tf.data` pipeline can be given as dictionaries
    keyed by head name. The data loaders of this library yield one target
    per image; a dataset with one target per head has to be assembled by
    the caller.

    Input
    ----------
        input_tensor: the input tensor, with a static spatial size, e.g.
                      `keras.layers.Input((384, 384, 3))`.
        heads: a mapping from head name to its specification, a dict with a
               `type` entry (see the module docstring for the table of
               types) and the options of that head. The head name becomes
               the name of the model output and the prefix of the head's
               layers, so it has to be a valid, unique layer name.
        backbone: name of a backbone registered in
                  `swiss_army_keras._backbone_zoo.BACKBONE_REGISTRY`. All
                  its feature levels are made available to the heads.
        weights: one of `None`, `'imagenet'`, or a path to a weights file.
        freeze_backbone: `True` to freeze the backbone's weights.
        freeze_batch_norm: `True` to also freeze the backbone's BN layers.
        name: name of the model.

    Output
    ----------
        A `tf.keras.Model` with one output per head, in the order of
        `heads`. It carries `.preprocessing` (the backbone's input
        preprocessing) and `.head_names` (the head names in output order).
        The heads are annotated for quantization-aware training like every
        other builder, and the model exports through `Quantizer` and
        `deploy_utils` unchanged.
    '''
    if not heads:
        raise ValueError('multi_task needs at least one head.')

    backbone_ = backbone_zoo(
        backbone, weights, input_tensor, backbone_num_levels(backbone), freeze_backbone, freeze_batch_norm)
    features = backbone_([input_tensor, ])
    if not isinstance(features, (list, tuple)):
        features = [features]

    outputs = []
    for head_name, spec in heads.items():
        options = dict(spec)
        if 'type' not in options:
            raise ValueError(f"Head '{head_name}' has no 'type'. Available types: {sorted(HEAD_TYPES)}, or a callable.")
        kind = options.pop('type')
        if callable(kind):
            builder = kind
        elif kind in HEAD_TYPES:
            builder = HEAD_TYPES[kind]
        else:
            raise ValueError(
                f"Unknown head type '{kind}' for head '{head_name}'. Available types: {sorted(HEAD_TYPES)}, or a callable.")

        output = builder(features, input_tensor, name=head_name, **options)
        # An identity layer named after the head, so that the model output
        # carries the head's name. It disappears at TFLite conversion.
        outputs.append(Activation('linear', name=head_name)(output))

    m = Model([input_tensor, ], outputs, name=name)

    with tfmot.quantization.keras.quantize_scope():
        m = tfmot.quantization.keras.quantize_apply(m)

    m.preprocessing = backbone_.preprocessing
    m.head_names = list(heads)

    return m
