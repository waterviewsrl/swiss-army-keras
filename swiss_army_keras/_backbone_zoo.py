'''Backbone registry (`BACKBONE_REGISTRY`) and the `backbone_zoo` factory that
turns a registered backbone into a multi-output feature-pyramid model.
'''
from __future__ import absolute_import

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple, Union

import tensorflow as tf

from tensorflow.keras.applications import *
from tensorflow.keras.models import Model

from swiss_army_keras.utils import freeze_model
from swiss_army_keras.layer_utils import dummy_preprocessing
from swiss_army_keras._backbone_repvit import (
    REPVIT_CONFIGS,
    RepViT_M0_9,
    RepViT_M1_0,
    RepViT_M1_1,
    RepViT_M1_5,
    RepViT_M2_3,
    repvit_preprocess_input,
    repvit_stage_output_names,
)
from swiss_army_keras._backbone_fastvit import (
    FASTVIT_CONFIGS,
    FastViT_MA36,
    FastViT_S12,
    FastViT_SA12,
    FastViT_SA24,
    FastViT_SA36,
    FastViT_T8,
    FastViT_T12,
    fastvit_preprocess_input,
    fastvit_stage_output_names,
)
from swiss_army_keras._backbone_mobileone import (
    MOBILEONE_CONFIGS,
    MobileOne_S0,
    MobileOne_S1,
    MobileOne_S2,
    MobileOne_S3,
    MobileOne_S4,
    mobileone_preprocess_input,
    mobileone_stage_output_names,
)
from swiss_army_keras._backbone_mobilenetv4 import (
    MOBILENETV4_CONFIGS,
    MobileNetV4_ConvS,
    MobileNetV4_ConvM,
    MobileNetV4_ConvL,
    MobileNetV4_HybridM,
    MobileNetV4_HybridL,
    mobilenetv4_preprocess_input,
    mobilenetv4_stage_output_names,
)
from swiss_army_keras._backbone_strideformer import (
    StrideFormer_Tiny,
    StrideFormer_Base,
    strideformer_preprocess_input,
    strideformer_stage_output_names,
)

import tensorflow_hub as hub

import warnings

try:
    # `efficientnet-lite-keras` is an unmaintained fork with a hard Python
    # ceiling (setup.py caps at Python<=3.11) that has nothing to do with
    # this package's own compatibility — degrade gracefully instead of
    # making the whole backbone zoo (and therefore every backbone/decoder)
    # unimportable just because EfficientNet-Lite specifically is unusable
    # in the current environment.
    from efficientnet_lite import *
    _HAS_EFFICIENTNET_LITE = True
except ImportError:
    warnings.warn(
        "\n----------------------------------------\n"
        "Could not import `efficientnet_lite` (from the `efficientnet-lite-keras` "
        "package) - EfficientNetLiteB0-B4 backbones will not be available. "
        "Every other backbone is unaffected.\n"
        "----------------------------------------"
    )
    _HAS_EFFICIENTNET_LITE = False

@dataclass(frozen=True)
class BackboneSpec:
    '''A single backbone's registration in `BACKBONE_REGISTRY`.

    Replaces the earlier three parallel dicts (`keras_layer_cadidates` /
    `keras_normalization_layers` / `keras_kwargs`), which had to be kept in
    sync by hand across every backbone and were about to gain a 4th
    (`custom_backbone_constructors`, for the non-`tf.keras.applications`
    backbones planned in plan.md §2) — one dataclass per backbone removes
    that drift risk and doubles as the extension point those custom
    backbones will register through.

    Input
    ----------
        constructor: the backbone-building callable, invoked as
            `constructor(include_top=False, weights=weights, input_tensor=input_tensor, pooling=None, **kwargs)`.
            Stored as an actual reference (e.g. `tf.keras.applications.MobileNetV2`)
            rather than a name to `eval()`.
        layers: 5 (or fewer) entries identifying the feature-pyramid taps, in
            increasing depth order. Each entry is either a layer **name**
            (preferred — used for every backbone whose layers have stable,
            semantic names) or a layer **index** (only where names are
            Keras's own auto-generated, globally-auto-incremented
            placeholders and therefore unstable across build order within a
            process — see `MobileNetV3Large` below and plan.md §10 for why).
        preprocessing: the matching `preprocess_input`-style callable.
        kwargs: extra keyword arguments passed to `constructor`.
    '''
    constructor: Callable
    layers: Tuple[Union[str, int], ...]
    preprocessing: Callable
    kwargs: dict = field(default_factory=dict)


BACKBONE_REGISTRY: Dict[str, BackboneSpec] = {
    'MobileNetV2': BackboneSpec(
        constructor=tf.keras.applications.MobileNetV2,
        layers=('block_1_expand_relu', 'block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu', 'out_relu'),
        preprocessing=tf.keras.applications.mobilenet_v2.preprocess_input,
    ),
    'MobileNetV3Large': BackboneSpec(
        constructor=tf.keras.applications.MobileNetV3Large,
        # Indices, not names: this model's own `tf.keras.applications` source
        # builds these 5 feature-pyramid-tap layers as plain `ReLU()`, with no
        # explicit `name=`, so Keras auto-generates one from a *global*,
        # process-wide counter. That makes any hardcoded name (as this entry
        # used to be — twice, see git history) correct only for whichever
        # build-order happens to align, and wrong otherwise. A freshly-built
        # MobileNetV3Large always has the same total layer count and internal
        # topology, so indexing by position is stable regardless of what else
        # was built earlier in the process; verified empirically (same layer
        # type/shape at these indices whether built first or 3rd in-process).
        layers=(13, 31, 58, 102, 138),
        preprocessing=tf.keras.applications.mobilenet_v3.preprocess_input,
        kwargs={'minimalistic': True},
    ),
    'VGG16': BackboneSpec(
        constructor=tf.keras.applications.VGG16,
        layers=('block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3'),
        preprocessing=tf.keras.applications.vgg16.preprocess_input,
    ),
    'VGG19': BackboneSpec(
        constructor=tf.keras.applications.VGG19,
        layers=('block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4'),
        preprocessing=tf.keras.applications.vgg19.preprocess_input,
    ),
    'ResNet50': BackboneSpec(
        constructor=tf.keras.applications.ResNet50,
        layers=('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'),
        preprocessing=tf.keras.applications.resnet.preprocess_input,
    ),
    'ResNet101': BackboneSpec(
        constructor=tf.keras.applications.ResNet101,
        layers=('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block23_out', 'conv5_block3_out'),
        preprocessing=tf.keras.applications.resnet.preprocess_input,
    ),
    'ResNet152': BackboneSpec(
        constructor=tf.keras.applications.ResNet152,
        layers=('conv1_relu', 'conv2_block3_out', 'conv3_block8_out', 'conv4_block36_out', 'conv5_block3_out'),
        preprocessing=tf.keras.applications.resnet.preprocess_input,
    ),
    'ResNet50V2': BackboneSpec(
        constructor=tf.keras.applications.ResNet50V2,
        layers=('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block6_1_relu', 'post_relu'),
        preprocessing=tf.keras.applications.resnet.preprocess_input,
    ),
    'ResNet101V2': BackboneSpec(
        constructor=tf.keras.applications.ResNet101V2,
        layers=('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block23_1_relu', 'post_relu'),
        preprocessing=tf.keras.applications.resnet.preprocess_input,
    ),
    'ResNet152V2': BackboneSpec(
        constructor=tf.keras.applications.ResNet152V2,
        layers=('conv1_conv', 'conv2_block3_1_relu', 'conv3_block8_1_relu', 'conv4_block36_1_relu', 'post_relu'),
        preprocessing=tf.keras.applications.resnet.preprocess_input,
    ),
    'DenseNet121': BackboneSpec(
        constructor=tf.keras.applications.DenseNet121,
        layers=('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
        preprocessing=tf.keras.applications.densenet.preprocess_input,
    ),
    'DenseNet169': BackboneSpec(
        constructor=tf.keras.applications.DenseNet169,
        layers=('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
        preprocessing=tf.keras.applications.densenet.preprocess_input,
    ),
    'DenseNet201': BackboneSpec(
        constructor=tf.keras.applications.DenseNet201,
        layers=('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
        preprocessing=tf.keras.applications.densenet.preprocess_input,
    ),
}

for _b0_7_name, _b0_7_ctor in [
    ('EfficientNetB0', tf.keras.applications.EfficientNetB0),
    ('EfficientNetB1', tf.keras.applications.EfficientNetB1),
    ('EfficientNetB2', tf.keras.applications.EfficientNetB2),
    ('EfficientNetB3', tf.keras.applications.EfficientNetB3),
    ('EfficientNetB4', tf.keras.applications.EfficientNetB4),
    ('EfficientNetB5', tf.keras.applications.EfficientNetB5),
    ('EfficientNetB6', tf.keras.applications.EfficientNetB6),
    ('EfficientNetB7', tf.keras.applications.EfficientNetB7),
]:
    BACKBONE_REGISTRY[_b0_7_name] = BackboneSpec(
        constructor=_b0_7_ctor,
        layers=('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        preprocessing=tf.keras.applications.efficientnet.preprocess_input,
    )
del _b0_7_name, _b0_7_ctor

if _HAS_EFFICIENTNET_LITE:
    for _lite_name, _lite_ctor in [
        ('EfficientNetLiteB0', EfficientNetLiteB0),
        ('EfficientNetLiteB1', EfficientNetLiteB1),
        ('EfficientNetLiteB2', EfficientNetLiteB2),
        ('EfficientNetLiteB3', EfficientNetLiteB3),
        ('EfficientNetLiteB4', EfficientNetLiteB4),
    ]:
        BACKBONE_REGISTRY[_lite_name] = BackboneSpec(
            constructor=_lite_ctor,
            layers=('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'block7a_activation'),
            preprocessing=get_preprocessing_layer(),
        )
    del _lite_name, _lite_ctor

# RepViT (Meituan/THU-MIG, 2023) - custom, non-`tf.keras.applications`
# backbone (plan.md §2.4); the first entrant in the "custom_backbone_constructors"
# extension point this registry was designed for (plan.md §2.5). Only 4
# pyramid taps, not 5 - see `_backbone_repvit.py`'s module docstring for why;
# `backbone_zoo()`'s `depth_max = len(cadidate)` clamp already handles that.
for _repvit_key, _repvit_ctor in [
    ('RepViT-M0.9', RepViT_M0_9),
    ('RepViT-M1.0', RepViT_M1_0),
    ('RepViT-M1.1', RepViT_M1_1),
    ('RepViT-M1.5', RepViT_M1_5),
    ('RepViT-M2.3', RepViT_M2_3),
]:
    _repvit_config = REPVIT_CONFIGS[_repvit_key]
    BACKBONE_REGISTRY[_repvit_key] = BackboneSpec(
        constructor=_repvit_ctor,
        layers=repvit_stage_output_names(_repvit_config['model_name'], _repvit_config['stage_blocks']),
        preprocessing=repvit_preprocess_input,
    )
del _repvit_key, _repvit_ctor, _repvit_config

# FastViT (Apple, 2023) - also a custom backbone (plan.md §2.3). Same 4-tap
# (not 5) native pyramid as RepViT, for the same reason (aggressive patchify
# stem) - see `_backbone_fastvit.py`'s module docstring.
for _fastvit_key, _fastvit_ctor in [
    ('FastViT-T8', FastViT_T8),
    ('FastViT-T12', FastViT_T12),
    ('FastViT-S12', FastViT_S12),
    ('FastViT-SA12', FastViT_SA12),
    ('FastViT-SA24', FastViT_SA24),
    ('FastViT-SA36', FastViT_SA36),
    ('FastViT-MA36', FastViT_MA36),
]:
    _fastvit_config = FASTVIT_CONFIGS[_fastvit_key]
    BACKBONE_REGISTRY[_fastvit_key] = BackboneSpec(
        constructor=_fastvit_ctor,
        layers=fastvit_stage_output_names(_fastvit_config['model_name']),
        preprocessing=fastvit_preprocess_input,
    )
del _fastvit_key, _fastvit_ctor, _fastvit_config

# MobileOne (Apple, 2022) - the third custom backbone. Unlike RepViT/FastViT,
# a genuine 5-tap 2x/4x/8x/16x/32x pyramid exists here (single-stride-2-conv
# stem, not a 4x-jump patchify stem) - see `_backbone_mobileone.py`'s module
# docstring.
for _mobileone_key, _mobileone_ctor in [
    ('MobileOne-S0', MobileOne_S0),
    ('MobileOne-S1', MobileOne_S1),
    ('MobileOne-S2', MobileOne_S2),
    ('MobileOne-S3', MobileOne_S3),
    ('MobileOne-S4', MobileOne_S4),
]:
    BACKBONE_REGISTRY[_mobileone_key] = BackboneSpec(
        constructor=_mobileone_ctor,
        layers=mobileone_stage_output_names(MOBILEONE_CONFIGS[_mobileone_key]['model_name']),
        preprocessing=mobileone_preprocess_input,
    )
del _mobileone_key, _mobileone_ctor

# MobileNetV4 (Google, 2024) - the fourth and final custom backbone
# (plan.md §2.2). Same genuine 5-tap 2x/4x/8x/16x/32x pyramid as MobileOne
# (single-stride-2-conv stem, not a patchify stem) - see
# `_backbone_mobilenetv4.py`'s module docstring. Unlike the other three
# custom backbones, this one has no structural reparameterization at all,
# so there is no `reparameterize_mobilenetv4` and no fusion golden-output
# test - confirmed during architecture research (plan.md §2.2), not
# assumed. Hybrid-M/L (Mobile-MQA attention variants) are included, per
# the precedent set by including FastViT's SA/MA attention variants under
# "all sizes" (plan.md §7).
for _mobilenetv4_key, _mobilenetv4_ctor in [
    ('MobileNetV4-Conv-S', MobileNetV4_ConvS),
    ('MobileNetV4-Conv-M', MobileNetV4_ConvM),
    ('MobileNetV4-Conv-L', MobileNetV4_ConvL),
    ('MobileNetV4-Hybrid-M', MobileNetV4_HybridM),
    ('MobileNetV4-Hybrid-L', MobileNetV4_HybridL),
]:
    BACKBONE_REGISTRY[_mobilenetv4_key] = BackboneSpec(
        constructor=_mobilenetv4_ctor,
        layers=mobilenetv4_stage_output_names(MOBILENETV4_CONFIGS[_mobilenetv4_key]['model_name']),
        preprocessing=mobilenetv4_preprocess_input,
    )
del _mobilenetv4_key, _mobilenetv4_ctor

# StrideFormer (Baidu PaddleSeg, 2023) - the fifth custom backbone
# (plan.md §3.5). A genuine **3-tap** pyramid (stride 8/16/32) - fewer taps
# than any other backbone registered here, including RepViT/FastViT's
# 4-tap patchify-stem situation - since this family's first tap is already
# at stride8 (`_backbone_strideformer.py`'s own `_strideformer_network`
# starts a stage2/3/4 tap numbering at stride8, not stride2/4). Decoders
# consuming this backbone need an explicit `feature_levels=(1,2,3)`/
# `deep_layer=3` override, the same pattern RepViT/FastViT already need.
# In the reference, this backbone's own `forward()` also runs AAM (a
# 3-level fusion module) internally, producing one already-fused output -
# deliberately not reproduced here, so this backbone stays genuinely
# swappable with every decoder in this repo, not just its own paper's head
# (`pp_mobileseg`, which applies AAM itself as a decoder-level step instead
# - see `_model_pp_mobileseg.py` and `_backbone_strideformer.py`'s own
# module docstring for the full reasoning).
for _strideformer_key, _strideformer_ctor in [
    ('StrideFormer-Tiny', StrideFormer_Tiny),
    ('StrideFormer-Base', StrideFormer_Base),
]:
    BACKBONE_REGISTRY[_strideformer_key] = BackboneSpec(
        constructor=_strideformer_ctor,
        layers=strideformer_stage_output_names(_strideformer_key.replace('-', '_')),
        preprocessing=strideformer_preprocess_input,
    )
del _strideformer_key, _strideformer_ctor


hub_layer_candidates = {
    'EfficientNet-lite0' : ('reduction_1/expansion_output', 'reduction_2/expansion_output', 'reduction_3/expansion_output', 'reduction_4/expansion_output', 'reduction_5/expansion_output'),
    'EfficientNet-lite1' : ('reduction_1/expansion_output', 'reduction_2/expansion_output', 'reduction_3/expansion_output', 'reduction_4/expansion_output', 'reduction_5/expansion_output'),
    'EfficientNet-lite2' : ('reduction_1/expansion_output', 'reduction_2/expansion_output', 'reduction_3/expansion_output', 'reduction_4/expansion_output', 'reduction_5/expansion_output'),
    'EfficientNet-lite3' : ('reduction_1/expansion_output', 'reduction_2/expansion_output', 'reduction_3/expansion_output', 'reduction_4/expansion_output', 'reduction_5/expansion_output'),
    'EfficientNet-lite4' : ('reduction_1/expansion_output', 'reduction_2/expansion_output', 'reduction_3/expansion_output', 'reduction_4/expansion_output', 'reduction_5/expansion_output'),

}

backbone_hub_urls = {
    'EfficientNet-lite0' : 'https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2',
    'EfficientNet-lite1' : 'https://tfhub.dev/tensorflow/efficientnet/lite1/feature-vector/2',
    'EfficientNet-lite2' : 'https://tfhub.dev/tensorflow/efficientnet/lite2/feature-vector/2',
    'EfficientNet-lite3' : 'https://tfhub.dev/tensorflow/efficientnet/lite3/feature-vector/2',
    'EfficientNet-lite4' : 'https://tfhub.dev/tensorflow/efficientnet/lite4/feature-vector/2',
}

backbone_hub_signatures = {
    'EfficientNet-lite0' : 'image_feature_vector',
    'EfficientNet-lite1' : 'image_feature_vector',
    'EfficientNet-lite2' : 'image_feature_vector',
    'EfficientNet-lite3' : 'image_feature_vector',
    'EfficientNet-lite4' : 'image_feature_vector',
}

def bach_norm_checker(backbone_name, batch_norm):
    '''batch norm checker'''
    if 'VGG' in backbone_name:
        batch_norm_backbone = False
    else:
        batch_norm_backbone = True
        
    if batch_norm_backbone != batch_norm:       
        if batch_norm_backbone:    
            param_mismatch = "\n\nBackbone {} uses batch norm, but other layers received batch_norm={}".format(backbone_name, batch_norm)
        else:
            param_mismatch = "\n\nBackbone {} does not use batch norm, but other layers received batch_norm={}".format(backbone_name, batch_norm)
            
        warnings.warn(param_mismatch);
        
def _get_layer(model, layer_ref):
    '''Resolve a `BackboneSpec.layers` entry to a layer, by name or index.'''
    if isinstance(layer_ref, str):
        return model.get_layer(name=layer_ref)
    return model.get_layer(index=layer_ref)


def backbone_num_levels(backbone_name):
    '''Number of feature levels (taps) that a backbone exposes.

    Most backbones expose five levels; RepViT and FastViT expose four and
    StrideFormer three. Builders use this to pick default taps that exist
    for the requested backbone.

    Input
    ----------
        backbone_name: a key of `BACKBONE_REGISTRY`, or a `'hub://<name>'`
                       backbone.

    Output
    ----------
        The number of feature levels, as an int.
    '''
    if backbone_name.startswith('hub://'):
        return len(hub_layer_candidates[backbone_name[6:]])
    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(
            "Unknown backbone '{}'. Available backbones: {}".format(
                backbone_name, sorted(BACKBONE_REGISTRY)
            )
        )
    return len(BACKBONE_REGISTRY[backbone_name].layers)


def backbone_zoo(backbone_name, weights, input_tensor, depth, freeze_backbone, freeze_batch_norm, return_outputs=False):
    '''
    Configuring a user specified encoder model based on `BACKBONE_REGISTRY`
    (built-ins from `tensorflow.keras.applications`, plus any custom
    backbones registered there — see plan.md §2.5) or, for `hub://`-prefixed
    names, `tensorflow_hub`.

    Input
    ----------
        backbone_name: the backbone model name — a key of `BACKBONE_REGISTRY`
                       (`list(BACKBONE_REGISTRY)` for the current set), or
                       `'hub://<name>'` for a `tensorflow_hub`-backed backbone
                       (a key of `hub_layer_candidates`).
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet),
                 or the path to the weights file to be loaded.
        input_tensor: the input tensor
        depth: number of encoded feature maps.
               If four dwonsampling levels are needed, then depth=4.

        freeze_backbone: True for a frozen backbone
        freeze_batch_norm: False for not freezing batch normalization layers.

    Output
    ----------
        model: a keras backbone model.

    '''

    # ----- #

    using_hub = False
    if 'hub://' in backbone_name:
        using_hub = True
        backbone_name = backbone_name[6:]
        backbone_ = hub.KerasLayer(backbone_hub_urls[backbone_name], input_shape=(513, 513, 3), signature=backbone_hub_signatures[backbone_name], signature_outputs_as_dict=True, trainable=False)
        backbone_ = backbone_(input_tensor)
        cadidate = hub_layer_candidates[backbone_name]

    else:
        if backbone_name not in BACKBONE_REGISTRY:
            raise ValueError(
                "Unknown backbone '{}'. Available backbones: {}".format(
                    backbone_name, sorted(BACKBONE_REGISTRY)
                )
            )
        spec = BACKBONE_REGISTRY[backbone_name]
        backbone_ = spec.constructor(include_top=False, weights=weights, input_tensor=input_tensor, pooling=None, **spec.kwargs)

        cadidate = spec.layers

    # ----- #
    # depth checking
    depth_max = len(cadidate)
    if depth > depth_max:
        depth = depth_max

    X_skip = []

    for i in range(depth):
        if using_hub:
            X_skip.append(backbone_[cadidate[i]])
        else:
            X_skip.append(_get_layer(backbone_, cadidate[i]).output)

    # `tensorflow_hub` feature-vector modules (the only `hub://` backbones
    # registered here) already expect [0,1]-normalized float32 input, so no
    # separate preprocessing function is needed - unlike the
    # `tf.keras.applications`/registry path, whose `preprocess_input`
    # functions vary per architecture (see `BackboneSpec.preprocessing`).
    # NOTE: the pre-existing code before this refactor always looked up
    # `keras_normalization_layers[backbone_name]` here regardless of
    # `using_hub`, which would `KeyError` for every hub backbone (none of
    # those names were ever in that dict) - fixed here rather than
    # reproduced, since it's a one-line difference either way.
    preprocessing = dummy_preprocessing if using_hub else spec.preprocessing

    if return_outputs:
        return X_skip, preprocessing

    model = Model(inputs=[input_tensor,], outputs=X_skip, name='{}_backbone'.format(backbone_name))

    if freeze_backbone:
        model = freeze_model(model, freeze_batch_norm=freeze_batch_norm)

    model.preprocessing = preprocessing
    return model
