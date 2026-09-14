from __future__ import absolute_import
import math
import tensorflow as tf

from tensorflow.keras.regularizers import l2

from swiss_army_keras._backbone_zoo import backbone_zoo, bach_norm_checker
from swiss_army_keras.layer_utils import GeMPooling2D, apply_pre_pool_attention, qat_annotate_layer

import tensorflow_model_optimization as tfmot


def classifier(input_tensor, n_classes, backbone='MobileNetV3Large', weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='classifier', deep_layer=5, pooling='avg', pre_pool_attention=None, size=1024, activation="swish", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), dropout=0.3):
    '''
    ---------- (`pre_pool_attention`, plan.md §9.1) ----------
    pre_pool_attention: `None` (default, existing behavior unchanged),
        `'eca'`/`'coordatt'` (a channel/coordinate attention gate applied to
        the backbone's deepest feature map *before* the `pooling='avg'|'max'`
        reduction below), or `'gem'` (replaces the `pooling` reduction
        itself with learnable Generalized-Mean pooling, ignoring
        `pooling`) — see `swiss_army_keras.layer_utils.GeMPooling2D`/
        `ECABlock`/`CoordAttLite` for what each one actually does.
    '''

    backbone_ = backbone_zoo(
        backbone, weights, input_tensor, deep_layer, freeze_backbone, freeze_batch_norm)

    base_model = backbone_([input_tensor, ])[deep_layer-1]
    base_model = apply_pre_pool_attention(base_model, pre_pool_attention, name='{}_attn'.format(name))

    if pre_pool_attention == 'gem':
        pool = GeMPooling2D(name='{}_gem'.format(name))(base_model)
    else:
        pool = qat_annotate_layer(tf.keras.layers.GlobalAveragePooling2D())(
            base_model) if pooling == 'avg' else qat_annotate_layer(tf.keras.layers.GlobalMaxPool2D())(base_model)

    pre_classifier = qat_annotate_layer(tf.keras.layers.Dense(
        size,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
    ))(
        pool
    )  # was 128

    drop_out_class = qat_annotate_layer(
        tf.keras.layers.Dropout(dropout))(pre_classifier)
    classifier = qat_annotate_layer(tf.keras.layers.Dense(
        n_classes, activation='softmax', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))(drop_out_class)

    res = tf.keras.models.Model(inputs=input_tensor, outputs=classifier)

    with tfmot.quantization.keras.quantize_scope():
        res = tfmot.quantization.keras.quantize_apply(res)

    res.preprocessing = backbone_.preprocessing

    return res


def conv_block(x, filters, kernel_size, strides, activation=tf.keras.layers.LeakyReLU(0.2)):
    x = qat_annotate_layer(tf.keras.layers.SeparableConv2D(filters, kernel_size, strides,
                                                                                         padding="same", use_bias=False))(x)
    x = qat_annotate_layer(
        tf.keras.layers.BatchNormalization())(x)
    if activation:
        x = qat_annotate_layer(activation)(x)
    return x


def res_block(x, resizer_filters=16):
    inputs = x
    x = conv_block(x, resizer_filters, 3, 1)
    x = conv_block(x, resizer_filters, 3, 1, activation=None)
    return qat_annotate_layer(tf.keras.layers.Add())([inputs, x])


def learnable_resizer_classifier(input_tensor, n_classes, target_size=(480, 480), resizer_filters=16, resizer_kernel_size=5, num_res_blocks=1,
                                 interpolation='bilinear', backbone='EfficientNetLiteB0', weights='imagenet', freeze_backbone=True,
                                 freeze_batch_norm=True, name='classifier', deep_layer=5,
                                 pooling='avg', size=1024, activation="swish", kernel_regularizer=l2(0.001),
                                 bias_regularizer=l2(0.001), dropout=0.3):

    inputs = tf.keras.layers.Input(shape=[target_size[0], target_size[1], 3])

    # First, perform naive resizing.
    naive_resize = qat_annotate_layer(tf.keras.layers.Resizing(
        *target_size, interpolation=interpolation
    ))(input_tensor)

    # First convolution block without batch normalization.
    x = qat_annotate_layer(tf.keras.layers.Conv2D(filters=resizer_filters, kernel_size=resizer_kernel_size,
                                                                                strides=1, padding="same"))(input_tensor)
    x = qat_annotate_layer(
        tf.keras.layers.LeakyReLU(0.2))(x)

    # Second convolution block with batch normalization.
    x = qat_annotate_layer(tf.keras.layers.Conv2D(filters=resizer_filters, kernel_size=1,
                                                                                strides=1, padding="same"))(x)
    x = qat_annotate_layer(
        tf.keras.layers.LeakyReLU(0.2))(x)
    x = qat_annotate_layer(
        tf.keras.layers.BatchNormalization())(x)

    # Intermediate resizing as a bottleneck.
    bottleneck = qat_annotate_layer(tf.keras.layers.Resizing(
        *target_size, interpolation=interpolation
    ))(x)

    # Residual passes.
    for _ in range(num_res_blocks):
        x = res_block(bottleneck, resizer_filters)

    # Projection.
    x = qat_annotate_layer(tf.keras.layers.SeparableConv2D(
        filters=resizer_filters, kernel_size=3, strides=1, padding="same", use_bias=False
    ))(x)
    x = qat_annotate_layer(
        tf.keras.layers.BatchNormalization())(x)

    # Skip connection.
    x = qat_annotate_layer(
        tf.keras.layers.Add())([bottleneck, x])

    # Final resized image.
    x = qat_annotate_layer(tf.keras.layers.Conv2D(
        filters=3, kernel_size=resizer_kernel_size, strides=1, padding="same"))(x)
    final_resize = qat_annotate_layer(
        tf.keras.layers.Add())([naive_resize, x])

    backbone_ = backbone_zoo(
        backbone, weights, inputs, deep_layer, freeze_backbone, freeze_batch_norm)  # , return_outputs=True)

    base_model = backbone_([final_resize, ])[deep_layer-1]

    pool = qat_annotate_layer(tf.keras.layers.GlobalAveragePooling2D())(
        base_model) if pooling == 'avg' else qat_annotate_layer(tf.keras.layers.GlobalMaxPool2D())(base_model)

    pre_classifier = qat_annotate_layer(tf.keras.layers.Dense(
        size,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
    ))(
        pool
    )  # was 128

    drop_out_class = qat_annotate_layer(
        tf.keras.layers.Dropout(dropout))(pre_classifier)
    classifier = qat_annotate_layer(tf.keras.layers.Dense(
        n_classes, activation='softmax', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))(drop_out_class)

    res = tf.keras.models.Model(inputs=input_tensor, outputs=classifier)

    with tfmot.quantization.keras.quantize_scope():
        res = tfmot.quantization.keras.quantize_apply(res)

    res.preprocessing = backbone_.preprocessing
    return res


def wise_srnet_classifier(input_tensor, n_classes, backbone='MobileNetV3Large', weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='classifier', deep_layer=5, pooling='avg', pre_pool_attention=None, pool_size=3, size=512, activation="swish", pool_activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), dropout=0.3):
    '''
    ---------- (`pre_pool_attention`, plan.md §9.1) ----------
    pre_pool_attention: `None` (default, existing behavior unchanged) or
        `'eca'`/`'coordatt'` (a channel/coordinate attention gate applied to
        the backbone's deepest feature map before the local
        `AveragePooling2D`/`MaxPool2D` below) — see
        `swiss_army_keras.layer_utils.ECABlock`/`CoordAttLite`. `'gem'` is
        **not** supported here (raises `ValueError`): GeM pooling replaces a
        *global* pooling reduction, but this head's pooling is a local
        `pool_size`-window reduction feeding a `DepthwiseConv2D` trunk, not
        a global one — there's no matching slot for it in this
        architecture, unlike `classifier()` above.
    '''
    if pre_pool_attention == 'gem':
        raise ValueError(
            "pre_pool_attention='gem' is not supported by wise_srnet_classifier - "
            "its pooling is a local pool_size-window reduction, not the global "
            "pooling GeM replaces. Use 'eca'/'coordatt', or use classifier() "
            "instead if you specifically want GeM pooling."
        )

    backbone_ = backbone_zoo(
        backbone, weights, input_tensor, deep_layer, freeze_backbone, freeze_batch_norm)

    base_model = backbone_([input_tensor, ])[deep_layer-1]
    base_model = apply_pre_pool_attention(base_model, pre_pool_attention, name='{}_attn'.format(name))

    avg = qat_annotate_layer(tf.keras.layers.AveragePooling2D(
        pool_size, padding='valid'))(base_model) if pooling == 'avg' else qat_annotate_layer(tf.keras.layers.MaxPool2D(
            pool_size, padding='valid'))(base_model)

    # Read the actual built feature map's spatial size directly, rather
    # than assuming `stride = 2**deep_layer` (true for `tf.keras.applications`
    # backbones and MobileOne/MobileNetV4's single-stride-2-conv stems, but
    # FALSE for RepViT/FastViT's patchify stems - their first tap already
    # jumps 4x, not 2x, so a 4-tap `deep_layer=4` backbone from that family
    # sits at actual stride 32, not the assumed 2**4=16). The old formula
    # silently computed a `pool_out_size` larger than the real pooled feature
    # map, crashing the DepthwiseConv2D below with a negative-dimension error
    # - confirmed by actually building this against RepViT, not guessed
    # (plan.md §10/§13 Phase 8).
    out_size = int(base_model.shape[1])
    pool_out_size = math.floor((out_size - pool_size)/pool_size + 1)

    depthw = qat_annotate_layer(tf.keras.layers.DepthwiseConv2D(pool_out_size,
                                             activation=pool_activation,
                                             depthwise_initializer=tf.keras.initializers.RandomNormal(
                                                 mean=0.0, stddev=0.01),
                                             bias_initializer=tf.keras.initializers.Zeros(), depthwise_constraint=tf.keras.constraints.NonNeg()))(avg)
    flat = tf.keras.layers.Flatten()(depthw)

    pre_classifier = qat_annotate_layer(tf.keras.layers.Dense(
        size,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
    ))(
        flat
    )  #

    drop_out_class = qat_annotate_layer(tf.keras.layers.Dropout(dropout))(pre_classifier)
    classifier = qat_annotate_layer(tf.keras.layers.Dense(
        n_classes, activation='softmax', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))(drop_out_class)

    res = tf.keras.models.Model(inputs=input_tensor, outputs=classifier)

    with tfmot.quantization.keras.quantize_scope():
        res = tfmot.quantization.keras.quantize_apply(res)

    res.preprocessing = backbone_.preprocessing
    return res


def learnable_resizer_distiller_classifier(input_tensor, n_classes, target_size=(480, 480), resizer_filters=16, resizer_kernel_size=5, num_res_blocks=1,
                                           interpolation='bilinear',  backbone='MobileNetV3Large', weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='classifier', deep_layer=5, pooling='avg', pool_size=3, macrofeatures_number=8, size=64, activation="swish", pool_activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), dropout=0.3):

    inputs = tf.keras.layers.Input(shape=[target_size[0], target_size[1], 3])

    # First, perform naive resizing.
    naive_resize = qat_annotate_layer(tf.keras.layers.Resizing(
        *target_size, interpolation=interpolation
    ))(input_tensor)

    # First convolution block without batch normalization.
    x = qat_annotate_layer(tf.keras.layers.Conv2D(filters=resizer_filters, kernel_size=resizer_kernel_size,
                               strides=1, padding="same"))(input_tensor)
    x = qat_annotate_layer(tf.keras.layers.LeakyReLU(0.2))(x)

    # Second convolution block with batch normalization.
    x = qat_annotate_layer(tf.keras.layers.Conv2D(filters=resizer_filters, kernel_size=1,
                               strides=1, padding="same"))(x)
    x = qat_annotate_layer(tf.keras.layers.LeakyReLU(0.2))(x)
    x = qat_annotate_layer(tf.keras.layers.BatchNormalization())(x)

    # Intermediate resizing as a bottleneck.
    bottleneck = qat_annotate_layer(tf.keras.layers.Resizing(
        *target_size, interpolation=interpolation
    ))(x)

    # Residual passes.
    for _ in range(num_res_blocks):
        x = res_block(bottleneck, resizer_filters)

    # Projection.
    x = qat_annotate_layer(tf.keras.layers.SeparableConv2D(
        filters=resizer_filters, kernel_size=3, strides=1, padding="same", use_bias=False
    ))(x)
    x = qat_annotate_layer(tf.keras.layers.BatchNormalization())(x)

    # Skip connection.
    x = qat_annotate_layer(tf.keras.layers.Add())([bottleneck, x])

    # Final resized image.
    x = qat_annotate_layer(tf.keras.layers.Conv2D(
        filters=3, kernel_size=resizer_kernel_size, strides=1, padding="same"))(x)
    final_resize = qat_annotate_layer(tf.keras.layers.Add())([naive_resize, x])

    backbone_ = backbone_zoo(
        backbone, weights, inputs, deep_layer, freeze_backbone, freeze_batch_norm)

    base_model = backbone_([final_resize, ])[deep_layer-1]

    avg = qat_annotate_layer(tf.keras.layers.AveragePooling2D(
        pool_size, padding='valid'))(base_model) if pooling == 'avg' else qat_annotate_layer(tf.keras.layers.MaxPool2D(
            pool_size, padding='valid'))(base_model)

    # Read the actual built feature map's spatial size directly, rather
    # than assuming `stride = 2**deep_layer` (true for `tf.keras.applications`
    # backbones and MobileOne/MobileNetV4's single-stride-2-conv stems, but
    # FALSE for RepViT/FastViT's patchify stems - their first tap already
    # jumps 4x, not 2x, so a 4-tap `deep_layer=4` backbone from that family
    # sits at actual stride 32, not the assumed 2**4=16). The old formula
    # silently computed a `pool_out_size` larger than the real pooled feature
    # map, crashing the DepthwiseConv2D below with a negative-dimension error
    # - confirmed by actually building this against RepViT, not guessed
    # (plan.md §10/§13 Phase 8).
    out_size = int(base_model.shape[1])
    pool_out_size = math.floor((out_size - pool_size)/pool_size + 1)

    depthw = []

    for i in range(macrofeatures_number):
        d = qat_annotate_layer(tf.keras.layers.DepthwiseConv2D(pool_out_size,
                                            activation=pool_activation,
                                            depthwise_initializer=tf.keras.initializers.RandomNormal(
                                                mean=0.0, stddev=0.01),
                                            bias_initializer=tf.keras.initializers.Zeros(), depthwise_constraint=tf.keras.constraints.NonNeg()))(avg)
        flat = tf.keras.layers.Flatten()(d)
        drop_out = qat_annotate_layer(tf.keras.layers.Dropout(dropout))(flat)
        depthw.append(drop_out)

    concatenate = tf.keras.layers.Concatenate()(depthw)

    pre_classifier = qat_annotate_layer(tf.keras.layers.Dense(
        size,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
    ))(
        concatenate
    )  #

    drop_out_class = qat_annotate_layer(tf.keras.layers.Dropout(dropout))(pre_classifier)
    classifier = qat_annotate_layer(tf.keras.layers.Dense(
        n_classes, activation='softmax', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))(drop_out_class)

    res = tf.keras.models.Model(inputs=input_tensor, outputs=classifier)

    with tfmot.quantization.keras.quantize_scope():
        res = tfmot.quantization.keras.quantize_apply(res)

    res.preprocessing = backbone_.preprocessing
    return res


def distiller_classifier(input_tensor, n_classes, backbone='MobileNetV3Large', weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='classifier', deep_layer=5, pooling='avg', pool_size=3, macrofeatures_number=8, size=64, activation="swish", pool_activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), dropout=0.3):

    backbone_ = backbone_zoo(
        backbone, weights, input_tensor, deep_layer, freeze_backbone, freeze_batch_norm)

    base_model = backbone_([input_tensor, ])[deep_layer-1]

    avg = qat_annotate_layer(tf.keras.layers.AveragePooling2D(
        pool_size, padding='valid'))(base_model) if pooling == 'avg' else qat_annotate_layer(tf.keras.layers.MaxPool2D(
        pool_size, padding='valid'))(base_model)

    # Read the actual built feature map's spatial size directly, rather
    # than assuming `stride = 2**deep_layer` (true for `tf.keras.applications`
    # backbones and MobileOne/MobileNetV4's single-stride-2-conv stems, but
    # FALSE for RepViT/FastViT's patchify stems - their first tap already
    # jumps 4x, not 2x, so a 4-tap `deep_layer=4` backbone from that family
    # sits at actual stride 32, not the assumed 2**4=16). The old formula
    # silently computed a `pool_out_size` larger than the real pooled feature
    # map, crashing the DepthwiseConv2D below with a negative-dimension error
    # - confirmed by actually building this against RepViT, not guessed
    # (plan.md §10/§13 Phase 8).
    out_size = int(base_model.shape[1])
    pool_out_size = math.floor((out_size - pool_size)/pool_size + 1)

    depthw = []

    for i in range(macrofeatures_number):
        d = qat_annotate_layer(tf.keras.layers.DepthwiseConv2D(pool_out_size,
                                            activation=pool_activation,
                                            depthwise_initializer=tf.keras.initializers.RandomNormal(
                                                mean=0.0, stddev=0.01),
                                            bias_initializer=tf.keras.initializers.Zeros(), depthwise_constraint=tf.keras.constraints.NonNeg()))(avg)
        flat = tf.keras.layers.Flatten()(d)
        drop_out = qat_annotate_layer(tf.keras.layers.Dropout(dropout))(flat)
        depthw.append(drop_out)

    concatenate = tf.keras.layers.Concatenate()(depthw)

    pre_classifier = qat_annotate_layer(tf.keras.layers.Dense(
        size,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
    ))(
        concatenate
    )  #

    drop_out_class = qat_annotate_layer(tf.keras.layers.Dropout(dropout))(pre_classifier)
    classifier = qat_annotate_layer(tf.keras.layers.Dense(
        n_classes, activation='softmax', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))(drop_out_class)

    res = tf.keras.models.Model(inputs=input_tensor, outputs=classifier)

    with tfmot.quantization.keras.quantize_scope():
        res = tfmot.quantization.keras.quantize_apply(res)

    res.preprocessing = backbone_.preprocessing
    return res


def multilabel_classifier(input_tensor, n_classes, backbone='MobileNetV3Large', weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='classifier', deep_layer=5, pooling='avg', pre_pool_attention=None, size=1024, activation="swish", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), dropout=0.3):
    '''Multi-label classification head (plan.md §9.2) — identical trunk to
    `classifier()` above, but a `sigmoid`-activated, independent per-class
    output instead of a single `softmax` — for tasks where multiple
    attributes/labels can be simultaneously true for one input (framing
    that as single-label multi-class softmax is a structural mismatch, not
    a tuning problem: softmax forces every class's probability to compete
    against every other's, which is wrong whenever more than one can
    legitimately be true at once).

    Same arguments as `classifier()`, including `pre_pool_attention`
    (`swiss_army_keras.layer_utils.apply_pre_pool_attention`).
    '''

    backbone_ = backbone_zoo(
        backbone, weights, input_tensor, deep_layer, freeze_backbone, freeze_batch_norm)

    base_model = backbone_([input_tensor, ])[deep_layer-1]
    base_model = apply_pre_pool_attention(base_model, pre_pool_attention, name='{}_attn'.format(name))

    if pre_pool_attention == 'gem':
        pool = GeMPooling2D(name='{}_gem'.format(name))(base_model)
    else:
        pool = qat_annotate_layer(tf.keras.layers.GlobalAveragePooling2D())(
            base_model) if pooling == 'avg' else qat_annotate_layer(tf.keras.layers.GlobalMaxPool2D())(base_model)

    pre_classifier = qat_annotate_layer(tf.keras.layers.Dense(
        size,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
    ))(pool)

    drop_out_class = qat_annotate_layer(
        tf.keras.layers.Dropout(dropout))(pre_classifier)
    classifier_out = qat_annotate_layer(tf.keras.layers.Dense(
        n_classes, activation='sigmoid', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))(drop_out_class)

    res = tf.keras.models.Model(inputs=input_tensor, outputs=classifier_out)

    with tfmot.quantization.keras.quantize_scope():
        res = tfmot.quantization.keras.quantize_apply(res)

    res.preprocessing = backbone_.preprocessing

    return res


@tf.keras.utils.register_keras_serializable(package='swiss_army_keras')
class OrdinalThresholds(tf.keras.layers.Layer):
    '''Adds a learnable, independent bias per ordinal threshold to a single
    shared logit — the CORAL trick (Cao et al. 2020) that gives
    rank-consistent ordinal regression "for free": every threshold's logit
    is the *same* shared linear projection of the trunk features, only
    shifted by its own bias, so the thresholds' relative order is fixed by
    construction (whichever way the biases are ordered), not merely
    encouraged by a loss term the way an unconstrained `Dense(K-1)` head
    would be — that could, and in practice sometimes does, produce
    non-monotonic ("rank-inconsistent") threshold predictions for a given
    input.

    Input
    ----------
        n_thresholds: number of ordinal thresholds (`n_classes - 1` for an
                      `n_classes`-level ordinal target).

    Input/Output shapes
    ----------
        Input: `(batch, 1)` — a single shared logit per example.
        Output: `(batch, n_thresholds)` — that same logit, once per
                threshold, each shifted by its own learned bias.
    '''

    def __init__(self, n_thresholds, **kwargs):
        super().__init__(**kwargs)
        self.n_thresholds = n_thresholds

    def build(self, input_shape):
        self.bias = self.add_weight(
            name='thresholds_bias', shape=(self.n_thresholds,),
            initializer='zeros', trainable=True,
        )
        super().build(input_shape)

    def call(self, shared_logit):
        return shared_logit + self.bias[tf.newaxis, :]

    def get_config(self):
        config = super().get_config()
        config.update({'n_thresholds': self.n_thresholds})
        return config


def ordinal_classifier(input_tensor, n_classes, backbone='MobileNetV3Large', weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='classifier', deep_layer=5, pooling='avg', pre_pool_attention=None, size=1024, activation="swish", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), dropout=0.3):
    '''CORAL-style ordinal classification head (plan.md §9.2) — predicts
    `n_classes - 1` independent "is severity > threshold k" sigmoids
    sharing one trunk *and* one shared weight vector (`OrdinalThresholds`
    above), rather than `n_classes` independent softmax classes. Matters
    for any ordinally-ranked label set: plain categorical cross-entropy
    treats every pair of wrong classes as equally wrong, discarding the
    ordering information the labels actually carry (predicting "severe"
    for a "mild" case is scored no worse than predicting "moderate" for
    it, even though one error is clearly bigger than the other).

    ----------
    Cao, W., Mirjalili, V. and Raschka, S., 2020. Rank consistent ordinal
    regression for neural networks with application to age estimation.
    Pattern Recognition Letters.

    Output
    ----------
        A `tf.keras.Model` producing `(batch, n_classes - 1)` sigmoid
        probabilities — `P(label > k)` for each threshold `k`. Convert to a
        predicted class via `n_predicted = 1 + sum(P(label > k) > 0.5 for
        all k)` (i.e. count how many thresholds the prediction clears), the
        standard CORAL decoding rule — not done inside this function, since
        that's an inference-time decision, not part of the model graph.
        Train with binary cross-entropy against `n_classes - 1`
        binarized-threshold targets (`target_k = 1 if true_label > k else
        0`), not categorical cross-entropy.
    '''

    backbone_ = backbone_zoo(
        backbone, weights, input_tensor, deep_layer, freeze_backbone, freeze_batch_norm)

    base_model = backbone_([input_tensor, ])[deep_layer-1]
    base_model = apply_pre_pool_attention(base_model, pre_pool_attention, name='{}_attn'.format(name))

    if pre_pool_attention == 'gem':
        pool = GeMPooling2D(name='{}_gem'.format(name))(base_model)
    else:
        pool = qat_annotate_layer(tf.keras.layers.GlobalAveragePooling2D())(
            base_model) if pooling == 'avg' else qat_annotate_layer(tf.keras.layers.GlobalMaxPool2D())(base_model)

    pre_classifier = qat_annotate_layer(tf.keras.layers.Dense(
        size,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
    ))(pool)

    drop_out_class = qat_annotate_layer(
        tf.keras.layers.Dropout(dropout))(pre_classifier)

    shared_logit = qat_annotate_layer(tf.keras.layers.Dense(
        1, use_bias=False, kernel_regularizer=kernel_regularizer, name='{}_shared_logit'.format(name)))(drop_out_class)
    ordinal_logits = OrdinalThresholds(n_classes - 1, name='{}_thresholds'.format(name))(shared_logit)
    classifier_out = qat_annotate_layer(
        tf.keras.layers.Activation('sigmoid', name='{}_sigmoid'.format(name)))(ordinal_logits)

    res = tf.keras.models.Model(inputs=input_tensor, outputs=classifier_out)

    with tfmot.quantization.keras.quantize_scope():
        res = tfmot.quantization.keras.quantize_apply(res)

    res.preprocessing = backbone_.preprocessing

    return res
