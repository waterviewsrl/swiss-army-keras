from __future__ import absolute_import
import math
import tensorflow as tf

from tensorflow.keras.regularizers import l2

from swiss_army_keras._backbone_zoo import backbone_zoo, bach_norm_checker
from swiss_army_keras.layer_utils import GeMPooling2D, apply_pre_pool_attention, iqa_regression_head, qat_annotate_layer

import tensorflow_model_optimization as tfmot


def classifier(input_tensor, n_classes, backbone='MobileNetV3Large', weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='classifier', deep_layer=5, pooling='avg', pre_pool_attention=None, size=1024, activation="swish", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), dropout=0.3, return_logits=False, iqa_head=False):
    '''
    ---------- (`pre_pool_attention`, plan.md §9.1) ----------
    pre_pool_attention: `None` (default, existing behavior unchanged),
        `'eca'`/`'coordatt'` (a channel/coordinate attention gate applied to
        the backbone's deepest feature map *before* the `pooling='avg'|'max'`
        reduction below), or `'gem'` (replaces the `pooling` reduction
        itself with learnable Generalized-Mean pooling, ignoring
        `pooling`) — see `swiss_army_keras.layer_utils.GeMPooling2D`/
        `ECABlock`/`CoordAttLite` for what each one actually does.

    ---------- (`return_logits`, plan.md §13.2) ----------
    return_logits: `False` (default) — unchanged behavior, a single
        softmax-probabilities output, exactly as before. `True` — splits
        the final `Dense`+softmax into two separate layer calls (same
        restructuring `ordinal_classifier` already does, zero new
        trainable weights) and returns `[probabilities, logits]` instead
        of just `probabilities`, so raw logits become available for
        `swiss_army_keras.layer_utils.add_energy_score_output` (softmax
        alone can't recover them — it normalizes away the scale the Energy
        Score formula needs).

    ---------- (`iqa_head`, plan.md §13.7) ----------
    iqa_head: `False` (default) — unchanged behavior. `True` — adds a
        small `swiss_army_keras.layer_utils.iqa_regression_head` side
        branch tapped off this same backbone (`base_model`, before this
        function's own pooling/Dense tail), trained simultaneously with
        the main classification task (e.g. via
        `swiss_army_keras.iqa_utils.add_teacher_iqa_labels` supplying a
        teacher-distilled quality pseudo-label) — the exported model
        carries both heads off one shared encoder. Appends its `[0, 1]`
        score as the *last* output — `[probabilities, iqa_score]`, or
        `[probabilities, logits, iqa_score]` if `return_logits=True` is
        also set (the two flags compose).
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

    if return_logits:
        logits = qat_annotate_layer(tf.keras.layers.Dense(
            n_classes, activation=None, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))(drop_out_class)
        classifier = qat_annotate_layer(tf.keras.layers.Activation('softmax'))(logits)
        outputs = [classifier, logits]
    else:
        classifier = qat_annotate_layer(tf.keras.layers.Dense(
            n_classes, activation='softmax', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))(drop_out_class)
        outputs = classifier

    if iqa_head:
        iqa_score = iqa_regression_head(base_model, name='{}_iqa'.format(name))
        outputs = (outputs if isinstance(outputs, list) else [outputs]) + [iqa_score]

    res = tf.keras.models.Model(inputs=input_tensor, outputs=outputs)

    with tfmot.quantization.keras.quantize_scope():
        res = tfmot.quantization.keras.quantize_apply(res)

    res.preprocessing = backbone_.preprocessing

    return res


def conv_block(x, filters, kernel_size, strides, activation=tf.keras.layers.LeakyReLU(0.2)):
    '''Separable conv + BatchNorm + activation block used by the learnable resizer. Internal.
    '''
    x = qat_annotate_layer(tf.keras.layers.SeparableConv2D(filters, kernel_size, strides,
                                                                                         padding="same", use_bias=False))(x)
    x = qat_annotate_layer(
        tf.keras.layers.BatchNormalization())(x)
    if activation:
        x = qat_annotate_layer(activation)(x)
    return x


def res_block(x, resizer_filters=16):
    '''Residual block made of two `conv_block` calls, used by the learnable resizer. Internal.
    '''
    inputs = x
    x = conv_block(x, resizer_filters, 3, 1)
    x = conv_block(x, resizer_filters, 3, 1, activation=None)
    return qat_annotate_layer(tf.keras.layers.Add())([inputs, x])


def learnable_resizer_classifier(input_tensor, n_classes, target_size=(480, 480), resizer_filters=16, resizer_kernel_size=5, num_res_blocks=1,
                                 interpolation='bilinear', backbone='EfficientNetLiteB0', weights='imagenet', freeze_backbone=True,
                                 freeze_batch_norm=True, name='classifier', deep_layer=5,
                                 pooling='avg', size=1024, activation="swish", kernel_regularizer=l2(0.001),
                                 bias_regularizer=l2(0.001), dropout=0.3):

    '''`classifier` preceded by a learnable resizer (Talebi & Milanfar, 2021,
    "Learning to Resize Images for Computer Vision Tasks"): a small CNN that
    resizes the input to `target_size` jointly trained with the classifier,
    added to a plain `interpolation` resize as a residual.

    Input
    ----------
        input_tensor: the input tensor (any fixed size).
        n_classes: number of classes.
        target_size: `(height, width)` fed to the backbone.
        resizer_filters, resizer_kernel_size, num_res_blocks: size of the
            resizer network.
        interpolation: interpolation of the resize operations.
        (remaining arguments: see `classifier`)

    Output
    ----------
        a QAT-wrapped `tf.keras.Model` with a `preprocessing` attribute.
    '''
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

    '''`distiller_classifier` preceded by the learnable resizer described in
    `learnable_resizer_classifier`. Arguments are the union of the two.
    '''
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

    '''Classifier head that distils the backbone's last feature map into a few
    "macro-features" instead of globally pooling it: the feature map is
    pooled with a `pool_size` window, then `macrofeatures_number` parallel
    full-size depthwise convolutions (non-negative weights) each reduce it to
    one vector; the vectors are concatenated and fed to a small dense layer
    and a softmax classifier.

    Input
    ----------
        input_tensor: the input tensor.
        n_classes: number of classes (softmax output).
        backbone, weights, freeze_backbone, freeze_batch_norm: see `classifier`.
        deep_layer: index (1-based) of the backbone tap to use.
        pooling: 'avg' or 'max' for the initial pooling.
        pool_size: window (and stride) of the initial pooling.
        macrofeatures_number: number of parallel depthwise reductions.
        size: units of the dense layer before the classifier.
        activation: activation of that dense layer.
        pool_activation: optional activation of the depthwise reductions.
        kernel_regularizer, bias_regularizer, dropout: regularization of the
            dense layers.

    Output
    ----------
        a QAT-wrapped `tf.keras.Model` with a `preprocessing` attribute.
    '''
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


def pose_regression_head(input_tensor, n_keypoints, backbone='MobileNetV3Large', weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='pose_regression', deep_layer=5, pooling='avg', pre_pool_attention=None, size=1024, activation="swish", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), dropout=0.3):
    '''Lightweight single-person keypoint-regression head (plan.md §13.3) —
    a direct, heatmap-free alternative to MoveNet-style pose estimation,
    architecturally identical to `classifier()`'s own trunk with the final
    softmax classification swapped for a linear-then-sigmoid keypoint
    regression (BlazePose's own inference-time shape, Bazarevsky et al.
    2020, arXiv:2006.10204, once its training-only heatmap branch is
    dropped) — no upsampling, no argmax, no gather, per-tensor
    int8-quantization-friendly with no caveats.

    **Single-person only, by design** (plan.md §13.3's own explicit scoping
    decision, not an oversight) — no person-center heatmap, no
    multi-instance grouping. Intended for a top-down pipeline: crop a
    person box from this repo's own `fcos_lite_head`/`gfl_lite_head`
    detection output, resize the crop to this head's own input size, run
    this model on it. Multi-person/heatmap-based pose (MoveNet's own
    architecture) is a separately-scoped, much larger addition, not
    attempted here.

    Input
    ----------
        input_tensor, n_keypoints: the input tensor, and how many keypoints
                                   to regress per person.
        backbone, weights, freeze_backbone, freeze_batch_norm, deep_layer,
        pooling, pre_pool_attention, size, activation, kernel_regularizer,
        bias_regularizer, dropout: identical meaning to `classifier()`'s
            own arguments — same trunk (backbone tap → optional
            `pre_pool_attention` gate → pooling → `Dense(size)` →
            `Dropout`).

    Output
    ----------
        A `tf.keras.Model` with a `.preprocessing` attribute, producing
        `(batch, n_keypoints, 3)` — `(x, y, visibility)` per keypoint, all
        three **sigmoid-bounded to `[0, 1]`**: `(x, y)` are fraction-of-
        input-size coordinates, not pixel coordinates (multiply by this
        model's own input width/height to get pixels — the same
        normalized-coordinate convention BlazePose itself uses);
        `visibility` is a soft confidence/occlusion score, not a hard flag.
        Train with a regression loss (e.g. smooth-L1/MSE) against
        `(x, y, visibility)` targets in the same `[0, 1]` convention.
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

    pre_head = qat_annotate_layer(tf.keras.layers.Dense(
        size,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
    ))(pool)

    drop_out = qat_annotate_layer(
        tf.keras.layers.Dropout(dropout))(pre_head)

    raw = qat_annotate_layer(tf.keras.layers.Dense(
        n_keypoints * 3, activation='sigmoid',
        kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
        name='{}_keypoints'.format(name),
    ))(drop_out)
    keypoints = tf.keras.layers.Reshape((n_keypoints, 3), name='{}_reshape'.format(name))(raw)

    res = tf.keras.models.Model(inputs=input_tensor, outputs=keypoints)

    with tfmot.quantization.keras.quantize_scope():
        res = tfmot.quantization.keras.quantize_apply(res)

    res.preprocessing = backbone_.preprocessing

    return res


def reid_embedding_head(input_tensor, embedding_size=128, backbone='MobileNetV3Large', weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='reid_embedding', deep_layer=5, pooling='avg', pre_pool_attention='gem', size=1024, activation="swish", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), dropout=0.3):
    '''Lightweight embedding head for re-identification / metric-learning
    (plan.md §13.4) — `classifier()`'s own trunk with the final softmax
    classification removed, replaced by a linear projection + L2
    normalization, trained via a metric-learning loss (`losses.triplet_1d`,
    already implemented — no new loss needed) rather than cross-entropy.
    Architecturally close enough to `classifier()`'s own trunk that this is
    a near-trivial addition, not a new architecture.

    **`pre_pool_attention` defaults to `'gem'`, unlike every other
    classifier variant in this file (`classifier`/`wise_srnet_classifier`/
    etc. default to `None`)** — a deliberate, domain-specific choice:
    Generalized-Mean pooling (`GeMPooling2D`, plan.md §9.1) is the same
    pooling convention the well-known "strong ReID baseline" uses (Luo,
    Gu, Liao, Lai, Jiang, "Bag of Tricks and a Strong Baseline for Deep
    Person Re-identification", CVPR-W 2019, arXiv:1903.07071), not copied
    from `classifier()`'s own default verbatim.

    **The final `UnitNormalization` layer is deliberately not
    `qat_annotate_layer`-wrapped** — confirmed empirically that
    `tfmot.quantization.keras.quantize_apply` raises `RuntimeError:
    ...UnitNormalization is not supported` when annotated, the same
    "unregistered layer type" rule already established in this repo for
    `Add`/`Multiply`/`Concatenate`/`BilinearResize` — left un-annotated
    like those.

    ----------
    Luo, H., Gu, Y., Liao, X., Lai, S. and Jiang, W., 2019. Bag of tricks
    and a strong baseline for deep person re-identification. CVPR
    Workshops.

    Input
    ----------
        input_tensor, embedding_size: the input tensor, and the output
                                      embedding's dimensionality.
        backbone, weights, freeze_backbone, freeze_batch_norm, deep_layer,
        pooling, size, activation, kernel_regularizer, bias_regularizer,
        dropout: identical meaning to `classifier()`'s own arguments.
        pre_pool_attention: as `classifier()`'s own argument, but defaults
                            to `'gem'` here (see above) instead of `None`.

    Output
    ----------
        A `tf.keras.Model` with a `.preprocessing` attribute, producing an
        L2-normalized `(batch, embedding_size)` embedding (`||embedding||_2
        == 1`, not a class probability). **Training**: call this same
        model 3 times (anchor/positive/negative crops), concatenate the 3
        `(batch, embedding_size)` outputs along the last axis into one
        `(batch, 3*embedding_size)` tensor, and pass that as `y_pred` to
        `losses.triplet_1d(y_true, y_pred, N=embedding_size)` — a
        training-loop concern, not built into this model's own graph (the
        same reasoning already established for why this repo's dynamic
        detection assigners aren't wired into a forced loss either).
        ArcFace/CosFace-style angular-margin losses are not implemented in
        this repo and not required — `triplet_1d` alone is sufficient to
        train this head.
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

    pre_head = qat_annotate_layer(tf.keras.layers.Dense(
        size,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
    ))(pool)

    drop_out = qat_annotate_layer(
        tf.keras.layers.Dropout(dropout))(pre_head)

    embedding = qat_annotate_layer(tf.keras.layers.Dense(
        embedding_size, activation=None,
        kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
        name='{}_embedding'.format(name),
    ))(drop_out)
    embedding = tf.keras.layers.UnitNormalization(axis=-1, name='{}_l2norm'.format(name))(embedding)

    res = tf.keras.models.Model(inputs=input_tensor, outputs=embedding)

    with tfmot.quantization.keras.quantize_scope():
        res = tfmot.quantization.keras.quantize_apply(res)

    res.preprocessing = backbone_.preprocessing

    return res
