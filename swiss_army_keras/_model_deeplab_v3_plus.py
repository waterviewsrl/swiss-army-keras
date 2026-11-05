
from __future__ import absolute_import

from swiss_army_keras.layer_utils import *
from swiss_army_keras.activations import GELU, Snake
from swiss_army_keras._backbone_zoo import backbone_zoo, bach_norm_checker
from swiss_army_keras._model_unet_2d import UNET_left, UNET_right

from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, AveragePooling2D, UpSampling2D, Concatenate, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal

from swiss_army_keras.utils import freeze_model

from tensorflow.nn import relu

import tensorflow as tf

import tensorflow_model_optimization as tfmot


def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = qat_annotate_layer(Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=HeNormal(),
    ))(block_input)
    x = qat_annotate_layer(BatchNormalization())(x)
    return qat_annotate_layer(Activation(relu))(x)


def depth_convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
    stride=1,
    depth_padding='same',
    epsilon=1e-5,
    namesuffix=''
):

    x = qat_annotate_layer(DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(dilation_rate, dilation_rate),
                        padding=depth_padding, use_bias=False, name=f'depthwise_{dilation_rate}_{namesuffix}'))(block_input)
    x = qat_annotate_layer(BatchNormalization(
        name=f'depthwise_BN__{dilation_rate}_{namesuffix}', epsilon=epsilon))(x)
    x = qat_annotate_layer(Activation(relu))(x)
    x = qat_annotate_layer(Conv2D(num_filters, (1, 1), padding='same',
               use_bias=False, name=f'pointwise_{dilation_rate}_{namesuffix}'))(x)
    x = qat_annotate_layer(BatchNormalization(
        name=f'pointwise_BN_{dilation_rate}_{namesuffix}', epsilon=epsilon))(x)
    x = qat_annotate_layer(Activation(relu))(x)
    return x


def DilatedSpatialPyramidPooling(dspp_input, atrous_rates, num_filters):
    dims = dspp_input.shape
    x = qat_annotate_layer(AveragePooling2D(pool_size=(dims[-3], dims[-2])))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = qat_annotate_layer(UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    ))(x)

    out_1 = convolution_block(
        dspp_input, kernel_size=1, dilation_rate=1, num_filters=num_filters)

    outputs = [out_pool, out_1]

    for rate in atrous_rates:
        out = depth_convolution_block(
            dspp_input, kernel_size=3, dilation_rate=rate, num_filters=num_filters)
        outputs.append(out)

    x = Concatenate(axis=-1)(outputs)
    output = convolution_block(x, kernel_size=1, num_filters=num_filters)
    output = qat_annotate_layer(Dropout(0.1))(output)
    return output


def deeplab_v3_plus(input_tensor, n_labels, filter_num_down=[64, 128, 256, 512, 1024],
                    deep_layer=5, shallow_layer=2, num_filters_deep=256, num_filters_shallow=48, multiscale_factor=0, atrous_rates=[6, 12, 18],
                    stack_num_down=2, stack_num_up=1, activation='ReLU', batch_norm=False, pool=True, unpool=True,
                    backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='deeplab_v3_plus'):
    '''
    DeepLabV3+ semantic segmentation model: backbone, atrous spatial pyramid
    pooling (ASPP) on a deep feature, fusion with a shallow feature, and
    bilinear upsampling back to the input resolution.

    ----------
    Chen, L.C., Zhu, Y., Papandreou, G., Schroff, F. and Adam, H., 2018.
    Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation. In ECCV 2018.

    Input
    ----------
        input_tensor: the input tensor, e.g. `keras.layers.Input((512, 512, 3))`.
                      A fixed spatial size is required (upsampling factors
                      are computed from static shapes).
        n_labels: number of segmentation classes (output channels).
        filter_num_down: only used when `backbone=None`: number of filters
                         of each level of the built-in plain encoder.
        deep_layer: 1-based index of the backbone tap fed to the ASPP module
                    (5 = stride 32 for 5-tap backbones; use 4 for the 4-tap
                    RepViT/FastViT families).
        shallow_layer: 1-based index of the tap used for the low-level skip
                       connection.
        num_filters_deep: channels of the ASPP branches and its output.
        num_filters_shallow: channels of the 1x1 projection of the shallow
                             tap.
        multiscale_factor: 0 (default) disables it. Otherwise a second
                           `MobileNetV3Large` encoder is run on the input
                           downscaled by this factor and its deep feature is
                           concatenated to the main one before the ASPP.
        atrous_rates: dilation rates of the ASPP 3x3 branches. Dilated
                      convolutions are exported to TFLite as
                      SPACE_TO_BATCH_ND / BATCH_TO_SPACE_ND pairs, which many
                      NPUs do not accelerate; prefer `lr_aspp`/`pp_liteseg`
                      for such targets.
        stack_num_down, stack_num_up, activation, batch_norm, pool, unpool:
                      only used by the `backbone=None` encoder.
        name: prefix of the created layers.

        ---------- (keywords of backbone options) ----------
        backbone: name of a registered backbone (any key of
                  `swiss_army_keras._backbone_zoo.BACKBONE_REGISTRY`, or a
                  `hub://` name), or None for a plain built-in encoder.
        weights: None (random initialization), 'imagenet' (only for the
                 `tf.keras.applications` backbones), or the path of a weights
                 file.
        freeze_backbone: True for a frozen backbone.
        freeze_batch_norm: False for not freezing batch normalization layers.

    Output
    ----------
        a QAT-wrapped `tf.keras.Model` returning raw logits
        (no softmax; use `losses.loss_adapter(..., from_logits=True)`), with a
        `preprocessing` attribute holding the backbone's input normalization.
        Output shape: `(batch, H, W, n_labels)`, same spatial size as the input.

    '''

    depth_ = len(filter_num_down)

    X_encoder = []

    multiscale_resizing = None
    X_encoder_small = None

    # no backbone cases
    if backbone is None:

        X = input_tensor

        # stacked conv2d before downsampling
        X = CONV_stack(X, filter_num_down[0], kernel_size=3, stack_num=stack_num_down,
                       activation=activation, batch_norm=batch_norm, name='{}_down0'.format(name))
        X_encoder.append(X)

        # downsampling levels
        for i, f in enumerate(filter_num_down[1:]):

            # UNET-like downsampling
            X = UNET_left(X, f, kernel_size=3, stack_num=stack_num_down, activation=activation,
                          pool=pool, batch_norm=batch_norm, name='{}_down{}'.format(name, i+1))
            X_encoder.append(X)

        preprocessing = dummy_preprocessing

    else:
        # handling VGG16 and VGG19 separately
        if 'VGG' in backbone:
            backbone_ = backbone_zoo(
                backbone, weights, input_tensor, depth_, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_encoder = backbone_([input_tensor, ])
            depth_encode = len(X_encoder)

            preprocessing = backbone_.preprocessing

        # for other backbones
        else:

            if multiscale_factor != 0:
                multiscale_resizing = tf.keras.layers.Resizing(int(
                    input_tensor.shape[1]/multiscale_factor), int(input_tensor.shape[2]/multiscale_factor))(input_tensor)
                backbone_small_, _ = backbone_zoo(
                    'MobileNetV3Large', weights, multiscale_resizing, deep_layer, freeze_backbone, freeze_batch_norm, return_outputs=True)
                print(backbone_small_[deep_layer-1])
                backbone, preprocessing = backbone_zoo(
                    backbone, weights, input_tensor, deep_layer, freeze_backbone, freeze_batch_norm, return_outputs=True)
                X_encoder = backbone[deep_layer-1]
                X_encoder_shallow = backbone[shallow_layer-1]
                # X_encoder_small = backbone_small_[deep
                #X_encoder_small = backbone_small_[deep_layer-1]([multiscale_resizing, ])

            else:
                backbone_ = backbone_zoo(
                    backbone, weights, input_tensor, deep_layer, freeze_backbone, freeze_batch_norm)
                # collecting backbone feature maps
                X_encoder = backbone_([input_tensor, ])
                preprocessing = backbone_.preprocessing

                depth_encode = len(X_encoder) + 1

    if multiscale_factor != 0:
        X_encoder_back = tf.keras.layers.Resizing(
            X_encoder.shape[1], X_encoder.shape[2])(backbone_small_[deep_layer-1])
        #X_encoder_back = X_encoder[deep_layer-1]
        x = Concatenate(axis=-1)([X_encoder, X_encoder_back])
        #x = X_encoder_back
        print(X_encoder)
        print(X_encoder_back)
        print(x)
        x = Model([input_tensor, ], [x, ])
        if freeze_backbone:
            x = freeze_model(x, freeze_batch_norm=freeze_batch_norm)
        x = x([input_tensor, ])
    else:
        x = X_encoder[deep_layer-1]

    x = DilatedSpatialPyramidPooling(
        x, atrous_rates, num_filters=num_filters_deep)

    if multiscale_factor != 0:
        input_b = X_encoder_shallow
    else:
        input_b = X_encoder[shallow_layer-1]

    # Upsample the deep branch to match the shallow branch's *actual* spatial
    # size, read directly off `input_b`, rather than deriving it from
    # `shallow_resize_map[shallow_layer]` — that map hardcodes a 2**shallow_layer
    # downsampling factor that only holds for the `backbone=<name>` path
    # (backbone_zoo's returned feature list has one pooling "baked into" its
    # first entry). The no-backbone encoder above builds `X_encoder[0]` with
    # *zero* poolings applied, so `X_encoder[shallow_layer-1]` there is
    # downsampled by 2**(shallow_layer-1), one factor of 2 less than the map
    # assumes — which made this `Concatenate` fail with a shape mismatch for
    # any `backbone=None` call (see plan.md §10/§13 Phase 3). Reading the
    # target size from `input_b` directly is correct for both encoder paths,
    # and any future encoder, without needing a lookup table at all.
    input_a = qat_annotate_layer(UpSampling2D(
        size=(input_b.shape[1] // x.shape[1],
              input_b.shape[2] // x.shape[2]),
        interpolation="bilinear",
    ))(x)

    input_b = convolution_block(
        input_b, num_filters=num_filters_shallow, kernel_size=1)

    # Both left un-annotated here in the past, unlike `deeplab_v3_plus_lite`'s
    # otherwise-identical pattern below - not a deliberate divergence (no
    # "not tfmot-safe" finding backs it, the way the `Multiply`/`Add`/
    # standalone-`BatchNormalization` exclusions elsewhere in this repo are
    # documented), just drift between the two decoders. Wrapping both here
    # for QAT-annotation parity fixes a real, separate bug it was hiding:
    # under a global mixed-precision policy (`Policy('mixed_float16')`, e.g.
    # to speed up backbone training - backbones are never QAT-annotated, see
    # `backbone_zoo`, so this doesn't help the un-annotated layers below),
    # `model_output`'s dtype fell out as float16 (Keras's own guidance is
    # that a model's final output layer should stay float32 for numerical
    # stability) - `qat_annotate_layer` forces float32 regardless of the
    # ambient policy (see its docstring), fixing this incidentally.
    x = qat_annotate_layer(Concatenate(axis=-1))([input_a, input_b])
    x = depth_convolution_block(x, namesuffix='shallow1')
    x = depth_convolution_block(x, namesuffix='shallow2')
    x = qat_annotate_layer(UpSampling2D(
        size=(input_tensor.shape[1] // x.shape[1],
              input_tensor.shape[2] // x.shape[2]),
        interpolation="bilinear",
    ))(x)
    model_output = qat_annotate_layer(Conv2D(n_labels, kernel_size=(1, 1), padding="same"))(x)

    m = Model([input_tensor, ], [model_output, ])

    with tfmot.quantization.keras.quantize_scope():
        m = tfmot.quantization.keras.quantize_apply(m)

    m.preprocessing = preprocessing

    return m


def deeplab_v3_plus_lite(input_tensor, n_labels, filter_num_down=[64, 128, 256, 512, 1024],
                    deep_layer=5, shallow_layer=2, num_filters_deep=256, num_filters_shallow=48, multiscale_factor=0, atrous_rates=[6, 12, 18],
                    stack_num_down=2, stack_num_up=1, activation='ReLU', batch_norm=False, pool=True, unpool=True,
                    backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='deeplab_v3_plus'):
    '''
    DeepLabV3+ without the final upsampling stage: identical to
    `deeplab_v3_plus` up to the fused decoder feature, but the logits are
    returned at the resolution of the shallow tap (e.g. 1/4 of the input for
    `shallow_layer=2`) instead of being upsampled to the input size. This
    removes the most expensive resize from the deployed graph; upsample the
    mask on the host side, and train with `mask_downsample` set accordingly
    in `SegmentationAlbumentationsDataLoader` (or
    `losses.loss_adapter(..., contract=True)`).

    ----------
    Chen, L.C., Zhu, Y., Papandreou, G., Schroff, F. and Adam, H., 2018.
    Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation. In ECCV 2018.

    Input
    ----------
        input_tensor: the input tensor, e.g. `keras.layers.Input((512, 512, 3))`.
                      A fixed spatial size is required (upsampling factors
                      are computed from static shapes).
        n_labels: number of segmentation classes (output channels).
        filter_num_down: only used when `backbone=None`: number of filters
                         of each level of the built-in plain encoder.
        deep_layer: 1-based index of the backbone tap fed to the ASPP module
                    (5 = stride 32 for 5-tap backbones; use 4 for the 4-tap
                    RepViT/FastViT families).
        shallow_layer: 1-based index of the tap used for the low-level skip
                       connection.
        num_filters_deep: channels of the ASPP branches and its output.
        num_filters_shallow: channels of the 1x1 projection of the shallow
                             tap.
        multiscale_factor: 0 (default) disables it. Otherwise a second
                           `MobileNetV3Large` encoder is run on the input
                           downscaled by this factor and its deep feature is
                           concatenated to the main one before the ASPP.
        atrous_rates: dilation rates of the ASPP 3x3 branches. Dilated
                      convolutions are exported to TFLite as
                      SPACE_TO_BATCH_ND / BATCH_TO_SPACE_ND pairs, which many
                      NPUs do not accelerate; prefer `lr_aspp`/`pp_liteseg`
                      for such targets.
        stack_num_down, stack_num_up, activation, batch_norm, pool, unpool:
                      only used by the `backbone=None` encoder.
        name: prefix of the created layers.

        ---------- (keywords of backbone options) ----------
        backbone: name of a registered backbone (any key of
                  `swiss_army_keras._backbone_zoo.BACKBONE_REGISTRY`, or a
                  `hub://` name), or None for a plain built-in encoder.
        weights: None (random initialization), 'imagenet' (only for the
                 `tf.keras.applications` backbones), or the path of a weights
                 file.
        freeze_backbone: True for a frozen backbone.
        freeze_batch_norm: False for not freezing batch normalization layers.

    Output
    ----------
        a QAT-wrapped `tf.keras.Model` returning raw logits
        (no softmax; use `losses.loss_adapter(..., from_logits=True)`), with a
        `preprocessing` attribute holding the backbone's input normalization.
        Output shape: `(batch, H / s, W / s, n_labels)` where `s` is the
        stride of the shallow tap.

    '''

    depth_ = len(filter_num_down)

    X_encoder = []

    multiscale_resizing = None
    X_encoder_small = None

    # no backbone cases
    if backbone is None:

        X = input_tensor

        # stacked conv2d before downsampling
        X = CONV_stack(X, filter_num_down[0], kernel_size=3, stack_num=stack_num_down,
                       activation=activation, batch_norm=batch_norm, name='{}_down0'.format(name))
        X_encoder.append(X)

        # downsampling levels
        for i, f in enumerate(filter_num_down[1:]):

            # UNET-like downsampling
            X = UNET_left(X, f, kernel_size=3, stack_num=stack_num_down, activation=activation,
                          pool=pool, batch_norm=batch_norm, name='{}_down{}'.format(name, i+1))
            X_encoder.append(X)

        preprocessing = dummy_preprocessing

    else:
        # handling VGG16 and VGG19 separately
        if 'VGG' in backbone:
            backbone_ = backbone_zoo(
                backbone, weights, input_tensor, depth_, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_encoder = backbone_([input_tensor, ])
            depth_encode = len(X_encoder)

            preprocessing = backbone_.preprocessing

        # for other backbones
        else:

            if multiscale_factor != 0:
                multiscale_resizing = tf.keras.layers.Resizing(int(
                    input_tensor.shape[1]/multiscale_factor), int(input_tensor.shape[2]/multiscale_factor))(input_tensor)
                backbone_small_, _ = backbone_zoo(
                    'MobileNetV3Large', weights, multiscale_resizing, deep_layer, freeze_backbone, freeze_batch_norm, return_outputs=True)
                print(backbone_small_[deep_layer-1])
                backbone, preprocessing = backbone_zoo(
                    backbone, weights, input_tensor, deep_layer, freeze_backbone, freeze_batch_norm, return_outputs=True)
                X_encoder = backbone[deep_layer-1]
                X_encoder_shallow = backbone[shallow_layer-1]
                # X_encoder_small = backbone_small_[deep
                #X_encoder_small = backbone_small_[deep_layer-1]([multiscale_resizing, ])

            else:
                backbone_ = backbone_zoo(
                    backbone, weights, input_tensor, deep_layer, freeze_backbone, freeze_batch_norm)
                # collecting backbone feature maps
                X_encoder = backbone_([input_tensor, ])
                preprocessing = backbone_.preprocessing

                depth_encode = len(X_encoder) + 1

    if multiscale_factor != 0:
        X_encoder_back = tf.keras.layers.Resizing(
            X_encoder.shape[1], X_encoder.shape[2])(backbone_small_[deep_layer-1])
        #X_encoder_back = X_encoder[deep_layer-1]
        x = Concatenate(axis=-1)([X_encoder, X_encoder_back])
        #x = X_encoder_back
        print(X_encoder)
        print(X_encoder_back)
        print(x)
        x = Model([input_tensor, ], [x, ])
        if freeze_backbone:
            x = freeze_model(x, freeze_batch_norm=freeze_batch_norm)
        x = x([input_tensor, ])
    else:
        x = X_encoder[deep_layer-1]

    x = DilatedSpatialPyramidPooling(
        x, atrous_rates, num_filters=num_filters_deep)

    if multiscale_factor != 0:
        input_b = X_encoder_shallow
    else:
        input_b = X_encoder[shallow_layer-1]

    # See the identical fix (and full explanation) in `deeplab_v3_plus` above
    # — `shallow_resize_map[shallow_layer]` assumed a downsampling factor
    # that only holds for the `backbone=<name>` path; reading the shallow
    # branch's actual spatial size directly off `input_b` is correct for the
    # no-backbone path too.
    input_a = qat_annotate_layer(UpSampling2D(
        size=(input_b.shape[1] // x.shape[1],
              input_b.shape[2] // x.shape[2]),
        interpolation="bilinear",
    ))(x)

    input_b = convolution_block(
        input_b, num_filters=num_filters_shallow, kernel_size=1)

    x = qat_annotate_layer(Concatenate(axis=-1))([input_a, input_b])
    x = depth_convolution_block(x, namesuffix='shallow1')
    x = depth_convolution_block(x, namesuffix='shallow2')
    """x = UpSampling2D(
        size=(input_tensor.shape[1] // x.shape[1],
              input_tensor.shape[2] // x.shape[2]),
        interpolation="bilinear",
    )(x)"""
    model_output = qat_annotate_layer(Conv2D(n_labels, kernel_size=(1, 1), padding="same"))(x)

    m = Model([input_tensor, ], [model_output, ])

    with tfmot.quantization.keras.quantize_scope():
        m = tfmot.quantization.keras.quantize_apply(m)

    m.preprocessing = preprocessing

    return m

