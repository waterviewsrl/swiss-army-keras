'''Plain convolutional backbones for accelerators.

Four families built only from operators that camera accelerators run
natively: regular convolutions, ReLU or ReLU6, additions, concatenations and
max or average pooling. No depthwise-heavy blocks, squeeze-excite, attention,
hard-swish or GELU. They complement the phone-oriented families of the
library (MobileOne, RepViT, FastViT, MobileNetV4), following the guidance of
`CANDIDATES_BY_TASK.md`.

- **ResNet-18 and ResNet-34** (He et al., 2016): basic residual blocks.
  Not available in `tf.keras.applications`, which starts at ResNet-50.
- **STDC1 and STDC2** (Fan et al., 2021, "Rethinking BiSeNet for Real-time
  Semantic Segmentation"): short-term dense concatenation modules, the
  native backbone of PP-LiteSeg.
- **QARepVGG** (Chu et al., 2022, "Make RepVGG Greater Again"): a
  three-branch block (3x3 convolution with batch norm, 1x1 convolution,
  identity, then one batch norm on the sum) that fuses into a single 3x3
  convolution and, unlike the original RepVGG, is designed to survive 8-bit
  quantization. `reparameterize_qarepvgg` performs the fusion and
  `deploy_utils.reparameterize_model` applies it automatically at export.
- **CSP** : cross-stage-partial stages with a selectable activation, ReLU6
  by default: the backbone pattern Axis uses for its own detection models.
  Channel counts are multiples of 16.

All of them expose five feature levels (strides 2, 4, 8, 16, 32) and are
registered in `_backbone_zoo.BACKBONE_REGISTRY`.

**Padding.** Stride-2 convolutions use Keras `'same'` padding, which the
accelerators run without a separate padding operator. The PyTorch
references pad symmetrically, so for even input sizes the two differ by a
half-pixel shift. This matters only when porting reference weights: pass
`torch_padding=True` to the constructor to reproduce the reference
alignment with explicit padding.

**Weights.** Random initialization only (`weights=None`), or a path to a
`.weights.h5` file. No pretrained weights are shipped and the
implementations have not been compared numerically with the references;
the structure and the parameter counts of ResNet-18 and ResNet-34 match
torchvision.
'''

from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Cropping2D,
    Dense,
    DepthwiseConv2D,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Input,
    MaxPooling2D,
    ReLU,
    ZeroPadding2D,
)
from tensorflow.keras.models import Model

from swiss_army_keras.reparam_utils import fuse_conv_bn, fuse_conv_bn_layers, identity_kernel, pad_kernel_to

# Batch-norm constants of the PyTorch references (momentum 0.1 there).
_BN = dict(momentum=0.9, epsilon=1e-5)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def plain_preprocess_input(x):
    '''ImageNet normalization in the PyTorch convention: scale to `[0, 1]`,
    subtract the per-channel mean, divide by the per-channel standard
    deviation. Used by the ResNet, STDC and QARepVGG entries.
    '''
    x = x / 255.0
    return (x - _IMAGENET_MEAN) / _IMAGENET_STD


def csp_preprocess_input(x):
    '''Scale to `[0, 1]`, the convention of the detection networks the CSP
    backbone comes from.
    '''
    return x / 255.0


def _conv(x, filters, kernel_size, stride=1, torch_padding=False, use_bias=False, name=''):
    '''A convolution with `'same'` padding, or with the explicit symmetric
    padding of the PyTorch references when `torch_padding` is set and the
    stride is 2.
    '''
    if torch_padding and stride > 1 and kernel_size > 1:
        x = ZeroPadding2D(kernel_size // 2, name=f'{name}_pad')(x)
        padding = 'valid'
    else:
        padding = 'same'
    return Conv2D(filters, kernel_size, strides=stride, padding=padding, use_bias=use_bias,
                  kernel_initializer='he_normal', name=name)(x)


def _conv_bn_act(x, filters, kernel_size, stride=1, activation='relu', torch_padding=False, bn=_BN, name=''):
    x = _conv(x, filters, kernel_size, stride=stride, torch_padding=torch_padding, name=f'{name}_conv')
    x = BatchNormalization(name=f'{name}_bn', **bn)(x)
    if activation == 'relu6':
        return ReLU(max_value=6.0, name=f'{name}_relu6')(x)
    if activation is not None:
        return Activation(activation, name=f'{name}_{activation}')(x)
    return x


def _finish(x, include_top, pooling, classes, name):
    if include_top:
        x = GlobalAveragePooling2D(name=f'{name}_head_gap')(x)
        return Dense(classes, activation='softmax', name=f'{name}_head_dense')(x)
    if pooling == 'avg':
        return GlobalAveragePooling2D(name=f'{name}_pool')(x)
    if pooling == 'max':
        return GlobalMaxPooling2D(name=f'{name}_pool')(x)
    return x


def _constructor(network_fn, model_name, config_attr=None, config=None):
    '''Wrap a network function into a `tf.keras.applications`-style
    constructor, as `BackboneSpec` expects.
    '''
    def constructor(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling=None,
                    classes=1000, **kwargs):
        if weights == 'imagenet':
            raise NotImplementedError(
                "weights='imagenet' is not available for {}: no pretrained weights are shipped for the plain "
                "backbones. Pass weights=None, or the path of a `.weights.h5` file.".format(model_name))
        if input_tensor is None:
            if input_shape is None:
                raise ValueError('Either `input_tensor` or `input_shape` must be provided.')
            input_tensor = Input(shape=input_shape)
        output = network_fn(input_tensor, include_top=include_top, pooling=pooling, classes=classes, **kwargs)
        model = Model(inputs=input_tensor, outputs=output, name=model_name)
        if weights not in (None, 'none'):
            model.load_weights(weights, skip_mismatch=True)
        if config_attr is not None:
            setattr(model, config_attr, dict(config, **kwargs))
        return model

    constructor.__name__ = model_name
    return constructor


# ========================= #
# ResNet-18 / ResNet-34

RESNET_CONFIGS = {
    'ResNet18': dict(blocks=(2, 2, 2, 2), model_name='ResNet18'),
    'ResNet34': dict(blocks=(3, 4, 6, 3), model_name='ResNet34'),
}


def resnet_stage_output_names(model_name, blocks):
    '''Layer names of the five feature levels (strides 2 to 32).'''
    return (f'{model_name}_stem_relu',) + tuple(
        f'{model_name}_layer{stage + 1}_block{count - 1}_out' for stage, count in enumerate(blocks))


def _resnet_basic_block(x, filters, stride, torch_padding, name):
    shortcut = x
    if stride != 1 or x.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, use_bias=False, kernel_initializer='he_normal',
                          name=f'{name}_down_conv')(x)
        shortcut = BatchNormalization(name=f'{name}_down_bn', **_BN)(shortcut)
    h = _conv_bn_act(x, filters, 3, stride=stride, torch_padding=torch_padding, name=f'{name}_1')
    h = _conv_bn_act(h, filters, 3, activation=None, name=f'{name}_2')
    h = Add(name=f'{name}_add')([h, shortcut])
    return Activation('relu', name=f'{name}_out')(h)


def _resnet_network(input_tensor, blocks, model_name, include_top=False, pooling=None, classes=1000,
                    torch_padding=False):
    x = _conv_bn_act(input_tensor, 64, 7, stride=2, torch_padding=torch_padding, name=f'{model_name}_stem')
    if torch_padding:
        x = ZeroPadding2D(1, name=f'{model_name}_pool_pad')(x)
        x = MaxPooling2D(3, strides=2, padding='valid', name=f'{model_name}_pool')(x)
    else:
        x = MaxPooling2D(3, strides=2, padding='same', name=f'{model_name}_pool')(x)
    for stage, count in enumerate(blocks):
        filters = 64 * 2 ** stage
        for block in range(count):
            stride = 2 if (block == 0 and stage > 0) else 1
            x = _resnet_basic_block(x, filters, stride, torch_padding,
                                    name=f'{model_name}_layer{stage + 1}_block{block}')
    return _finish(x, include_top, pooling, classes, model_name)


def _build_resnet(blocks, model_name):
    def network(input_tensor, **kwargs):
        return _resnet_network(input_tensor, blocks, model_name, **kwargs)
    return _constructor(network, model_name)


ResNet18 = _build_resnet(**RESNET_CONFIGS['ResNet18'])
ResNet34 = _build_resnet(**RESNET_CONFIGS['ResNet34'])


# ========================= #
# STDC1 / STDC2

STDC_CONFIGS = {
    'STDC1': dict(blocks=(2, 2, 2), model_name='STDC1'),
    'STDC2': dict(blocks=(4, 5, 3), model_name='STDC2'),
}


def stdc_stage_output_names(model_name, blocks):
    '''Layer names of the five feature levels (strides 2 to 32).'''
    return (f'{model_name}_stage1_relu', f'{model_name}_stage2_relu') + tuple(
        f'{model_name}_stage{stage + 3}_block{count - 1}_concat' for stage, count in enumerate(blocks))


def _stdc_module(x, filters, stride, steps=4, name=''):
    '''One short-term dense concatenation module ("CatBottleneck" in the
    reference): a 1x1 convolution to half the output width, then `steps - 1`
    3x3 convolutions that halve the width each time (the last keeps it),
    and the concatenation of all their outputs. With `stride=2` the first
    3x3 convolution is preceded by a stride-2 depthwise convolution and the
    1x1 branch is average-pooled before the concatenation.
    '''
    out1 = _conv_bn_act(x, filters // 2, 1, name=f'{name}_conv0')
    outputs = []
    h = out1
    if stride == 2:
        h = DepthwiseConv2D(3, strides=2, padding='same', use_bias=False, name=f'{name}_avd_conv')(h)
        h = BatchNormalization(name=f'{name}_avd_bn', **_BN)(h)
        out1 = AveragePooling2D(3, strides=2, padding='same', name=f'{name}_skip_pool')(out1)
    for step in range(1, steps):
        width = filters // 2 ** (step + 1) if step < steps - 1 else filters // 2 ** step
        h = _conv_bn_act(h, width, 3, name=f'{name}_conv{step}')
        outputs.append(h)
    return Concatenate(name=f'{name}_concat')([out1] + outputs)


def _stdc_network(input_tensor, blocks, model_name, include_top=False, pooling=None, classes=1000, base=64):
    x = _conv_bn_act(input_tensor, base // 2, 3, stride=2, name=f'{model_name}_stage1')
    x = _conv_bn_act(x, base, 3, stride=2, name=f'{model_name}_stage2')
    for stage, count in enumerate(blocks):
        filters = base * 4 * 2 ** stage
        for block in range(count):
            x = _stdc_module(x, filters, stride=2 if block == 0 else 1,
                             name=f'{model_name}_stage{stage + 3}_block{block}')
    return _finish(x, include_top, pooling, classes, model_name)


def _build_stdc(blocks, model_name):
    def network(input_tensor, **kwargs):
        return _stdc_network(input_tensor, blocks, model_name, **kwargs)
    return _constructor(network, model_name)


STDC1 = _build_stdc(**STDC_CONFIGS['STDC1'])
STDC2 = _build_stdc(**STDC_CONFIGS['STDC2'])


# ========================= #
# QARepVGG

QAREPVGG_CONFIGS = {
    'QARepVGG-A0': dict(blocks=(2, 4, 14, 1), widths=(0.75, 0.75, 0.75, 2.5), model_name='QARepVGG_A0'),
    'QARepVGG-A1': dict(blocks=(2, 4, 14, 1), widths=(1.0, 1.0, 1.0, 2.5), model_name='QARepVGG_A1'),
    'QARepVGG-B0': dict(blocks=(4, 6, 16, 1), widths=(1.0, 1.0, 1.0, 2.5), model_name='QARepVGG_B0'),
}


def qarepvgg_stage_output_names(model_name, blocks):
    '''Layer names of the five feature levels (strides 2 to 32). The same
    names exist in the training-time and in the fused network.
    '''
    return (f'{model_name}_stem_relu',) + tuple(
        f'{model_name}_stage{stage + 1}_block{count - 1}_relu' for stage, count in enumerate(blocks))


def qarepvgg_block(x, filters, stride, deploy=False, name=''):
    '''One QARepVGG block.

    Training time: `ReLU(BN(BN(conv3x3(x)) + conv1x1(x) + x))`, the identity
    term only when the block keeps resolution and width. The 1x1 and
    identity branches carry no batch norm of their own; the single batch
    norm after the sum is what keeps the fused weights well conditioned for
    quantization. Deploy time (`deploy=True`): one 3x3 convolution with
    bias, named `<name>_fused`, and the ReLU.

    In a stride-2 block the 1x1 branch has to sample the pixels that sit
    under the centre of the 3x3 windows, or the fusion would not be exact.
    With Keras `'same'` padding and an even input size the 3x3 windows are
    centred on the odd rows and columns, so the 1x1 branch first drops the
    first row and column. This needs a static input size.
    '''
    if deploy:
        h = Conv2D(filters, 3, strides=stride, padding='same', use_bias=True, name=f'{name}_fused')(x)
        return Activation('relu', name=f'{name}_relu')(h)

    branches = []
    h = Conv2D(filters, 3, strides=stride, padding='same', use_bias=False, kernel_initializer='he_normal',
               name=f'{name}_conv3')(x)
    branches.append(BatchNormalization(name=f'{name}_conv3_bn', **_BN)(h))
    side = x
    if stride == 2:
        if x.shape[1] is None or x.shape[2] is None:
            raise ValueError('QARepVGG needs a static input size.')
        crop = ((1 - int(x.shape[1]) % 2, 0), (1 - int(x.shape[2]) % 2, 0))
        if crop != ((0, 0), (0, 0)):
            side = Cropping2D(crop, name=f'{name}_conv1_align')(x)
    branches.append(Conv2D(filters, 1, strides=stride, padding='valid', use_bias=False,
                           kernel_initializer='he_normal', name=f'{name}_conv1')(side))
    if stride == 1 and x.shape[-1] == filters:
        branches.append(x)
    h = Add(name=f'{name}_sum')(branches)
    h = BatchNormalization(name=f'{name}_post_bn', **_BN)(h)
    return Activation('relu', name=f'{name}_relu')(h)


def _qarepvgg_widths(widths):
    return min(64, int(64 * widths[0])), [int(64 * 2 ** i * w) for i, w in enumerate(widths)]


def _qarepvgg_network(input_tensor, blocks, widths, model_name, include_top=False, pooling=None, classes=1000,
                      deploy=False):
    stem, stage_widths = _qarepvgg_widths(widths)
    x = qarepvgg_block(input_tensor, stem, 2, deploy=deploy, name=f'{model_name}_stem')
    for stage, (count, filters) in enumerate(zip(blocks, stage_widths)):
        for block in range(count):
            x = qarepvgg_block(x, filters, 2 if block == 0 else 1, deploy=deploy,
                               name=f'{model_name}_stage{stage + 1}_block{block}')
    return _finish(x, include_top, pooling, classes, model_name)


def _build_qarepvgg(blocks, widths, model_name):
    def network(input_tensor, **kwargs):
        return _qarepvgg_network(input_tensor, blocks, widths, model_name, **kwargs)
    return _constructor(network, model_name, config_attr='_qarepvgg_config',
                        config=dict(blocks=blocks, widths=widths, model_name=model_name))


QARepVGG_A0 = _build_qarepvgg(**QAREPVGG_CONFIGS['QARepVGG-A0'])
QARepVGG_A1 = _build_qarepvgg(**QAREPVGG_CONFIGS['QARepVGG-A1'])
QARepVGG_B0 = _build_qarepvgg(**QAREPVGG_CONFIGS['QARepVGG-B0'])


def fuse_qarepvgg_block(conv3, conv3_bn, conv1, post_bn, has_identity):
    '''Fused `(kernel, bias)` of one QARepVGG block, from its layers.

    The 3x3 branch is folded with its own batch norm; the 1x1 kernel is
    padded to 3x3 and the identity added as a 3x3 kernel with a one in the
    centre; the sum is then folded with the batch norm that follows it.
    '''
    kernel, bias = fuse_conv_bn_layers(conv3, conv3_bn)
    kernel = kernel + pad_kernel_to(conv1.get_weights()[0], 3)
    if has_identity:
        kernel = kernel + identity_kernel(kernel.shape[-1], 3, dtype=kernel.dtype)
    gamma, beta, mean, variance = post_bn.get_weights()
    return fuse_conv_bn(kernel, bias, gamma, beta, mean, variance, epsilon=post_bn.epsilon)


def reparameterize_qarepvgg(model):
    '''Fuse every block of a QARepVGG model into a single 3x3 convolution.

    Input
    ----------
        model: a model built by `QARepVGG_A0` / `QARepVGG_A1` /
               `QARepVGG_B0` (it carries a `_qarepvgg_config` attribute), or
               the nested backbone `backbone_zoo` builds from one.

    Output
    ----------
        A new model computing the same function (to floating-point
        tolerance) with one convolution and one ReLU per block. The layers
        that end each stage keep their names, so feature taps are found by
        the same names.
    '''
    if not hasattr(model, '_qarepvgg_config'):
        raise ValueError('`model` has no `_qarepvgg_config`: reparameterize_qarepvgg only supports models built by '
                         'the QARepVGG constructors of this module.')
    config = model._qarepvgg_config
    name = config['model_name']
    deploy_input = Input(shape=model.input_shape[1:])
    deploy_output = _qarepvgg_network(deploy_input, config['blocks'], config['widths'], name, deploy=True)
    deploy_model = Model(deploy_input, deploy_output, name=f'{name}_fused')

    for layer in deploy_model.layers:
        if not layer.name.endswith('_fused'):
            continue
        prefix = layer.name[:-len('_fused')]
        conv3 = model.get_layer(f'{prefix}_conv3')
        has_identity = len(model.get_layer(f'{prefix}_sum').input) == 3
        kernel, bias = fuse_qarepvgg_block(
            conv3, model.get_layer(f'{prefix}_conv3_bn'), model.get_layer(f'{prefix}_conv1'),
            model.get_layer(f'{prefix}_post_bn'), has_identity)
        layer.set_weights([kernel, bias])
    deploy_model._qarepvgg_config = config
    return deploy_model


# ========================= #
# CSP backbone

CSP_CONFIGS = {
    'CSP-N': dict(width=0.25, depth=1 / 3, model_name='CSP_N'),
    'CSP-S': dict(width=0.50, depth=1 / 3, model_name='CSP_S'),
    'CSP-M': dict(width=0.75, depth=2 / 3, model_name='CSP_M'),
}

# momentum 0.03 and epsilon 1e-3 in the PyTorch detection references
_CSP_BN = dict(momentum=0.97, epsilon=1e-3)


def csp_stage_output_names(model_name):
    '''Layer names of the five feature levels (strides 2 to 32).'''
    act = 'relu6'
    return (f'{model_name}_stem_{act}',) + tuple(
        f'{model_name}_stage{stage}_out_{act}' for stage in (1, 2, 3)) + (f'{model_name}_sppf_out_{act}',)


def _csp_stage(x, filters, repeats, activation, name):
    '''A cross-stage-partial stage: the input is projected twice to half
    the width, one projection runs through `repeats` residual bottlenecks
    (1x1 then 3x3), and the two are concatenated and merged by a 1x1
    convolution.
    '''
    half = filters // 2
    main = _conv_bn_act(x, half, 1, activation=activation, bn=_CSP_BN, name=f'{name}_main')
    for i in range(repeats):
        h = _conv_bn_act(main, half, 1, activation=activation, bn=_CSP_BN, name=f'{name}_b{i}_1')
        h = _conv_bn_act(h, half, 3, activation=activation, bn=_CSP_BN, name=f'{name}_b{i}_2')
        main = Add(name=f'{name}_b{i}_add')([main, h])
    side = _conv_bn_act(x, half, 1, activation=activation, bn=_CSP_BN, name=f'{name}_side')
    merged = Concatenate(name=f'{name}_concat')([main, side])
    return _conv_bn_act(merged, filters, 1, activation=activation, bn=_CSP_BN, name=f'{name}_out')


def _csp_network(input_tensor, width, depth, model_name, include_top=False, pooling=None, classes=1000,
                 activation='relu6', channel_multiple=16):
    if activation != 'relu6':
        raise ValueError("The registered CSP backbones use activation='relu6'; other activations change the "
                         "names of the feature layers and are not supported through this constructor.")

    def channels(base):
        return max(channel_multiple, int(round(base * width / channel_multiple)) * channel_multiple)

    def repeats(base):
        return max(1, int(round(base * depth)))

    x = _conv_bn_act(input_tensor, channels(64), 3, stride=2, activation=activation, bn=_CSP_BN,
                     name=f'{model_name}_stem')
    for stage, (base_channels, base_repeats) in enumerate(((128, 3), (256, 6), (512, 9), (1024, 3)), start=1):
        x = _conv_bn_act(x, channels(base_channels), 3, stride=2, activation=activation, bn=_CSP_BN,
                         name=f'{model_name}_stage{stage}_down')
        x = _csp_stage(x, channels(base_channels), repeats(base_repeats), activation, name=f'{model_name}_stage{stage}')

    # fast spatial pyramid pooling: three cascaded 5x5 max pools, concatenated
    filters = int(x.shape[-1])
    h = _conv_bn_act(x, filters // 2, 1, activation=activation, bn=_CSP_BN, name=f'{model_name}_sppf_in')
    pools = [h]
    for i in range(3):
        pools.append(MaxPooling2D(5, strides=1, padding='same', name=f'{model_name}_sppf_pool{i}')(pools[-1]))
    x = Concatenate(name=f'{model_name}_sppf_concat')(pools)
    x = _conv_bn_act(x, filters, 1, activation=activation, bn=_CSP_BN, name=f'{model_name}_sppf_out')
    return _finish(x, include_top, pooling, classes, model_name)


def _build_csp(width, depth, model_name):
    def network(input_tensor, **kwargs):
        return _csp_network(input_tensor, width, depth, model_name, **kwargs)
    return _constructor(network, model_name)


CSP_N = _build_csp(**CSP_CONFIGS['CSP-N'])
CSP_S = _build_csp(**CSP_CONFIGS['CSP-S'])
CSP_M = _build_csp(**CSP_CONFIGS['CSP-M'])
