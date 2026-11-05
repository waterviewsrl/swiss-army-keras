'''Shared building blocks: QAT annotation helper, encoder/decoder convolution
stacks, attention gate, ASPP, attention pooling layers (GeM, ECA, CoordAtt),
the IQA regression head, the uint8 input wrapper and energy-score outputs.
'''
from __future__ import absolute_import

import math

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from swiss_army_keras._compat import ensure_legacy_keras

# Every decoder/head/backbone module imports this file, so this is the one
# place a Keras-3-active process is caught before any QAT wrapping happens.
ensure_legacy_keras()

from swiss_army_keras.activations import GELU, Snake
from tensorflow import expand_dims
from tensorflow.compat.v1 import image
from tensorflow.keras.layers import Input, MaxPooling2D, AveragePooling2D, UpSampling2D, Conv2DTranspose, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, multiply, add
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU, Softmax
from tensorflow.keras.models import Model


def qat_annotate_layer(layer):
    '''Wrap `layer` for QAT (in place of a bare
    `tfmot.quantization.keras.quantize_annotate_layer` call), forcing float32
    compute on it regardless of any ambient mixed-precision policy (e.g.
    `tf.keras.mixed_precision.set_global_policy('mixed_float16')`, set by a
    caller to speed up backbone training on a Tensor-Core GPU - backbones are
    never QAT-annotated, see `backbone_zoo`, so they benefit from such a
    policy freely).

    `tfmot`'s fake-quantization op (`FakeQuantWithMinMaxVars`, used by every
    QAT-wrapped layer's output quantizer) hard-requires float32 inputs -
    `TypeError: Input 'inputs' of 'FakeQuantWithMinMaxVars' Op has type
    float16 that does not match expected type of float32` - confirmed
    empirically by actually building a decoder under a `mixed_float16`
    policy, not documented anywhere in `tfmot` itself. Every decoder/head
    builder function in this repo unconditionally ends with `quantize_apply`
    on its `quantize_annotate_layer`-wrapped layers (no opt-out), so without
    this, setting a global mixed-precision policy anywhere in a process
    would break every one of them.

    Two dtype-policy overrides are needed, not one: forcing the raw `layer`
    itself is not enough, since the `QuantizeAnnotate` wrapper this returns
    calls `self.layer.call(...)` directly (bypassing the normal
    `Layer.__call__` auto-cast machinery), so the *wrapper's own* policy is
    what actually determines the dtype its output gets cast to before
    reaching the inner layer. A third override, for the separate
    `QuantizeWrapperV2` that `quantize_apply()` constructs later (at model-
    build time, not here), is handled once globally by
    `_force_quantize_wrapper_dtype_float32()` below, since that wrapper is
    built internally by `tfmot`, not by this function.
    '''
    layer._set_dtype_policy(tf.keras.mixed_precision.Policy('float32'))
    wrapped = tfmot.quantization.keras.quantize_annotate_layer(layer)
    wrapped._set_dtype_policy(tf.keras.mixed_precision.Policy('float32'))
    return wrapped


def _force_quantize_wrapper_dtype_float32():
    '''Patch `tfmot`'s `QuantizeWrapper` (base class of `QuantizeWrapperV2`,
    which `quantize_apply()` constructs internally for every
    `quantize_annotate_layer`-wrapped layer when a model is finished being
    built) and `QuantizeLayer` (a separate, plain-`Layer` class `tfmot` adds
    to quantize a model's raw *input* tensor, seen so far only on `ddrnet` -
    its first op reads the input directly, unlike this repo's other
    decoders) so every instance of either is forced to float32 compute, the
    same way `qat_annotate_layer` above forces its own two wrappers - see
    that function's docstring for why. Applied once, at import time of this
    module (imported, directly or via a `from .layer_utils import *`, by
    every decoder/head file that uses QAT), rather than per-layer, since
    `quantize_apply()` builds these objects itself; there is no per-call-site
    hook for this repo's code to intervene at.
    '''
    from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
    from tensorflow_model_optimization.python.core.quantization.keras.quantize_layer import QuantizeLayer

    for cls in (QuantizeWrapper, QuantizeLayer):
        if getattr(cls, '_sak_float32_patched', False):
            continue
        _orig_init = cls.__init__

        def _patched_init(self, *args, _orig_init=_orig_init, **kwargs):
            _orig_init(self, *args, **kwargs)
            self._set_dtype_policy(tf.keras.mixed_precision.Policy('float32'))
        cls.__init__ = _patched_init
        cls._sak_float32_patched = True


_force_quantize_wrapper_dtype_float32()


def decode_layer(X, channel, pool_size, unpool, kernel_size=3,
                 activation='ReLU', batch_norm=False, name='decode'):
    '''
    An overall decode layer, based on either upsampling or trans conv.
    
    decode_layer(X, channel, pool_size, unpool, kernel_size=3,
                 activation='ReLU', batch_norm=False, name='decode')
    
    Input
    ----------
        X: input tensor.
        pool_size: the decoding factor.
        channel: (for trans conv only) number of convolution filters.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.           
        kernel_size: size of convolution kernels. 
                     If kernel_size='auto', then it equals to the `pool_size`.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
    
    * The defaut: `kernel_size=3`, is suitable for `pool_size=2`.
    
    '''
    # parsers
    if unpool is False:
        # trans conv configurations
        bias_flag = not batch_norm
    
    elif unpool == 'nearest':
        # upsample2d configurations
        unpool = True
        interp = 'nearest'
    
    elif (unpool is True) or (unpool == 'bilinear'):
        # upsample2d configurations
        unpool = True
        interp = 'bilinear'
    
    else:
        raise ValueError('Invalid unpool keyword')
        
    if unpool:
        X = UpSampling2D(size=(pool_size, pool_size), interpolation=interp, name='{}_unpool'.format(name))(X)
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size
            
        X = Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size), 
                            padding='same', name='{}_trans_conv'.format(name))(X)
        
        # batch normalization
        if batch_norm:
            X = BatchNormalization(axis=3, name='{}_bn'.format(name))(X)
            
        # activation
        if activation is not None:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)
        
    return X

def encode_layer(X, channel, pool_size, pool, kernel_size='auto', 
                 activation='ReLU', batch_norm=False, name='encode'):
    '''
    An overall encode layer, based on one of the:
    (1) max-pooling, (2) average-pooling, (3) strided conv2d.
    
    encode_layer(X, channel, pool_size, pool, kernel_size='auto', 
                 activation='ReLU', batch_norm=False, name='encode')
    
    Input
    ----------
        X: input tensor.
        pool_size: the encoding factor.
        channel: (for strided conv only) number of convolution filters.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        kernel_size: size of convolution kernels. 
                     If kernel_size='auto', then it equals to the `pool_size`.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    '''
    # parsers
    if (pool in [False, True, 'max', 'ave']) is not True:
        raise ValueError('Invalid pool keyword')
        
    # maxpooling2d as default
    if pool is True:
        pool = 'max'
        
    elif pool is False:
        # stride conv configurations
        bias_flag = not batch_norm
    
    if pool == 'max':
        X = MaxPooling2D(pool_size=(pool_size, pool_size), name='{}_maxpool'.format(name))(X)
        
    elif pool == 'ave':
        X = AveragePooling2D(pool_size=(pool_size, pool_size), name='{}_avepool'.format(name))(X)
        
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size
        
        # linear convolution with strides
        X = Conv2D(channel, kernel_size, strides=(pool_size, pool_size), 
                   padding='valid', use_bias=bias_flag, name='{}_stride_conv'.format(name))(X)
        
        # batch normalization
        if batch_norm:
            X = BatchNormalization(axis=3, name='{}_bn'.format(name))(X)
            
        # activation
        if activation is not None:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)
            
    return X

def attention_gate(X, g, channel,  
                   activation='ReLU', 
                   attention='add', name='att'):
    '''
    Self-attention gate modified from Oktay et al. 2018.
    
    attention_gate(X, g, channel,  activation='ReLU', attention='add', name='att')
    
    Input
    ----------
        X: input tensor, i.e., key and value.
        g: gated tensor, i.e., query.
        channel: number of intermediate channel.
                 Oktay et al. (2018) did not specify (denoted as F_int).
                 intermediate channel is expected to be smaller than the input channel.
        activation: a nonlinear attnetion activation.
                    The `sigma_1` in Oktay et al. 2018. Default is 'ReLU'.
        attention: 'add' for additive attention; 'multiply' for multiplicative attention.
                   Oktay et al. 2018 applied additive attention.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X_att: output tensor.
    
    '''
    activation_func = eval(activation)
    attention_func = eval(attention)
    
    # mapping the input tensor to the intermediate channel
    theta_att = Conv2D(channel, 1, use_bias=True, name='{}_theta_x'.format(name))(X)
    
    # mapping the gate tensor
    phi_g = Conv2D(channel, 1, use_bias=True, name='{}_phi_g'.format(name))(g)
    
    # ----- attention learning ----- #
    query = attention_func([theta_att, phi_g], name='{}_add'.format(name))
    
    # nonlinear activation
    f = activation_func(name='{}_activation'.format(name))(query)
    
    # linear transformation
    psi_f = Conv2D(1, 1, use_bias=True, name='{}_psi_f'.format(name))(f)
    # ------------------------------ #
    
    # sigmoid activation as attention coefficients
    coef_att = Activation('sigmoid', name='{}_sigmoid'.format(name))(psi_f)
    
    # multiplicative attention masking
    X_att = multiply([X, coef_att], name='{}_masking'.format(name))
    
    return X_att

def CONV_stack(X, channel, kernel_size=3, stack_num=2, 
               dilation_rate=1, activation='ReLU', 
               batch_norm=False, name='conv_stack'):
    '''
    Stacked convolutional layers:
    (Convolutional layer --> batch normalization --> Activation)*stack_num
    
    CONV_stack(X, channel, kernel_size=3, stack_num=2, dilation_rate=1, activation='ReLU', 
               batch_norm=False, name='conv_stack')
    
    
    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of stacked Conv2D-BN-Activation layers.
        dilation_rate: optional dilated convolution kernel.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor
        
    '''
    
    bias_flag = not batch_norm
    
    # stacking Convolutional layers
    for i in range(stack_num):
        
        activation_func = eval(activation)
        
        # linear convolution
        X = Conv2D(channel, kernel_size, padding='same', use_bias=bias_flag, 
                   dilation_rate=dilation_rate, name='{}_{}'.format(name, i))(X)
        
        # batch normalization
        if batch_norm:
            X = BatchNormalization(axis=3, name='{}_{}_bn'.format(name, i))(X)
        
        # activation
        activation_func = eval(activation)
        X = activation_func(name='{}_{}_activation'.format(name, i))(X)
        
    return X

def Res_CONV_stack(X, X_skip, channel, res_num, activation='ReLU', batch_norm=False, name='res_conv'):
    '''
    Stacked convolutional layers with residual path.
     
    Res_CONV_stack(X, X_skip, channel, res_num, activation='ReLU', batch_norm=False, name='res_conv')
     
    Input
    ----------
        X: input tensor.
        X_skip: the tensor that does go into the residual path 
                can be a copy of X (e.g., the identity block of ResNet).
        channel: number of convolution filters.
        res_num: number of convolutional layers within the residual path.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    '''  
    X = CONV_stack(X, channel, kernel_size=3, stack_num=res_num, dilation_rate=1, 
                   activation=activation, batch_norm=batch_norm, name=name)

    X = add([X_skip, X], name='{}_add'.format(name))
    
    activation_func = eval(activation)
    X = activation_func(name='{}_add_activation'.format(name))(X)
    
    return X

def Sep_CONV_stack(X, channel, kernel_size=3, stack_num=1, dilation_rate=1, activation='ReLU', batch_norm=False, name='sep_conv'):
    '''
    Depthwise separable convolution with (optional) dilated convolution kernel and batch normalization.
    
    Sep_CONV_stack(X, channel, kernel_size=3, stack_num=1, dilation_rate=1, activation='ReLU', batch_norm=False, name='sep_conv')
    
    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of stacked depthwise-pointwise layers.
        dilation_rate: optional dilated convolution kernel.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
    
    '''
    
    activation_func = eval(activation)
    bias_flag = not batch_norm
    
    for i in range(stack_num):
        X = DepthwiseConv2D(kernel_size, dilation_rate=dilation_rate, padding='same', 
                            use_bias=bias_flag, name='{}_{}_depthwise'.format(name, i))(X)
        
        if batch_norm:
            X = BatchNormalization(name='{}_{}_depthwise_BN'.format(name, i))(X)

        X = activation_func(name='{}_{}_depthwise_activation'.format(name, i))(X)

        X = Conv2D(channel, (1, 1), padding='same', use_bias=bias_flag, name='{}_{}_pointwise'.format(name, i))(X)
        
        if batch_norm:
            X = BatchNormalization(name='{}_{}_pointwise_BN'.format(name, i))(X)

        X = activation_func(name='{}_{}_pointwise_activation'.format(name, i))(X)
    
    return X

def ASPP_conv(X, channel, activation='ReLU', batch_norm=True, name='aspp'):
    '''
    Atrous Spatial Pyramid Pooling (ASPP).
    
    ASPP_conv(X, channel, activation='ReLU', batch_norm=True, name='aspp')
    
    ----------
    Wang, Y., Liang, B., Ding, M. and Li, J., 2019. Dense semantic labeling 
    with atrous spatial pyramid pooling and decoder for high-resolution remote 
    sensing imagery. Remote Sensing, 11(1), p.20.
    
    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    * dilation rates are fixed to `[6, 9, 12]`.
    '''
    
    activation_func = eval(activation)
    bias_flag = not batch_norm

    shape_before = X.get_shape().as_list()
    b4 = GlobalAveragePooling2D(name='{}_avepool_b4'.format(name))(X)
    
    b4 = expand_dims(expand_dims(b4, 1), 1, name='{}_expdim_b4'.format(name))
    
    b4 = Conv2D(channel, 1, padding='same', use_bias=bias_flag, name='{}_conv_b4'.format(name))(b4)
    
    if batch_norm:
        b4 = BatchNormalization(name='{}_conv_b4_BN'.format(name))(b4)
        
    b4 = activation_func(name='{}_conv_b4_activation'.format(name))(b4)
    
    # <----- tensorflow v1 resize.
    b4 = Lambda(lambda X: image.resize(X, shape_before[1:3], method='bilinear', align_corners=True), 
                name='{}_resize_b4'.format(name))(b4)
    
    b0 = Conv2D(channel, (1, 1), padding='same', use_bias=bias_flag, name='{}_conv_b0'.format(name))(X)

    if batch_norm:
        b0 = BatchNormalization(name='{}_conv_b0_BN'.format(name))(b0)
        
    b0 = activation_func(name='{}_conv_b0_activation'.format(name))(b0)
    
    # dilation rates are fixed to `[6, 9, 12]`.
    b_r6 = Sep_CONV_stack(X, channel, kernel_size=3, stack_num=1, activation='ReLU', 
                        dilation_rate=6, batch_norm=True, name='{}_sepconv_r6'.format(name))
    b_r9 = Sep_CONV_stack(X, channel, kernel_size=3, stack_num=1, activation='ReLU', 
                        dilation_rate=9, batch_norm=True, name='{}_sepconv_r9'.format(name))
    b_r12 = Sep_CONV_stack(X, channel, kernel_size=3, stack_num=1, activation='ReLU', 
                        dilation_rate=12, batch_norm=True, name='{}_sepconv_r12'.format(name))
    
    return concatenate([b4, b0, b_r6, b_r9, b_r12])

def CONV_output(X, n_labels, kernel_size=1, activation='Softmax', name='conv_output'):
    '''
    Convolutional layer with output activation.
    
    CONV_output(X, n_labels, kernel_size=1, activation='Softmax', name='conv_output')
    
    Input
    ----------
        X: input tensor.
        n_labels: number of classification label(s).
        kernel_size: size of 2-d convolution kernels. Default is 1-by-1.
        activation: one of the `tensorflow.keras.layers` or `swiss_army_keras.activations` interface or 'Sigmoid'.
                    Default option is 'Softmax'.
                    if None is received, then linear activation is applied.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    '''
    
    X = Conv2D(n_labels, kernel_size, padding='same', use_bias=True, name=name)(X)
    
    if activation:
        
        if activation == 'Sigmoid':
            X = Activation('sigmoid', name='{}_activation'.format(name))(X)
            
        else:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)
            
    return X




def dummy_preprocessing(x):
    '''Identity preprocessing, used when a model has no backbone-specific input normalization.
    '''
    return x


# ========================= #
# Lightweight attention/pooling layers (plan.md §9.1) - shared building
# blocks for the classifier heads in `_model_classifier.py`, wired in via
# each head's `pre_pool_attention=None|'gem'|'eca'|'coordatt'` argument
# rather than forking a near-duplicate classifier function per combination.


@tf.keras.utils.register_keras_serializable(package='swiss_army_keras')
class GeMPooling2D(tf.keras.layers.Layer):
    '''Generalized-Mean (GeM) pooling — `(mean(clip(x, eps)^p))^(1/p)` over
    the spatial dims, with a single learnable exponent `p` shared across all
    channels. `p=1` reduces to average pooling, `p -> inf` approaches max
    pooling — GeM lets training find the right point between those two
    (and beyond, since `p` isn't constrained to `[1, inf)`), which
    consistently outperforms committing to either one outright on
    fine-grained/attribute classification tasks, for one extra scalar
    parameter.

    A drop-in replacement for `GlobalAveragePooling2D`/`GlobalMaxPool2D`
    (input `(B, H, W, C)`, output `(B, C)`) — not a "pre-pool attention"
    gate the way `ECABlock`/`CoordAttLite` below are, since it replaces the
    pooling reduction itself rather than reweighting the feature map before
    an existing pooling op; grouped with them under one
    `pre_pool_attention` hook regardless, per plan.md §9.1's own framing.

    ----------
    Radenović, F., Tolias, G. and Chum, O., 2018. Fine-tuning CNN image
    retrieval with no human annotation. IEEE TPAMI.

    Input
    ----------
        p_init: initial value of the learnable exponent `p` (3.0, a common
                starting point in the literature — closer to max-pooling
                than average-pooling, since image-retrieval/fine-grained
                tasks tend to benefit from emphasizing salient regions).
        epsilon: clip floor before raising to the power `p`, avoiding
                 `0^p` (undefined for `p <= 0`) or exploding gradients from
                 a near-zero base raised to a large power.
        trainable_p: if `False`, `p` stays fixed at `p_init` instead of
                     being learned — a fixed-`p` GeM pool, sometimes used
                     when the extra parameter isn't wanted.
    '''

    def __init__(self, p_init=3.0, epsilon=1e-6, trainable_p=True, **kwargs):
        super().__init__(**kwargs)
        self.p_init = p_init
        self.epsilon = epsilon
        self.trainable_p = trainable_p

    def build(self, input_shape):
        self.p = self.add_weight(
            name='p', shape=(), dtype=self.dtype,
            initializer=tf.keras.initializers.Constant(self.p_init),
            trainable=self.trainable_p,
        )
        super().build(input_shape)

    def call(self, x):
        x = tf.clip_by_value(x, self.epsilon, tf.float32.max)
        x = tf.pow(x, self.p)
        x = tf.reduce_mean(x, axis=[1, 2])
        return tf.pow(x, 1.0 / self.p)

    def get_config(self):
        config = super().get_config()
        config.update({'p_init': self.p_init, 'epsilon': self.epsilon, 'trainable_p': self.trainable_p})
        return config


@tf.keras.utils.register_keras_serializable(package='swiss_army_keras')
class ECABlock(tf.keras.layers.Layer):
    '''Efficient Channel Attention (ECA) — a single 1-D conv (no
    dimensionality reduction, unlike SE's two-FC bottleneck) over the
    globally-pooled per-channel descriptor, sigmoid-gated back onto the
    input. Orders of magnitude fewer parameters than SE for a comparable
    accuracy gain, since it never projects the channel descriptor down to
    a bottleneck and back.

    ----------
    Wang, Q., Wu, B., Zhu, P., Li, P., Zuo, W. and Hu, Q., 2020. ECA-Net:
    Efficient channel attention for deep convolutional neural networks.
    In CVPR.

    Input
    ----------
        kernel_size: the 1-D conv's kernel size (the *only* free parameter
                     ECA introduces). `None` (default) picks it adaptively
                     from the input's channel count via the paper's own
                     formula (`k = |log2(C)/gamma + b/gamma|`, rounded to
                     the nearest odd integer, floored at 3) — a fixed
                     kernel size can also be passed directly.
        gamma, b: coefficients of the adaptive kernel-size formula above;
                  only used when `kernel_size=None`. `2`/`1`, the paper's
                  own defaults.
    '''

    def __init__(self, kernel_size=None, gamma=2, b=1, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.gamma = gamma
        self.b = b

    def build(self, input_shape):
        channels = int(input_shape[-1])
        if self.kernel_size is not None:
            k = self.kernel_size
        else:
            t = int(abs((math.log(channels, 2) / self.gamma) + (self.b / self.gamma)))
            k = t if t % 2 else t + 1
            k = max(k, 3)
        self._resolved_kernel_size = k
        self.conv = tf.keras.layers.Conv1D(1, k, padding='same', use_bias=False, name='eca_conv1d')
        super().build(input_shape)

    def call(self, x):
        y = tf.reduce_mean(x, axis=[1, 2])          # (B, C)
        y = tf.expand_dims(y, axis=-1)               # (B, C, 1)
        y = self.conv(y)                             # (B, C, 1)
        y = tf.squeeze(y, axis=-1)                    # (B, C)
        y = tf.nn.sigmoid(y)
        y = y[:, tf.newaxis, tf.newaxis, :]           # (B, 1, 1, C)
        return x * y

    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size, 'gamma': self.gamma, 'b': self.b})
        return config


@tf.keras.utils.register_keras_serializable(package='swiss_army_keras')
class CoordAttLite(tf.keras.layers.Layer):
    '''Coordinate Attention (lite variant) — unlike GAP-based channel
    attention (SE/ECA above), which pools away *all* spatial information
    before computing its gate, CoordAtt keeps positional information by
    pooling along the height and width axes *separately* first (giving one
    per-row and one per-column descriptor), fusing them through a shared
    1x1 conv, then splitting back into an independent per-row and
    per-column attention map. Matters whenever *where* in the frame carries
    signal, not just *what's* in it — plain channel attention discards that
    signal entirely by construction.

    "Lite" here: the shared fusion nonlinearity is a plain `relu` rather
    than the original paper's own `h-swish` (no standard Keras
    layer for it, and this keeps the block dependency-free) — a documented
    simplification, not a different algorithm.

    ----------
    Hou, Q., Zhou, D. and Feng, J., 2021. Coordinate attention for
    efficient mobile network design. In CVPR.

    Input
    ----------
        reduction: channel-reduction ratio for the shared fusion conv's
                   bottleneck width (`max(8, channels // reduction)`) — 32,
                   the paper's own default.
    '''

    def __init__(self, reduction=32, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        self.height = int(input_shape[1])
        self.width = int(input_shape[2])
        channels = int(input_shape[-1])
        mid_channels = max(8, channels // self.reduction)

        self.conv_reduce = tf.keras.layers.Conv2D(mid_channels, 1, use_bias=False, name='coordatt_reduce')
        self.bn = tf.keras.layers.BatchNormalization(name='coordatt_bn')
        self.conv_h = tf.keras.layers.Conv2D(channels, 1, use_bias=True, name='coordatt_h')
        self.conv_w = tf.keras.layers.Conv2D(channels, 1, use_bias=True, name='coordatt_w')
        super().build(input_shape)

    def call(self, x):
        pool_h = tf.reduce_mean(x, axis=2, keepdims=True)              # (B, H, 1, C)
        pool_w = tf.reduce_mean(x, axis=1, keepdims=True)               # (B, 1, W, C)
        pool_w = tf.transpose(pool_w, [0, 2, 1, 3])                     # (B, W, 1, C)

        y = tf.concat([pool_h, pool_w], axis=1)                         # (B, H+W, 1, C)
        y = self.conv_reduce(y)
        y = self.bn(y)
        y = tf.nn.relu(y)

        y_h, y_w = tf.split(y, [self.height, self.width], axis=1)
        y_w = tf.transpose(y_w, [0, 2, 1, 3])                            # (B, 1, W, C)

        attn_h = tf.nn.sigmoid(self.conv_h(y_h))                        # (B, H, 1, C)
        attn_w = tf.nn.sigmoid(self.conv_w(y_w))                        # (B, 1, W, C)

        return x * attn_h * attn_w

    def get_config(self):
        config = super().get_config()
        config.update({'reduction': self.reduction})
        return config


def apply_pre_pool_attention(x, pre_pool_attention=None, name='pre_pool_attention'):
    '''Dispatch helper for the `pre_pool_attention=None|'eca'|'coordatt'`
    hook (`_model_classifier.py`'s classifier heads) — `'gem'` is handled
    separately by the caller (it replaces the pooling op itself, see
    `GeMPooling2D`'s docstring for why), so it's a no-op here.
    '''
    if pre_pool_attention == 'eca':
        return ECABlock(name=f'{name}_eca')(x)
    if pre_pool_attention == 'coordatt':
        return CoordAttLite(name=f'{name}_coordatt')(x)
    return x


def iqa_regression_head(base_model_tap, size=64, pooling='gem', name='iqa'):
    '''A small, cheap regression tail for a shared-backbone Image Quality
    Assessment (IQA) side-head (plan.md §13.7, Phase 11) — trained
    simultaneously with a task head on the same backbone, against a
    teacher-distilled pseudo-label (`swiss_army_keras.iqa_utils.
    add_teacher_iqa_labels`), so a single exported graph carries both the
    main task output and a cheap per-inference quality/confidence signal.

    Deliberately much smaller than a task head's own trunk (`size=64`
    default vs. `classifier()`'s own `size=1024`) — this branch only
    regresses one scalar, not `n_classes`/`n_labels`-many outputs.

    Input
    ----------
        base_model_tap: the backbone feature tensor to tap - an
                        *intermediate* tensor (e.g. `classifier()`'s own
                        `base_model`, before its own pooling/Dense tail
                        runs), not a model's final output. Callers thread
                        this in themselves; this function has no opinion
                        on which backbone tap to use.
        size: the single hidden `Dense` layer's width (`64`).
        pooling: `'gem'` (default, `GeMPooling2D`) or `'avg'`
                 (`GlobalAveragePooling2D`).
        name: prefix of the created Keras layers.

    Output
    ----------
        A `(batch, 1)` sigmoid-bounded `[0, 1]` quality score tensor - not
        a NIMA-style score distribution (the FP/FN-filtering use case this
        was built for only needs a threshold-able scalar).
    '''
    if pooling == 'gem':
        pool = GeMPooling2D(name=f'{name}_gem')(base_model_tap)
    else:
        pool = qat_annotate_layer(GlobalAveragePooling2D())(base_model_tap)

    x = qat_annotate_layer(tf.keras.layers.Dense(size, activation='relu', name=f'{name}_dense'))(pool)
    return qat_annotate_layer(tf.keras.layers.Dense(1, activation='sigmoid', name=f'{name}_score'))(x)


# ========================= #
# Edge-deployment extras (plan.md §13.1/§13.2, Phase 10) - in-graph uint8
# input handling and a training-free OOD/low-confidence signal (Energy
# Score), both built as post-hoc wrappers around an already-built model
# rather than parameters threaded through every model-builder function.
# This works because every builder in this repo (`classifier`, `lr_aspp`,
# `fcos_lite_detection`, etc.) already takes an externally-built
# `input_tensor` rather than constructing its own `Input(...)` - the same
# "outer model calls an already-`quantize_apply`'d inner model, adds
# float32-only processing around it, no re-quantization needed" pattern
# `_model_detection.py`'s `fcos_lite_detection`/`gfl_lite_detection`
# already use at the *output* side (`decode_boxes`/NMS after the base
# model), applied here at the *input* side (`uint8_input_wrapper`) and
# reused unmodified at the output side too (`add_energy_score_output`).


def uint8_input_wrapper(model, channels_first=False, name=None):
    '''Wraps an already-built model (any task this repo produces -
    classifier, segmentation decoder, detector) with a new raw `uint8`
    `(H, W, 3)` input, moving dtype cast, normalization, and (optionally) a
    channels-last -> channels-first layout transpose entirely into the
    graph - motivated by a concrete Axis ACAP constraint: when a TFLite
    model's input dtype is set to `uint8`, Axis's own docs say the input's
    quantization currently has to be handled *manually* by the calling
    application code; baking it into the model removes that step entirely.

    Reuses `model.preprocessing` (the same per-backbone mean/std function
    already attached by `backbone_zoo`, e.g. `mobileone_preprocess_input`)
    rather than taking new `mean`/`std` parameters - every
    `*_preprocess_input` function in this repo is plain elementwise
    arithmetic (`x/255.0`, `(x-mean)/std`), which already works unchanged
    on a symbolic tensor exactly as well as on a numpy array, so no new
    normalization logic is needed here; the numeric contract stays defined
    in exactly one place (`BackboneSpec`), not duplicated at this call site.

    **Stays outside `quantize_apply` entirely - the cast/normalize/
    transpose are plain functional ops, never QAT-annotated**, the same
    "small math stays un-annotated" precedent `_model_detection.py`'s own
    `decode_boxes` already establishes.
    Concretely: since this preprocessing never gets quantized in the first
    place, the ARTPEC-8 DLPU's per-tensor-only quantization constraint
    (Axis ACAP) doesn't apply to it at all - no need to fold the per-channel
    mean/std into the backbone's first conv to work around it.

    Needs no changes to, and does **not** re-run `quantize_apply` on, the
    wrapped model - it's called as a single sub-model layer on the
    preprocessed tensor, exactly like `fcos_lite_detection`/
    `gfl_lite_detection` already call their own already-quantized base
    model on a tensor built outside its own graph.

    Input
    ----------
        model: an already-built `tf.keras.Model` with a `.preprocessing`
               attribute (every builder in this repo sets one) and a
               static `input_shape` - `(H, W, 3)` if `channels_first=False`
               (every backbone in this repo today), or `(3, H, W)` if
               `channels_first=True` (`model` itself must already be built
               that way for this to be usable - this wrapper transposes
               *into* that layout, it doesn't change what `model` itself
               expects).
        channels_first: if `True`, transposes the preprocessed float32
                        tensor from `(B, H, W, 3)` to `(B, 3, H, W)` before
                        calling `model` - for a future encoder built to
                        expect that layout (`data_format='channels_first'`).
                        No backbone in this repo needs this today (every
                        one is channels-last) - a forward-looking
                        capability, not a fix for an existing one. The
                        wrapper's own outward-facing input is always
                        `uint8 (H, W, 3)` regardless of this flag -
                        matching this project's own stated contract
                        ("images should be provided as WxHx3") even when
                        the wrapped model itself needs a different layout
                        internally.
        name: prefix for the new input tensor and the returned model.

    Output
    ----------
        A new `tf.keras.Model`, `uint8 (H, W, 3) -> model`'s original
        outputs unchanged - its own `.preprocessing` is set to
        `dummy_preprocessing`, since normalization is now baked into the
        graph and a caller must **not** reapply it externally. If `model`
        itself was built with a *static* batch dimension (e.g.
        `fcos_lite_detection`'s own opt-in `Input(batch_shape=(N, H, W,
        3))` path, `_model_detection.py`'s own `_nms_dispatch`), the
        wrapped model's own outward input keeps that same static batch
        size - **a real, confirmed bug fix, not new behavior**: this used
        to always build a *dynamic*-batch outward `Input` regardless of
        what `model` itself required, which TFLite's converter silently
        defaults to `batch=1` for - producing a `.tflite` whose outward
        interface claimed batch=1 while the model wrapped inside it
        actually required a larger fixed batch, a shape mismatch only
        surfacing at inference time (`ValueError: Cannot set tensor:
        Dimension mismatch`), not at conversion time. Found while
        verifying `fcos_lite_detection`'s own new static-batch NMS path.
    '''
    batch_size = model.input_shape[0]
    if channels_first:
        channels, height, width = model.input_shape[1:]
    else:
        height, width, channels = model.input_shape[1:]

    if batch_size is not None:
        uint8_input = Input(
            batch_shape=(batch_size, height, width, channels), dtype='uint8', name=f'{name}_uint8_input')
    else:
        uint8_input = Input(shape=(height, width, channels), dtype='uint8', name=f'{name}_uint8_input')

    x = tf.cast(uint8_input, tf.float32)
    # Preprocessing (per-channel mean/std) broadcasts against the *last*
    # axis, so it must run on the still-channels-last tensor - transposing
    # first would apply the wrong axis's mean/std to each channel.
    x = model.preprocessing(x)
    if channels_first:
        x = tf.transpose(x, [0, 3, 1, 2])

    outputs = model(x)

    wrapped = Model(uint8_input, outputs, name=name or f'{model.name}_uint8')
    wrapped.preprocessing = dummy_preprocessing
    return wrapped


def energy_score(logits, temperature=1.0, class_axis=-1):
    '''Energy Score (Liu, Wang, Owens, Li, "Energy-based Out-of-distribution
    Detection", NeurIPS 2020, arXiv:2010.03759) -
    `E(x) = -T * log(sum_i(exp(f_i(x) / T)))` over raw per-class logits
    (not post-softmax/sigmoid probabilities, which normalize away the
    absolute scale this formula needs). Low energy means the model is
    confidently familiar with an input's statistics; high energy means it's
    guessing - a training-free, zero-new-parameter OOD/low-confidence
    signal computed directly from a model's own already-computed logits.

    `tf.reduce_logsumexp` has no trainable state, so it needs no custom
    `Layer`/`QuantizeConfig` - safe as a plain functional call, the same
    class of op `_model_detection.py`'s `decode_boxes`/`gfl_integral_decode`
    already use outside `quantize_apply`'s boundary.

    Input
    ----------
        logits: raw (pre-activation) per-class logit tensor, any shape with
                the class dimension at `class_axis` - a single `(B,
                n_classes)` vector for a classifier, or a dense `(B, H, W,
                n_classes)` map for a segmentation decoder's own per-pixel
                logits.
        temperature: `T` in the formula above (`1.0`, the paper's own
                     common default).
        class_axis: which axis holds the per-class logits (`-1`, matching
                    every logit tensor this repo already produces).

    Output
    ----------
        A tensor one rank lower than `logits` (the class axis reduced away)
        - lower values indicate higher confidence/in-distribution.
    '''
    return -temperature * tf.reduce_logsumexp(logits / temperature, axis=class_axis)


def add_energy_score_output(model, logit_output_indices=(0,), temperature=1.0, class_axis=-1, name=None):
    '''Wraps an already-built model, appending one `energy_score` output
    per index in `logit_output_indices` (each computed from that existing
    output tensor, assumed to already be raw logits) to its output list -
    zero new trainable weights, no change to any existing output.

    Works **unmodified, today**, on: any segmentation decoder built by
    this repo (`logit_output_indices=(0,)` - its one output already is the
    raw per-pixel logit map, e.g. `lr_aspp`/`pp_liteseg`'s final `Conv2D`
    has no fused activation); `fcos_lite_detection_base`/
    `gfl_lite_detection_base`'s own outputs (e.g.
    `logit_output_indices=(0, 3, 6)` for the 3 per-level classification-
    logit tensors in their flat output list) - a dense, pre-NMS, per-level
    energy signal. **Does not solve gathering an energy score onto
    post-NMS survivors** (`tf.image.combined_non_max_suppression` returns
    no selection indices to gather with) - this stays a pre-NMS,
    base-model-level signal only, a deliberate scope boundary, not an
    oversight. Does **not** work on `classifier()`'s default output (its
    softmax is fused into one `Dense` layer call, so no logits tensor
    exists anywhere in the graph) - pass `return_logits=True` to
    `classifier()` first, then point `logit_output_indices` at its second
    (`index 1`) output.

    Input
    ----------
        model: an already-built `tf.keras.Model`.
        logit_output_indices: indices into `model.outputs` treated as raw
                              logit tensors, each energy-scored independently.
        temperature, class_axis: passed through to `energy_score`.
        name: prefix for the returned model.

    Output
    ----------
        A new `tf.keras.Model`, `model`'s own input(s) -> `model`'s
        original outputs, plus one `energy_score` tensor per entry in
        `logit_output_indices`, appended in order. `.preprocessing` is
        carried over unchanged from `model`.
    '''
    outputs = list(model.outputs)
    energy_outputs = [
        energy_score(outputs[i], temperature=temperature, class_axis=class_axis)
        for i in logit_output_indices
    ]

    wrapped = Model(model.input, outputs + energy_outputs, name=name or f'{model.name}_energy')
    wrapped.preprocessing = model.preprocessing
    return wrapped


_EMBEDDING_HEAD_LAYERS = ('Conv2D', 'Dense', 'DepthwiseConv2D', 'SeparableConv2D', 'Conv2DTranspose')
_EMBEDDING_PASS_THROUGH = ('Dropout', 'UpSampling2D', 'Resizing')


def _unwrapped(layer):
    '''The layer inside a QAT wrapper, or the layer itself.'''
    return layer.layer if type(layer).__name__.startswith('QuantizeWrapper') else layer


def _embedding_tensor(model, output_index):
    '''Find the tensor that feeds the last layer with weights before an output.'''
    def producer(tensor):
        layer, node_index, _ = tensor._keras_history
        return layer, tf.nest.flatten(layer._inbound_nodes[node_index].keras_inputs)

    hint = ' Pass `layer=` with the name of the layer whose output is the embedding.'
    tensor = model.outputs[output_index]
    for _ in range(16):
        layer, inputs = producer(tensor)
        kind = type(_unwrapped(layer)).__name__
        if kind in _EMBEDDING_HEAD_LAYERS:
            tensor = inputs[0]
            break
        if len(inputs) != 1 or kind in ('InputLayer', 'Functional', 'Model'):
            raise ValueError(
                "add_embedding_output: cannot find the head of output {} automatically (stopped at layer '{}').{}".format(
                    output_index, layer.name, hint))
        tensor = inputs[0]
    else:
        raise ValueError('add_embedding_output: no head layer found for output {}.{}'.format(output_index, hint))

    for _ in range(16):
        layer, inputs = producer(tensor)
        if type(_unwrapped(layer)).__name__ not in _EMBEDDING_PASS_THROUGH or len(inputs) != 1:
            break
        tensor = inputs[0]
    return tensor


def add_embedding_output(model, layer=None, output_index=0, name=None):
    '''Wraps an already-built model, appending the feature that feeds one
    of its heads as an extra output: the embedding that
    `swiss_army_keras.ood_utils` fits Mahalanobis statistics on.

    No layer is added and no weight changes, so this works on a trained
    model (the new model shares its layers) and the extra output is carried
    through TFLite export like any other.

    With `layer=None` the embedding is found from the graph: walking back
    from output `output_index`, it is the input of the last layer with
    weights (the head's final `Dense` or convolution), taken before any
    dropout or resize that precedes that layer. This gives:

    * `classifier`: the output of the hidden `Dense` layer, `(batch, size)`.
    * `lr_aspp`: the concatenated decoder feature at the resolution of the
      shallow skip, `(batch, H/4, W/4, 176)` with the defaults. It is taken
      before the final resize on purpose: the full-resolution tensor holds
      the same information in sixteen times the memory.
    * other single-path decoders and `multi_task` heads: the same rule.

    Input
    ----------
        model: an already-built functional `tf.keras.Model`.
        layer: `None` for the automatic choice, or the name of the layer
               whose output is the embedding (the name is also tried with
               the `quant_` prefix that quantization-aware training adds),
               or a Keras tensor of the model's graph.
        output_index: which output's head to look behind when `layer=None`.
        name: name of the returned model.

    Output
    ----------
        A new `tf.keras.Model` with the outputs of `model` followed by the
        embedding. `embedding_output_index` holds the position of the
        embedding in the output list; `.preprocessing` and `.head_names`
        are carried over.
    '''
    if layer is None:
        embedding = _embedding_tensor(model, output_index)
    elif isinstance(layer, str):
        names = {l.name for l in model.layers}
        found = next((n for n in (layer, 'quant_' + layer) if n in names), None)
        if found is None:
            raise ValueError("add_embedding_output: the model has no layer named '{}'.".format(layer))
        embedding = model.get_layer(found).output
    else:
        embedding = layer

    outputs = list(model.outputs)
    wrapped = Model(model.inputs, outputs + [embedding], name=name or f'{model.name}_embedding')
    for attribute in ('preprocessing', 'head_names'):
        if hasattr(model, attribute):
            setattr(wrapped, attribute, getattr(model, attribute))
    wrapped.embedding_output_index = len(outputs)
    return wrapped
