'''Activation layers not shipped with Keras 2: `GELU` and `Snake`.
'''
import tensorflow as tf
from tensorflow import math
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
import tensorflow.keras.backend as K


def gelu_(X):
    # Exact (erf-based) GELU, not the tanh approximation this used to be -
    # PyTorch's default `nn.GELU()` (`approximate='none'`) is exact, and
    # every backbone in this repo that uses GELU (RepViT, FastViT,
    # MobileNetV4) is architecturally modeled on a PyTorch/timm reference
    # that uses that default. The mismatch was invisible for `weights=None`
    # (random-init) training, but measurably hurt pretrained-weight-porting
    # fidelity - confirmed while porting RepViT's real ImageNet weights
    # (plan.md §12.1): swapping in the exact formula cut a ported model's
    # numerical divergence from the PyTorch reference by ~6x on the
    # deepest RepViT variant tested (RepViT-M1.5, max abs diff 0.28 -> 0.04),
    # with the gap growing with network depth under the old approximation
    # (more GELU calls compounding the same per-call error) - a smoking gun
    # for exactly this being the cause, not a deeper conversion bug.
    return tf.nn.gelu(X, approximate=False)

def snake_(X, beta):

    return X + (1/beta)*math.square(math.sin(beta*X))


@register_keras_serializable(package='swiss_army_keras')
class GELU(Layer):
    '''
    Gaussian Error Linear Unit (GELU), an alternative of ReLU
    
    Y = GELU()(X)
    
    ----------
    Hendrycks, D. and Gimpel, K., 2016. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415.
    
    Usage: use it as a tf.keras.Layer
    
    
    '''
    def __init__(self, trainable=False, **kwargs):
        super(GELU, self).__init__(**kwargs)
        self.supports_masking = True
        self.trainable = trainable

    def build(self, input_shape):
        super(GELU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return gelu_(inputs)

    def get_config(self):
        config = {'trainable': self.trainable}
        base_config = super(GELU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def compute_output_shape(self, input_shape):
        return input_shape

    
@register_keras_serializable(package='swiss_army_keras')
class Snake(Layer):
    '''
    Snake activation function $X + (1/b)*sin^2(b*X)$. Proposed to learn periodic targets.
    
    Y = Snake(beta=0.5, trainable=False)(X)
    
    ----------
    Ziyin, L., Hartwig, T. and Ueda, M., 2020. Neural networks fail to learn periodic functions 
    and how to fix it. arXiv preprint arXiv:2006.08195.
    
    '''
    def __init__(self, beta=0.5, trainable=False, **kwargs):
        super(Snake, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable

    def build(self, input_shape):
        self.beta_factor = K.variable(self.beta, dtype=K.floatx(), name='beta_factor')
        if self.trainable:
            self._trainable_weights.append(self.beta_factor)

        super(Snake, self).build(input_shape)

    def call(self, inputs, mask=None):
        return snake_(inputs, self.beta_factor)

    def get_config(self):
        config = {'beta': self.get_weights()[0] if self.trainable else self.beta, 'trainable': self.trainable}
        base_config = super(Snake, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
    