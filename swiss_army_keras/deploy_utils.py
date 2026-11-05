'''
Deployment helpers: target profiles, structural reparameterization of full
models, TFLite conversion, and a check of the exported operators against
what a target accelerates.

Typical use (also what `quantization_utils.Quantizer` does)::

    from swiss_army_keras import deploy_utils

    plain = deploy_utils.strip_qat(deploy_utils.reparameterize_model(model))
    plain = deploy_utils.drop_output_activations(plain)   # optional: no final sigmoid/softmax
    deploy_utils.equalize_cross_layer(plain)          # for per-tensor targets
    tflite = deploy_utils.convert_to_tflite(plain, representative_dataset, profile='artpec8')
    report = deploy_utils.lint_tflite(tflite, profile='artpec8')
    print(report)
'''

import collections
import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, FrozenSet, Optional, Tuple

import numpy as np
import tensorflow as tf

# ========================= #
# Target profiles

# Operator lists transcribed from the Axis developer documentation
# ("Supported Ops" page of each system-on-chip), read on 2026-11-05 and mapped
# to TFLite builtin operator names. They were extracted from the web pages by
# an automated reader and have NOT been confirmed on a device: treat a "not
# listed" finding as a prompt to test with `larod-client`, not as proof.
# QUANTIZE is always accepted (it is the uint8 <-> int8 requantization the
# converter inserts at the model boundary).
_AXIS_DOC_DATE = '2026-11-05'

_ARTPEC8_OPS = frozenset({
    'FULLY_CONNECTED', 'CONV_2D', 'DEPTHWISE_CONV_2D', 'TRANSPOSE_CONV',
    'MAX_POOL_2D', 'AVERAGE_POOL_2D', 'CONCATENATION', 'RESHAPE', 'EXPAND_DIMS',
    'SQUEEZE', 'SLICE', 'STRIDED_SLICE', 'GATHER', 'GATHER_ND', 'ONE_HOT',
    'SPLIT', 'TRANSPOSE', 'RESIZE_NEAREST_NEIGHBOR', 'RESIZE_BILINEAR', 'CAST',
    'ADD', 'ADD_N', 'MUL', 'DIV', 'FLOOR_DIV', 'POW', 'SQUARE', 'SIN', 'TANH',
    'ABS', 'NEG', 'SQRT', 'FLOOR', 'MINIMUM', 'MAXIMUM', 'EQUAL', 'NOT_EQUAL',
    'LESS', 'LESS_EQUAL', 'GREATER', 'GREATER_EQUAL', 'LOGICAL_NOT',
    'LOGICAL_AND', 'LOGICAL_OR', 'RELU', 'RELU6', 'SOFTMAX', 'GELU', 'ELU',
    'HARD_SWISH', 'REDUCE_MIN', 'REDUCE_MAX', 'REDUCE_ANY', 'REDUCE_PROD',
    'ARG_MIN', 'ARG_MAX', 'QUANTIZE',
})

_ARTPEC9_OPS = frozenset({
    'ADD', 'AVERAGE_POOL_2D', 'CONCATENATION', 'CONV_2D', 'DEPTH_TO_SPACE',
    'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED', 'LEAKY_RELU', 'MAX_POOL_2D', 'MEAN',
    'MUL', 'PAD', 'RELU', 'RELU6', 'RESHAPE', 'RESIZE_BILINEAR',
    'RESIZE_NEAREST_NEIGHBOR', 'LOGISTIC', 'SPLIT', 'TANH', 'TRANSPOSE',
    'TRANSPOSE_CONV', 'QUANTIZE',
})


@dataclass(frozen=True)
class DeploymentProfile:
    '''How a model must be exported for one deployment target.

    Input
    ----------
        name: profile name (key of `PROFILES`).
        description: one line describing the target.
        per_channel: True for per-channel weight quantization (the TFLite
                     default), False for one scale per weight tensor.
        io_dtype: 'uint8', 'int8' or 'float32': type of the model's input
                  and output tensors.
        integer_only: True when the target cannot run floating-point
                      operators at all (an accelerator, as opposed to a
                      CPU). `Quantizer` strips QAT wrappers before
                      converting for such targets, see `strip_qat`.
        channel_multiple: if set, convolution output channel counts that are
                          not a multiple of it are reported (the accelerator
                          wastes compute on them).
        supported_ops: TFLite builtin operator names the accelerator runs,
                       or None when there is no restriction to check.
        ops_source: where `supported_ops` comes from, for the report.
        constraints: names of extra per-operator checks to run (keys of
                     `_CONSTRAINT_CHECKS`).
        notes: free text shown in the lint report.
    '''
    name: str
    description: str
    per_channel: bool = True
    io_dtype: str = 'uint8'
    integer_only: bool = False
    channel_multiple: Optional[int] = None
    supported_ops: Optional[FrozenSet[str]] = None
    ops_source: str = ''
    constraints: Tuple[str, ...] = ()
    notes: str = ''


PROFILES = {
    'default': DeploymentProfile(
        name='default',
        description='Generic full-integer TFLite: per-channel weights, uint8 input/output.',
    ),
    'litert-cpu': DeploymentProfile(
        name='litert-cpu',
        description='aarch64 / x86 CPU with LiteRT (XNNPACK): per-channel weights, uint8 input/output.',
    ),
    'artpec7': DeploymentProfile(
        name='artpec7',
        description='Axis ARTPEC-7 (Edge TPU): full-integer TFLite, to be compiled with the Edge TPU compiler.',
        integer_only=True,
        notes='Run `edgetpu_compiler` on the exported file; it reports which operators stay on the CPU.',
    ),
    'artpec8': DeploymentProfile(
        name='artpec8',
        description='Axis ARTPEC-8 DLPU: per-tensor weights, uint8 input/output.',
        per_channel=False,
        integer_only=True,
        channel_multiple=6,
        supported_ops=_ARTPEC8_OPS,
        ops_source='Axis developer documentation, ARTPEC-8 "Supported Ops", read {} (unverified on device)'.format(_AXIS_DOC_DATE),
        notes='Axis recommends regular Conv2D over DepthwiseConv2D, 3x3 kernels, stride 2, ReLU over leaky ReLU.',
    ),
    'artpec9': DeploymentProfile(
        name='artpec9',
        description='Axis ARTPEC-9 DLPU: uint8 input/output; per-channel weights for convolutions.',
        per_channel=True,
        integer_only=True,
        channel_multiple=16,
        supported_ops=_ARTPEC9_OPS,
        ops_source='Axis developer documentation, ARTPEC-9 "Supported Ops", read {} (unverified on device)'.format(_AXIS_DOC_DATE),
        constraints=('mul_by_constant_only', 'global_mean_7_or_8', 'resize_2x'),
        notes='Axis recommends ReLU6, no explicit padding, batch size 1. Per-channel quantization is documented '
              'as supported "only for weights and bias tensors of a limited set of operations"; pass '
              '`per_channel=False` if a per-channel model misbehaves on the device.',
    ),
}


def get_profile(profile):
    '''Resolve a profile name (or pass a `DeploymentProfile` through).'''
    if isinstance(profile, DeploymentProfile):
        return profile
    if profile not in PROFILES:
        raise ValueError("Unknown deployment profile '{}'. Available: {}".format(profile, sorted(PROFILES)))
    return PROFILES[profile]


# ========================= #
# Structural reparameterization of full models

def _reparam_families():
    # Imported lazily: the backbone modules import `layer_utils`, which
    # performs the Keras-version check, and this module should stay cheap to
    # import for users who only need the profiles.
    from swiss_army_keras._backbone_mobileone import MOBILEONE_CONFIGS, reparameterize_mobileone
    from swiss_army_keras._backbone_repvit import REPVIT_CONFIGS, reparameterize_repvit
    from swiss_army_keras._backbone_fastvit import FASTVIT_CONFIGS, reparameterize_fastvit
    return (
        (MOBILEONE_CONFIGS, '_mobileone_config', reparameterize_mobileone),
        (REPVIT_CONFIGS, '_repvit_config', reparameterize_repvit),
        (FASTVIT_CONFIGS, '_fastvit_config', reparameterize_fastvit),
    )


def _fuse_standalone(model):
    '''Fuse a model built directly by a backbone constructor, or return None.'''
    for _, attr, fuse_fn in _reparam_families():
        if hasattr(model, attr):
            if getattr(model, '_sak_reparameterized', False):
                return None
            fused = fuse_fn(model)
            fused._sak_reparameterized = True
            return fused
    return None


def _fuse_nested_backbone(backbone):
    '''Fuse a `<registry name>_backbone` sub-model created by `backbone_zoo`.

    Returns a new sub-model with the same name, input and outputs (the same
    feature-pyramid taps), or None if `backbone` is not a reparameterizable
    backbone or is already fused.
    '''
    from swiss_army_keras._backbone_zoo import BACKBONE_REGISTRY

    name = backbone.name
    if not name.endswith('_backbone') or getattr(backbone, '_sak_reparameterized', False):
        return None
    key = name[:-len('_backbone')]
    for configs, attr, fuse_fn in _reparam_families():
        if key not in configs:
            continue
        # The family functions only need the layers (looked up by name), the
        # input shape and the family config; the sub-model `backbone_zoo`
        # builds has the first two, the registry provides the third.
        setattr(backbone, attr, dict(configs[key]))
        deploy = fuse_fn(backbone)
        tap_names = BACKBONE_REGISTRY[key].layers[:len(backbone.outputs)]
        taps = [deploy.get_layer(tap).output for tap in tap_names]
        for old, new in zip(backbone.outputs, taps):
            if tuple(old.shape) != tuple(new.shape):
                raise RuntimeError(
                    "Reparameterized backbone '{}' changed a feature shape: {} -> {}".format(key, old.shape, new.shape))
        fused = tf.keras.models.Model(inputs=[deploy.input, ], outputs=taps, name=name)
        fused.trainable = backbone.trainable
        fused._sak_reparameterized = True
        if hasattr(backbone, 'preprocessing'):
            fused.preprocessing = backbone.preprocessing
        return fused
    return None


def _reparameterize(model):
    replacements = {}
    fused_count = 0
    for layer in model.layers:
        if not isinstance(layer, tf.keras.Model):
            continue
        fused = _fuse_nested_backbone(layer)
        if fused is not None:
            replacements[layer.name] = fused
            fused_count += 1
            continue
        inner, inner_count = _reparameterize(layer)
        if inner_count:
            replacements[layer.name] = inner
            fused_count += inner_count
    if not replacements:
        return model, 0
    # Every other layer instance is reused as is (weights are shared with the
    # original model); only the backbone sub-models are swapped.
    new_model = tf.keras.models.clone_model(
        model, clone_function=lambda layer: replacements.get(layer.name, layer))
    if hasattr(model, 'preprocessing'):
        new_model.preprocessing = model.preprocessing
    return new_model, fused_count


def reparameterize_model(model):
    '''
    Fuse the training-time branches of every reparameterizable backbone
    (MobileOne, RepViT, FastViT) found in `model`, for deployment.

    Works on a backbone built directly from its constructor
    (`MobileOne_S0(...)`) and on any full model of this library that embeds
    such a backbone through `backbone_zoo` (decoders, classifiers,
    detectors), including one wrapped by `uint8_input_wrapper`. The result
    computes the same function (within floating-point tolerance) with far
    fewer layers.

    Input
    ----------
        model: a `tf.keras.Model`.

    Output
    ----------
        a new model if something was fused, otherwise `model` itself (models
        with other backbones pass through unchanged). Layers other than the
        fused backbone are shared with the original model, not copied, so
        keep training on the original and treat the result as an export
        artifact. `preprocessing` is carried over.
    '''
    fused = _fuse_standalone(model)
    if fused is not None:
        return fused
    new_model, _ = _reparameterize(model)
    return new_model


# ========================= #
# Removing quantization-aware-training wrappers

def _qat_classes():
    from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
    from tensorflow_model_optimization.python.core.quantization.keras.quantize_layer import QuantizeLayer
    from tensorflow_model_optimization.python.core.quantization.keras.quantize_aware_activation import QuantizeAwareActivation
    return QuantizeWrapper, QuantizeLayer, QuantizeAwareActivation


def has_qat_layers(model):
    '''True if `model` (or a nested sub-model) contains `tfmot` QAT wrappers.'''
    QuantizeWrapper, QuantizeLayer, _ = _qat_classes()
    for layer in model.layers:
        if isinstance(layer, (QuantizeWrapper, QuantizeLayer)):
            return True
        if isinstance(layer, tf.keras.Model) and has_qat_layers(layer):
            return True
    return False


def _real_activation(activation):
    '''Undo the activation substitutions `tfmot` makes inside a wrapper.'''
    _, _, QuantizeAwareActivation = _qat_classes()
    for _ in range(4):
        if isinstance(activation, QuantizeAwareActivation):
            activation = activation.activation
        elif type(activation).__name__ == 'NoOpActivation':
            # placed on a convolution that is followed by a batch norm, so
            # that nothing is quantized between the two; it is the identity
            activation = tf.keras.activations.linear
        else:
            break
    return activation


def _plain_copy(wrapper):
    '''A fresh, unwrapped copy of the layer inside a `QuantizeWrapper`.'''
    inner = wrapper.layer
    if not hasattr(inner, 'activation'):
        return type(inner).from_config(inner.get_config())
    substituted = inner.activation
    inner.activation = _real_activation(substituted)
    try:
        config = inner.get_config()
    finally:
        inner.activation = substituted
    return type(inner).from_config(config)


def _copy_wrapped_weights(wrapper, plain):
    '''Copy the trained weights of the layer inside `wrapper` into `plain`.

    The wrapped layer cannot simply be asked for its weights: while wrapped,
    its quantized weights (e.g. `kernel`) are tracked by the wrapper, not by
    the layer. Variables are therefore matched by their name relative to the
    layer (`kernel:0`, `bias:0`, `gamma:0`, ...), which the wrapper's own
    quantizer variables (`kernel_min:0`, `post_activation_max:0`,
    `optimizer_step:0`, ...) never collide with.
    '''
    if not plain.weights:
        return

    def relative(name, layer_name):
        return name.split(layer_name + '/', 1)[-1] if layer_name + '/' in name else name.rsplit('/', 1)[-1]

    available = {}
    for variable in wrapper.weights:
        available[relative(variable.name, wrapper.layer.name)] = variable
        available.setdefault(variable.name.rsplit('/', 1)[-1], variable)
    values = []
    for variable in plain.weights:
        key = relative(variable.name, plain.name)
        source = available.get(key, available.get(variable.name.rsplit('/', 1)[-1]))
        if source is None or tuple(source.shape) != tuple(variable.shape):
            raise RuntimeError(
                "strip_qat: cannot find the trained value of '{}' in the QAT wrapper '{}' (has: {})".format(
                    variable.name, wrapper.name, sorted(available)))
        values.append(source.numpy())
    plain.set_weights(values)


def strip_qat(model):
    '''
    Return a plain float copy of a QAT-wrapped model: every `tfmot`
    `QuantizeWrapper` is replaced by a fresh copy of the layer it wraps
    (with the trained weights), and every input `QuantizeLayer` is removed.

    Why this exists: the heads of this library are QAT-wrapped while the
    backbones are not. When such a model is converted to full-integer
    TFLite, the fake-quantization ranges of the head and the calibrated
    ranges of the backbone can disagree on a shared tensor, and the
    converter then leaves an operator in floating point (typically the last
    convolution of the backbone), which an integer-only accelerator cannot
    run. The QAT wrappers also quantize convolution weights per channel
    regardless of the converter settings. Converting the stripped model
    quantizes the whole graph post-training with one consistent calibration.

    Input
    ----------
        model: a `tf.keras.Model`, QAT-wrapped or not.

    Output
    ----------
        a new float model (weights copied, not shared), or `model` itself if
        it contains no QAT wrappers. `preprocessing` is carried over.
    '''
    if not has_qat_layers(model):
        return model
    QuantizeWrapper, QuantizeLayer, _ = _qat_classes()

    sources = {}

    def clone_function(layer):
        if isinstance(layer, QuantizeWrapper):
            plain = _plain_copy(layer)
            sources[plain] = layer
            return plain
        if isinstance(layer, QuantizeLayer):
            return tf.keras.layers.Activation('linear', name=layer.name)
        if isinstance(layer, tf.keras.Model) and has_qat_layers(layer):
            return strip_qat(layer)
        return layer

    stripped = tf.keras.models.clone_model(model, clone_function=clone_function)
    for plain, wrapper in sources.items():
        _copy_wrapped_weights(wrapper, plain)
    if hasattr(model, 'preprocessing'):
        stripped.preprocessing = model.preprocessing
    return stripped


# ========================= #
# Output activations

def _carry_attributes(source, target):
    '''Copy the attributes this library attaches to its models.'''
    for attribute in ('preprocessing', 'head_names', '_sak_reparameterized'):
        if hasattr(source, attribute):
            setattr(target, attribute, getattr(source, attribute))
    return target


def _output_activation(layer):
    '''Classify the activation a layer applies.

    Returns `(kind, name)`: kind is 'standalone' for a layer that only
    applies an activation, 'fused' for a layer with weights and a built-in
    activation, None otherwise. Looks inside a QAT wrapper.
    '''
    QuantizeWrapper, _, _ = _qat_classes()
    inner = layer.layer if isinstance(layer, QuantizeWrapper) else layer
    if type(inner).__name__ == 'TFOpLambda':
        symbol = (getattr(inner, 'symbol', '') or '').split('.')[-1]
        return ('standalone', symbol) if symbol else (None, None)
    if isinstance(inner, tf.keras.layers.Softmax):
        return 'standalone', 'softmax'
    if isinstance(inner, tf.keras.layers.Activation):
        return 'standalone', getattr(_real_activation(inner.activation), '__name__', None)
    if hasattr(inner, 'activation') and inner.activation is not None:
        return 'fused', getattr(_real_activation(inner.activation), '__name__', None)
    return None, None


def _producer(tensor):
    '''The layer that produced a Keras tensor, and that layer's inputs.'''
    layer, node_index, _ = tensor._keras_history
    inputs = tf.nest.flatten(layer._inbound_nodes[node_index].keras_inputs)
    return layer, inputs


def drop_output_activations(model, activations=('sigmoid', 'softmax')):
    '''
    Return a model whose outputs are taken before their final sigmoid or
    softmax, for accelerators that do not run those operators.

    A final sigmoid or softmax has no weights, so dropping it at export
    needs no retraining: the exported model returns the value that fed the
    activation (the logits) and the application applies the activation
    itself. On a uint8 output that is a 256-entry lookup table for a
    sigmoid. The operator lists that motivate this: `LOGISTIC` is not listed
    for the Axis ARTPEC-8 DLPU, `SOFTMAX` is not listed for ARTPEC-9.

    Handled cases, per output: a standalone activation layer (plain or
    QAT-wrapped, or a `tf.sigmoid` call), reached through any identity
    layers that follow it; and a plain `Dense` or convolution with the
    activation built in, which is replaced by a copy without activation
    (weights copied). A QAT-wrapped layer with a built-in activation is
    left unchanged with a warning: pass the model through `strip_qat` first.

    Input
    ----------
        model: a functional `tf.keras.Model`.
        activations: names of the activations to drop.

    Output
    ----------
        A new model sharing the layers of `model`, or `model` itself if no
        output ends with one of `activations`. The attribute
        `dropped_output_activations` lists, per output, the name of the
        activation that the application now has to apply (None where the
        output is unchanged). `preprocessing` is carried over. An output
        that was named by an identity layer, as the heads of `multi_task`
        are, keeps its name.
    '''
    QuantizeWrapper, _, _ = _qat_classes()
    outputs, dropped = [], []
    for tensor in model.outputs:
        current = tensor
        new_output, name = tensor, None
        label = None
        for _ in range(8):
            layer, inputs = _producer(current)
            kind, activation = _output_activation(layer)
            if len(inputs) != 1:
                break
            if kind == 'standalone' and activation == 'linear':
                # an identity layer that names the output (see `multi_task`)
                label = label or layer.name
                current = inputs[0]
                continue
            if kind == 'standalone' and activation in activations:
                new_output, name = inputs[0], activation
            elif kind == 'fused' and activation in activations:
                if isinstance(layer, QuantizeWrapper):
                    warnings.warn(
                        "drop_output_activations: output '{}' ends with a QAT-wrapped layer whose {} is built in; "
                        "it is left unchanged. Pass the model through strip_qat first.".format(layer.name, activation))
                else:
                    config = layer.get_config()
                    config['activation'] = 'linear'
                    config['name'] = layer.name + '_logits'
                    plain = type(layer).from_config(config)
                    new_output, name = plain(inputs[0]), activation
                    plain.set_weights(layer.get_weights())
            break
        if name is not None and label is not None:
            # keep the name the output had, so that a head is still found by name
            new_output = tf.keras.layers.Activation('linear', name=label)(new_output)
        outputs.append(new_output)
        dropped.append(name)

    if not any(dropped):
        return model
    result = tf.keras.Model(model.inputs, outputs if len(outputs) > 1 else outputs[0], name=model.name)
    _carry_attributes(model, result)
    result.dropped_output_activations = dropped
    return result


# ========================= #
# Cross-layer equalization

@dataclass
class EqualizationResult:
    '''Outcome of `equalize_cross_layer`.

    `pairs` is the number of consecutive layer pairs that were equalized,
    `iterations` how many sweeps were run, `max_scale` the largest
    per-channel rescaling factor applied (or its inverse, whichever is
    larger), and `restore()` puts the original weights back.
    '''
    pairs: int
    iterations: int
    max_scale: float
    _saved: list = field(default_factory=list, repr=False)

    def restore(self):
        '''Put back the weights the model had before equalization.'''
        for layer, weights in self._saved:
            layer.set_weights(weights)
        self._saved = []


def _is_plain_relu_or_identity(layer):
    L = tf.keras.layers
    if isinstance(layer, (L.ZeroPadding2D, L.Dropout)):
        return True
    if isinstance(layer, L.Activation):
        return layer.activation in (tf.keras.activations.relu, tf.keras.activations.linear)
    if isinstance(layer, L.ReLU):
        return layer.max_value is None and not layer.negative_slope and not layer.threshold
    return False


def _is_equalizable_conv(layer):
    L = tf.keras.layers
    if isinstance(layer, L.DepthwiseConv2D):
        return layer.depth_multiplier == 1
    if type(layer) is L.Conv2D:
        return layer.groups == 1
    return False


def _equalization_pairs(model, pairs=None):
    '''Find `conv -> [batch norm] -> [relu / padding]* -> conv` chains in
    which every intermediate tensor has exactly one consumer.'''
    L = tf.keras.layers
    pairs = [] if pairs is None else pairs
    if not hasattr(model, '_nodes_by_depth'):
        return pairs

    consumers = collections.defaultdict(list)
    for nodes in model._nodes_by_depth.values():
        for node in nodes:
            for tensor in tf.nest.flatten(node.keras_inputs):
                consumers[id(tensor)].append(node)
    external = {id(t) for t in tf.nest.flatten(model.outputs)}

    def single_consumer(tensor):
        users = consumers.get(id(tensor), [])
        if id(tensor) in external or len(users) != 1:
            return None
        node = users[0]
        if len(tf.nest.flatten(node.keras_inputs)) != 1:
            return None
        return node

    for nodes in model._nodes_by_depth.values():
        for node in nodes:
            layer = node.layer
            if not _is_equalizable_conv(layer):
                continue
            if layer.activation not in (tf.keras.activations.relu, tf.keras.activations.linear):
                continue
            outputs = tf.nest.flatten(node.outputs)
            if len(outputs) != 1:
                continue
            nxt = single_consumer(outputs[0])
            bn = None
            if nxt is not None and isinstance(nxt.layer, L.BatchNormalization):
                bn = nxt.layer
                if not bn.scale:
                    continue
                nxt = single_consumer(tf.nest.flatten(nxt.outputs)[0])
            while nxt is not None and _is_plain_relu_or_identity(nxt.layer):
                nxt = single_consumer(tf.nest.flatten(nxt.outputs)[0])
            if nxt is None or not _is_equalizable_conv(nxt.layer):
                continue
            pairs.append((layer, bn, nxt.layer))

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            _equalization_pairs(layer, pairs)
    return pairs


def _bn_gain(bn):
    '''Per-channel multiplier a batch norm applies at inference.'''
    return bn.gamma.numpy() / np.sqrt(bn.moving_variance.numpy() + bn.epsilon)


def equalize_cross_layer(model, max_iterations=40, tolerance=1e-3):
    '''
    Cross-layer equalization (Nagel et al., 2019, "Data-Free Quantization
    Through Weight Equalization and Bias Correction").

    For two consecutive convolutions separated only by a ReLU, output
    channel `i` of the first can be divided by any positive `s_i` and input
    channel `i` of the second multiplied by `s_i` without changing what the
    network computes. Choosing `s_i` so that both layers end up with the
    same weight range in that channel evens out the per-channel ranges
    inside each weight tensor, which is what **per-tensor** quantization
    needs: with one scale for the whole tensor, a single channel with large
    weights would otherwise cost every other channel its resolution.
    Depthwise and reparameterized (fused) networks are the typical victims.

    What is equalized: every chain
    `Conv2D | DepthwiseConv2D -> [BatchNormalization] -> [ReLU | ZeroPadding2D]* -> Conv2D | DepthwiseConv2D`
    whose intermediate tensors have a single consumer and are not model
    outputs (so backbone feature taps, residual branches, squeeze-excite
    gates and concatenations are left alone), inside the model and any
    nested sub-model. Chains through ReLU6 or other non-homogeneous
    activations are skipped. Weights are modified **in place**; the model
    must not contain QAT wrappers (apply `strip_qat` first).

    Input
    ----------
        model: a plain float `tf.keras.Model` (functional).
        max_iterations: maximum number of sweeps over all pairs.
        tolerance: stop when no scaling factor differs from 1 by more than
                   this (relative).

    Output
    ----------
        an `EqualizationResult`. Its `restore()` method puts the original
        weights back, which matters when layers are shared with a model
        that is still being trained.
    '''
    if has_qat_layers(model):
        raise ValueError('equalize_cross_layer needs a plain model; apply strip_qat() first.')
    L = tf.keras.layers
    pairs = _equalization_pairs(model)
    result = EqualizationResult(pairs=len(pairs), iterations=0, max_scale=1.0)
    if not pairs:
        return result

    # the batch norm (if any) that follows each convolution, for its effective weight range
    bn_after = {id(first): bn for first, bn, _ in pairs if bn is not None}
    touched = {}
    for first, bn, second in pairs:
        for layer in (first, bn, second):
            if layer is not None and id(layer) not in touched:
                touched[id(layer)] = (layer, layer.get_weights())
    result._saved = list(touched.values())

    def kernel_of(layer):
        return layer.depthwise_kernel if isinstance(layer, L.DepthwiseConv2D) else layer.kernel

    def out_range(layer, bn):
        kernel = np.abs(kernel_of(layer).numpy())
        r = kernel.max(axis=(0, 1, 3)) if isinstance(layer, L.DepthwiseConv2D) else kernel.max(axis=(0, 1, 2))
        return r if bn is None else r * np.abs(_bn_gain(bn))

    def in_range(layer):
        kernel = np.abs(kernel_of(layer).numpy())
        bn = bn_after.get(id(layer))
        if bn is not None:
            gain = np.abs(_bn_gain(bn))
            # depthwise: (kh, kw, c, 1), output channel = input channel; regular: (kh, kw, cin, cout)
            kernel = kernel * (gain[None, None, :, None] if isinstance(layer, L.DepthwiseConv2D) else gain[None, None, None, :])
        return kernel.max(axis=(0, 1, 3))

    for iteration in range(max_iterations):
        worst = 0.0
        for first, bn, second in pairs:
            r1, r2 = out_range(first, bn), in_range(second)
            valid = (r1 > 1e-12) & (r2 > 1e-12)
            scale = np.ones_like(r1)
            scale[valid] = np.sqrt(r1[valid] / r2[valid])
            deviation = float(np.abs(np.log(scale)).max())
            worst = max(worst, deviation)
            result.max_scale = max(result.max_scale, float(np.exp(deviation)))

            # first layer: divide output channel i by scale_i
            if bn is not None:
                bn.gamma.assign(bn.gamma.numpy() / scale)
                if bn.center:
                    bn.beta.assign(bn.beta.numpy() / scale)
            else:
                if isinstance(first, L.DepthwiseConv2D):
                    first.depthwise_kernel.assign(first.depthwise_kernel.numpy() / scale[None, None, :, None])
                else:
                    first.kernel.assign(first.kernel.numpy() / scale[None, None, None, :])
                if first.use_bias:
                    first.bias.assign(first.bias.numpy() / scale)
            # second layer: multiply input channel i by scale_i
            kernel = kernel_of(second)
            kernel.assign(kernel.numpy() * scale[None, None, :, None])
        result.iterations = iteration + 1
        if worst < tolerance:
            break
    return result


# ========================= #
# Measuring quantization error

def quantization_error(model, tflite_model, inputs):
    '''
    Compare a float Keras model with its quantized TFLite export on the same
    inputs.

    Input
    ----------
        model: the float `tf.keras.Model` the flatbuffer was converted from
               (single input).
        tflite_model: the flatbuffer `bytes` (integer or float input/output).
        inputs: a float array `(N, H, W, C)` in the model's input domain
                (already normalized).

    Output
    ----------
        a list with one dict per model output, in the order of
        `model.outputs`: `sqnr_db` (signal-to-quantization-noise ratio,
        higher is better; each 6 dB is roughly one more bit of precision),
        `rmse`, `max_abs_error`, and `float_std` (spread of the reference
        output, for scale).
    '''
    # Called directly, in chunks, rather than through `model.predict`: predict
    # caches a traced function that would not see weights or layer attributes
    # changed since the last call (e.g. by `equalize_cross_layer`).
    inputs = np.asarray(inputs, dtype=np.float32)
    chunks = []
    for start in range(0, len(inputs), 16):
        out = model(inputs[start:start + 16], training=False)
        chunks.append([np.asarray(t) for t in (out if isinstance(out, (list, tuple)) else [out])])
    reference = [np.concatenate([c[k] for c in chunks]) for k in range(len(chunks[0]))]

    interpreter = _interpreter(tflite_model)
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()

    def to_input(x):
        if in_detail['dtype'] in (np.uint8, np.int8):
            scale, zero = in_detail['quantization']
            info = np.iinfo(in_detail['dtype'])
            return np.clip(np.round(x / scale + zero), info.min, info.max).astype(in_detail['dtype'])
        return x.astype(in_detail['dtype'])

    def from_output(detail):
        y = interpreter.get_tensor(detail['index'])
        if detail['dtype'] in (np.uint8, np.int8):
            scale, zero = detail['quantization']
            return (y.astype(np.float32) - zero) * scale
        return y.astype(np.float32)

    collected = [[] for _ in out_details]
    for sample in inputs:
        interpreter.set_tensor(in_detail['index'], to_input(sample[None]))
        interpreter.invoke()
        for k, detail in enumerate(out_details):
            collected[k].append(from_output(detail)[0])
    quantized = [np.stack(c) for c in collected]

    # TFLite does not guarantee the Keras output order: match by shape.
    order, used = [], set()
    for ref in reference:
        match = next((k for k, q in enumerate(quantized) if k not in used and q.shape == ref.shape), None)
        if match is None:
            raise ValueError('Cannot match TFLite outputs {} to Keras outputs {}'.format(
                [q.shape for q in quantized], [r.shape for r in reference]))
        used.add(match)
        order.append(match)

    report = []
    for ref, k in zip(reference, order):
        err = quantized[k] - ref
        signal = float(np.sum(ref.astype(np.float64) ** 2))
        noise = float(np.sum(err.astype(np.float64) ** 2))
        report.append({
            'sqnr_db': float('inf') if noise == 0 else float(10.0 * np.log10(max(signal, 1e-30) / noise)),
            'rmse': float(np.sqrt(np.mean(err ** 2))),
            'max_abs_error': float(np.abs(err).max()),
            'float_std': float(ref.std()),
        })
    return report


# ========================= #
# TFLite conversion

_IO_DTYPES = {'uint8': tf.uint8, 'int8': tf.int8}


def convert_to_tflite(model, representative_dataset=None, profile='default', per_channel=None, io_dtype=None,
                      float16=False):
    '''
    Convert a Keras model to a TFLite flatbuffer for a deployment profile.

    Input
    ----------
        model: a `tf.keras.Model`.
        representative_dataset: a callable returning a generator of
            `[float32 batch-of-1 input]` lists (the TFLite converter
            convention). Required unless `float16=True`.
        profile: a profile name (see `PROFILES`) or a `DeploymentProfile`.
        per_channel: override the profile's weight-quantization granularity.
        io_dtype: override the profile's input/output type ('uint8', 'int8',
                  or 'float32' to keep float input and output).
        float16: if True, ignore the integer settings and produce a float16
                 model (float32 input and output) instead.

    Output
    ----------
        the flatbuffer, as `bytes`. Full-integer conversion allows
        `TFLITE_BUILTINS_INT8` operators only and therefore fails if any
        operator cannot be quantized.
    '''
    profile = get_profile(profile)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if float16:
        converter.target_spec.supported_types = [tf.float16]
        return converter.convert()

    if representative_dataset is None:
        raise ValueError('`representative_dataset` is required for full-integer conversion.')
    per_channel = profile.per_channel if per_channel is None else per_channel
    io_dtype = profile.io_dtype if io_dtype is None else io_dtype
    if io_dtype not in ('uint8', 'int8', 'float32'):
        raise ValueError("`io_dtype` must be 'uint8', 'int8' or 'float32', got {!r}".format(io_dtype))

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if io_dtype != 'float32':
        converter.inference_input_type = _IO_DTYPES[io_dtype]
        converter.inference_output_type = _IO_DTYPES[io_dtype]
    if not per_channel:
        # The only switch the TFLite converter offers for per-tensor weights.
        converter._experimental_disable_per_channel = True
    return converter.convert()


# ========================= #
# Inspection of exported models

_WEIGHTED_OPS = ('CONV_2D', 'DEPTHWISE_CONV_2D', 'TRANSPOSE_CONV', 'FULLY_CONNECTED')
_FLOAT_DTYPES = (np.float32, np.float16, np.float64)


def _interpreter(tflite_model):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return tf.lite.Interpreter(model_content=tflite_model)


def _graph(tflite_model):
    interpreter = _interpreter(tflite_model)
    ops = interpreter._get_ops_details()
    tensors = {t['index']: t for t in interpreter.get_tensor_details()}
    inputs = [d['index'] for d in interpreter.get_input_details()]
    outputs = [d['index'] for d in interpreter.get_output_details()]
    return interpreter, ops, tensors, inputs, outputs


def tflite_outputs(tflite_model):
    '''
    Describe the outputs of a TFLite model in the order the file stores them.

    The converter does not keep the output order of the Keras model, and
    the tensor names in the file (`StatefulPartitionedCall:0`, ...) do not
    say which head an output belongs to. A runtime that returns outputs by
    position, as an accelerator's inference service does, needs this
    mapping for any model with more than one output.

    Input
    ----------
        tflite_model: the flatbuffer as `bytes`, or the path of a `.tflite`
                      file.

    Output
    ----------
        A list with one dict per output, in file order: `position`, `name`
        (the name of the Keras output, e.g. the head name of a `multi_task`
        model; the tensor name if the file has no signature), `shape`,
        `dtype`, `scale` and `zero_point` (real value = scale * (stored
        value - zero_point); scale 0 for a float output).
    '''
    if isinstance(tflite_model, (bytes, bytearray)):
        interpreter = tf.lite.Interpreter(model_content=bytes(tflite_model))
    else:
        interpreter = tf.lite.Interpreter(model_path=str(tflite_model))

    names = {}
    try:
        signatures = interpreter._get_full_signature_list()
        signature = signatures.get('serving_default') or next(iter(signatures.values()))
        names = {tensor_index: name for name, tensor_index in signature['outputs'].items()}
    except Exception:  # no signature in the file, or a private API that moved
        names = {}

    described = []
    for position, detail in enumerate(interpreter.get_output_details()):
        scale, zero_point = detail.get('quantization', (0.0, 0))
        described.append({
            'position': position,
            'name': names.get(detail['index'], detail['name']),
            'shape': tuple(int(d) for d in detail['shape']),
            'dtype': np.dtype(detail['dtype']).name,
            'scale': float(scale),
            'zero_point': int(zero_point),
        })
    return described


def list_tflite_ops(tflite_model):
    '''
    Count the builtin operators of a TFLite flatbuffer.

    Output
    ----------
        a `collections.Counter` mapping operator name to occurrences.
    '''
    _, ops, _, _, _ = _graph(tflite_model)
    return collections.Counter(op['op_name'] for op in ops)


def _check_mul_by_constant_only(ops, tensors, dynamic):
    bad = sum(1 for op in ops if op['op_name'] == 'MUL' and all(i in dynamic for i in op['inputs'] if i >= 0))
    return ['{} MUL operator(s) multiply two computed tensors (only multiplication by a constant is listed)'.format(bad)] if bad else []


def _check_global_mean_7_or_8(ops, tensors, dynamic):
    bad = 0
    for op in ops:
        if op['op_name'] != 'MEAN':
            continue
        shape = tuple(tensors[op['inputs'][0]]['shape'])
        if len(shape) != 4 or shape[1] != shape[2] or shape[1] not in (7, 8):
            bad += 1
    return ['{} MEAN operator(s) do not reduce a 7x7 or 8x8 map'.format(bad)] if bad else []


def _check_resize_2x(ops, tensors, dynamic):
    bad = 0
    for op in ops:
        if op['op_name'] not in ('RESIZE_BILINEAR', 'RESIZE_NEAREST_NEIGHBOR'):
            continue
        src = tuple(tensors[op['inputs'][0]]['shape'])
        dst = tuple(tensors[op['outputs'][0]]['shape'])
        if len(src) != 4 or any(d not in (2 * s, 2 * s - 1) for s, d in zip(src[1:3], dst[1:3])):
            bad += 1
    return ['{} resize operator(s) do not scale by 2 (output size 2n or 2n-1)'.format(bad)] if bad else []


_CONSTRAINT_CHECKS = {
    'mul_by_constant_only': _check_mul_by_constant_only,
    'global_mean_7_or_8': _check_global_mean_7_or_8,
    'resize_2x': _check_resize_2x,
}


@dataclass
class LintReport:
    '''Result of `lint_tflite`.

    `errors` are certain problems (the file does not match the profile).
    `unsupported_ops` maps operators missing from the profile's operator list
    to their count; they are expected to run on the CPU. `warnings` are
    constraints or inefficiencies worth a look. `ok` is True when there are
    no errors and no unsupported operators.
    '''
    profile: str
    ops: collections.Counter
    errors: list = field(default_factory=list)
    unsupported_ops: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    ops_source: str = ''
    notes: str = ''

    @property
    def ok(self):
        return not self.errors and not self.unsupported_ops

    def __str__(self):
        total = sum(self.ops.values())
        lines = ["TFLite lint, profile '{}': {} operators, {}".format(
            self.profile, total, 'OK' if self.ok else 'ISSUES FOUND')]
        lines.append('  operators: ' + ', '.join('{} x{}'.format(k, v) for k, v in sorted(self.ops.items())))
        for e in self.errors:
            lines.append('  ERROR: ' + e)
        if self.unsupported_ops:
            lines.append('  NOT ON THE TARGET OPERATOR LIST (expected to run on the CPU): ' + ', '.join(
                '{} x{}'.format(k, v) for k, v in sorted(self.unsupported_ops.items())))
            lines.append('    list source: ' + self.ops_source)
        for w in self.warnings:
            lines.append('  warning: ' + w)
        if self.notes:
            lines.append('  note: ' + self.notes)
        return '\n'.join(lines)


def lint_tflite(tflite_model, profile='default', per_channel=None, io_dtype=None):
    '''
    Check an exported TFLite flatbuffer against a deployment profile.

    Checks performed:

    * input/output tensor types match the profile;
    * no operator computes in floating point inside an integer model
      ("float islands", which an integer-only accelerator cannot run);
    * weight tensors are quantized per tensor when the profile asks for it;
    * every operator is on the profile's operator list, when it has one;
    * the profile's extra per-operator constraints;
    * convolution output channels are multiples of the profile's
      `channel_multiple`.

    Input
    ----------
        tflite_model: the flatbuffer `bytes`.
        profile: a profile name or `DeploymentProfile`.
        per_channel, io_dtype: the overrides passed to `convert_to_tflite`,
            if any, so the check uses the same expectations.

    Output
    ----------
        a `LintReport`; `print(report)` gives a readable summary and
        `report.ok` a boolean.
    '''
    profile = get_profile(profile)
    per_channel = profile.per_channel if per_channel is None else per_channel
    io_dtype = profile.io_dtype if io_dtype is None else io_dtype
    interpreter, ops, tensors, inputs, outputs = _graph(tflite_model)
    counts = collections.Counter(op['op_name'] for op in ops)
    report = LintReport(profile=profile.name, ops=counts, ops_source=profile.ops_source, notes=profile.notes)

    # input / output types
    expected = {'uint8': np.uint8, 'int8': np.int8, 'float32': np.float32}[io_dtype]
    for kind, indices in (('input', inputs), ('output', outputs)):
        for index in indices:
            if tensors[index]['dtype'] != expected:
                report.errors.append('{} tensor is {}, profile expects {}'.format(
                    kind, np.dtype(tensors[index]['dtype']).name, io_dtype))

    # float islands
    if io_dtype != 'float32':
        float_ops = collections.Counter()
        for op in ops:
            if any(tensors[i]['dtype'] in _FLOAT_DTYPES for i in op['outputs'] if i >= 0):
                float_ops[op['op_name']] += 1
        if float_ops:
            report.errors.append('floating-point operators inside an integer model: ' + ', '.join(
                '{} x{}'.format(k, v) for k, v in sorted(float_ops.items()))
                + ' (if the model has QAT wrappers, convert `strip_qat(model)` instead)')

    # weight quantization granularity
    per_channel_weights = 0
    for op in ops:
        if op['op_name'] in _WEIGHTED_OPS and len(op['inputs']) > 1:
            scales = tensors[op['inputs'][1]]['quantization_parameters']['scales']
            if len(scales) > 1:
                per_channel_weights += 1
    if not per_channel and per_channel_weights:
        report.errors.append(
            '{} weight tensor(s) are quantized per channel, profile expects per tensor (QAT wrappers quantize '
            'convolution weights per channel whatever the converter is told; convert `strip_qat(model)` '
            'instead)'.format(per_channel_weights))

    # operator list
    if profile.supported_ops is not None:
        report.unsupported_ops = {k: v for k, v in counts.items()
                                  if k not in profile.supported_ops and k != 'DEQUANTIZE'}

    # extra constraints
    dynamic = set(inputs)
    for op in ops:
        dynamic.update(i for i in op['outputs'] if i >= 0)
    for name in profile.constraints:
        report.warnings.extend(_CONSTRAINT_CHECKS[name](ops, tensors, dynamic))

    # channel multiples
    if profile.channel_multiple:
        off = 0
        for op in ops:
            if op['op_name'] in ('CONV_2D', 'TRANSPOSE_CONV'):
                shape = tensors[op['outputs'][0]]['shape']
                if len(shape) == 4 and shape[-1] % profile.channel_multiple:
                    off += 1
        if off:
            report.warnings.append('{} convolution(s) have an output channel count that is not a multiple of {}'.format(
                off, profile.channel_multiple))

    return report
