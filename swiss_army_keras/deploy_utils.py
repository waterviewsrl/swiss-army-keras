'''
Deployment helpers: target profiles, structural reparameterization of full
models, TFLite conversion, and a check of the exported operators against
what a target accelerates.

Typical use (also what `quantization_utils.Quantizer` does)::

    from swiss_army_keras import deploy_utils

    plain = deploy_utils.strip_qat(deploy_utils.reparameterize_model(model))
    plain = deploy_utils.drop_output_activations(plain)   # optional: no final sigmoid/softmax
    plain = deploy_utils.fold_batch_norms(plain)      # needed for bias correction
    deploy_utils.equalize_cross_layer(plain)          # for per-tensor targets
    deploy_utils.correct_bias(plain, calibration_images)   # optional, per-tensor targets
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
        static_batch: if set, the model is converted with this fixed batch
                      size. Accelerators take one frame at a time and no
                      dynamic shapes; a fixed batch also lets the converter
                      fold the shape arithmetic of reshapes away.
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
    static_batch: Optional[int] = None
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
        static_batch=1,
        notes='Run `edgetpu_compiler` on the exported file; it reports which operators stay on the CPU.',
    ),
    'artpec8': DeploymentProfile(
        name='artpec8',
        description='Axis ARTPEC-8 DLPU: per-tensor weights, uint8 input/output.',
        per_channel=False,
        integer_only=True,
        static_batch=1,
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
        static_batch=1,
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
    from swiss_army_keras._backbone_plain import QAREPVGG_CONFIGS, reparameterize_qarepvgg
    return (
        (MOBILEONE_CONFIGS, '_mobileone_config', reparameterize_mobileone),
        (REPVIT_CONFIGS, '_repvit_config', reparameterize_repvit),
        (FASTVIT_CONFIGS, '_fastvit_config', reparameterize_fastvit),
        (QAREPVGG_CONFIGS, '_qarepvgg_config', reparameterize_qarepvgg),
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
            # compare without the batch dimension: the model may have been built with a fixed batch size
            if tuple(old.shape[1:]) != tuple(new.shape[1:]):
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
        it contains no QAT wrappers. `preprocessing` is carried over. For a
        model that Keras cannot clone (one with `tf` operations around a
        QAT sub-model, such as the full detectors or a model wrapped by
        `uint8_input_wrapper`), the unwrapped layers are shared with
        `model` instead of copied.
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

    try:
        stripped = tf.keras.models.clone_model(model, clone_function=clone_function)
    except Exception:
        # Keras cannot rebuild some graphs from their configuration (`tf`
        # operations with constant arguments, as in the box decoding of the
        # detectors and in `uint8_input_wrapper`). Replay the graph instead:
        # the layers that are not QAT wrappers are then shared with `model`.
        sources.clear()
        cache = {}

        def substitute(layer):
            if id(layer) not in cache:
                cache[id(layer)] = clone_function(layer)
            return cache[id(layer)]

        stripped = _replay(model, substitute)
    for plain, wrapper in sources.items():
        _copy_wrapped_weights(wrapper, plain)
    return _carry_attributes(model, stripped)


# ========================= #
# Output activations

def _carry_attributes(source, target):
    '''Copy the attributes this library attaches to its models.'''
    for attribute in ('preprocessing', 'head_names', 'strides', 'reg_max', 'detection',
                      'dropped_output_activations', '_sak_reparameterized'):
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
    if isinstance(inner, tf.keras.layers.UnitNormalization):
        return 'standalone', 'l2_normalize'
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


def drop_output_activations(model, activations=('sigmoid', 'softmax', 'l2_normalize')):
    '''
    Return a model whose outputs are taken before their final sigmoid,
    softmax or L2 normalization, for accelerators that do not run those
    operators.

    A final sigmoid or softmax has no weights, so dropping it at export
    needs no retraining: the exported model returns the value that fed the
    activation (the logits) and the application applies the activation
    itself. On a uint8 output that is a 256-entry lookup table for a
    sigmoid. The operator lists that motivate this: `LOGISTIC` is not listed
    for the Axis ARTPEC-8 DLPU, `SOFTMAX` is not listed for ARTPEC-9, and
    `L2_NORMALIZATION` (the `UnitNormalization` that ends an embedding
    head) is listed for neither.

    A reshape or flatten that follows the activation (as in the pose head)
    is kept: the activation is removed from behind it.

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
        reshapes = []
        for _ in range(8):
            layer, inputs = _producer(current)
            kind, activation = _output_activation(layer)
            if len(inputs) != 1:
                break
            if isinstance(layer, (tf.keras.layers.Reshape, tf.keras.layers.Flatten)):
                # a reshape after the activation (the pose head): look behind it
                reshapes.insert(0, layer)
                current = inputs[0]
                continue
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
        if name is not None:
            for layer in reshapes:
                new_output = layer(new_output)
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
# Graph rewrites

def _replay(model, substitute):
    '''
    Rebuild a functional model by calling its layers again on new inputs.

    `substitute(layer)` returns the layer to call in place of `layer` (the
    layer itself to keep it). Layers that are kept are shared with `model`,
    weights included. Unlike `tf.keras.models.clone_model` this does not go
    through layer configurations, so it also works on graphs that hold
    operations Keras cannot rebuild from a configuration (`tf` operations
    with constant arguments, as the in-graph box decoding of the detectors
    and `uint8_input_wrapper` have).
    '''
    usage = model._tensor_usage_count
    tensor_dict = {}
    new_inputs = []
    for tensor in model.inputs:
        new = tf.keras.Input(batch_shape=tensor.shape, dtype=tensor.dtype, name=tensor._keras_history.layer.name)
        new_inputs.append(new)
        tensor_dict[str(id(tensor))] = [new] * usage[str(id(tensor))]

    for depth in sorted(model._nodes_by_depth, reverse=True):
        for node in model._nodes_by_depth[depth]:
            if node.is_input or any(t_id not in tensor_dict for t_id in node.flat_input_ids):
                continue
            args, kwargs = node.map_arguments(tensor_dict)
            outputs = substitute(node.layer)(*args, **kwargs)
            for t_id, tensor in zip(node.flat_output_ids, tf.nest.flatten(outputs)):
                tensor_dict[t_id] = [tensor] * usage[t_id]

    new_outputs = [tensor_dict[str(id(tensor))].pop() for tensor in model.outputs]
    rebuilt = tf.keras.Model(new_inputs if len(new_inputs) > 1 else new_inputs[0],
                             new_outputs if len(new_outputs) > 1 else new_outputs[0], name=model.name)
    return _carry_attributes(model, rebuilt)


def select_outputs(model, outputs):
    '''
    Keep some of a model's outputs, for instance the main output of a
    network trained with auxiliary heads (`ddrnet`, `pidnet`).

    Input
    ----------
        model: a functional `tf.keras.Model`.
        outputs: an output index or name, or a list of them. Names are
                 those of `model.output_names`.

    Output
    ----------
        A new model sharing the layers of `model`, with only the requested
        outputs, in the order given. Layers that no kept output depends on
        (the auxiliary heads) are left out, so they do not reach the
        exported file.
    '''
    wanted = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    names = list(model.output_names)
    tensors = []
    for item in wanted:
        if isinstance(item, str):
            if item not in names:
                raise ValueError("select_outputs: no output named '{}' (outputs: {})".format(item, names))
            item = names.index(item)
        if not -len(model.outputs) <= item < len(model.outputs):
            raise ValueError('select_outputs: output index {} out of range ({} outputs)'.format(item, len(model.outputs)))
        tensors.append(model.outputs[item])
    selected = tf.keras.Model(model.inputs, tensors if len(tensors) > 1 else tensors[0], name=model.name)
    return _carry_attributes(model, selected)


def _has_global_average_pooling(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            return True
        if isinstance(layer, tf.keras.Model) and _has_global_average_pooling(layer):
            return True
    return False


def replace_global_pooling(model):
    '''
    Replace every `GlobalAveragePooling2D` by an `AveragePooling2D` over the
    whole feature map, followed by a flatten where the original dropped the
    spatial axes.

    The two compute the same values. They export differently: global
    pooling becomes a `MEAN` operator, the windowed form an
    `AVERAGE_POOL_2D`. `MEAN` is not on the Axis ARTPEC-8 operator list as
    published, and on ARTPEC-9 it is listed for 7x7 and 8x8 maps only.

    Input
    ----------
        model: a functional `tf.keras.Model` without QAT wrappers (pass it
               through `strip_qat` first). Nested sub-models are rewritten
               too. Layers that reduce with their own operations (GeM
               pooling, the attention blocks) are not touched.

    Output
    ----------
        A new model sharing every other layer with `model`, or `model`
        itself when it has no global average pooling.
    '''
    if has_qat_layers(model):
        raise ValueError('replace_global_pooling needs a model without QAT wrappers: call strip_qat first.')
    if not _has_global_average_pooling(model):
        return model

    replaced = {}

    def substitute(layer):
        if id(layer) in replaced:
            return replaced[id(layer)]
        new = layer
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            height, width = int(layer.input_shape[1]), int(layer.input_shape[2])
            pool = tf.keras.layers.AveragePooling2D(pool_size=(height, width), name=layer.name + '_window')
            if getattr(layer, 'keepdims', False):
                new = pool
            else:
                new = tf.keras.Sequential([pool, tf.keras.layers.Flatten(name=layer.name + '_flatten')],
                                          name=layer.name)
        elif isinstance(layer, tf.keras.Model) and _has_global_average_pooling(layer):
            new = replace_global_pooling(layer)
        replaced[id(layer)] = new
        return new

    return _replay(model, substitute)


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


def _static_batch_converter(model, batch_size):
    '''A converter for `model` with a fixed batch size, or None if the model
    cannot be traced that way (duplicate output names).'''
    names = list(model.output_names)
    if len(set(names)) != len(names):
        return None
    specs = [tf.TensorSpec([batch_size] + list(tensor.shape[1:]), tensor.dtype, name=tensor._keras_history.layer.name)
             for tensor in model.inputs]

    @tf.function(input_signature=specs)
    def serving(*args):
        outputs = model(list(args) if len(args) > 1 else args[0], training=False)
        return dict(zip(names, tf.nest.flatten(outputs)))

    return tf.lite.TFLiteConverter.from_concrete_functions([serving.get_concrete_function()], model)


def convert_to_tflite(model, representative_dataset=None, profile='default', per_channel=None, io_dtype=None,
                      float16=False, static_batch=None):
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
        static_batch: override the profile's fixed batch size (0 keeps the
                      batch dimension dynamic).

    Output
    ----------
        the flatbuffer, as `bytes`. Full-integer conversion allows
        `TFLITE_BUILTINS_INT8` operators only and therefore fails if any
        operator cannot be quantized.
    '''
    return _configured_converter(model, representative_dataset, profile, per_channel=per_channel,
                                 io_dtype=io_dtype, float16=float16, static_batch=static_batch).convert()


def _configured_converter(model, representative_dataset, profile, per_channel=None, io_dtype=None,
                          float16=False, static_batch=None):
    '''The `TFLiteConverter` that `convert_to_tflite` runs, not yet run.'''
    profile = get_profile(profile)
    static_batch = profile.static_batch if static_batch is None else static_batch
    converter = None
    if static_batch and not float16 and any(tensor.shape[0] is None for tensor in model.inputs):
        converter = _static_batch_converter(model, static_batch)
    if converter is None:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if float16:
        converter.target_spec.supported_types = [tf.float16]
        return converter

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
    return converter


# ========================= #
# Post-training quantization strategies

def _nodes(model):
    '''The nodes of a functional model in execution order, inputs excluded.'''
    for depth in sorted(model._nodes_by_depth, reverse=True):
        for node in model._nodes_by_depth[depth]:
            if not node.is_input:
                yield node


def _is_linear(activation):
    return activation is None or getattr(activation, '__name__', '') == 'linear'


def _foldable_pairs(model):
    '''`{id(conv): (conv, batch_norm)}` for every convolution of this model
    whose only consumer is a batch norm, and that is called once.'''
    usage = model._tensor_usage_count
    calls = collections.Counter(id(node.layer) for node in _nodes(model))
    pairs = {}
    for node in _nodes(model):
        layer = node.layer
        if not isinstance(layer, tf.keras.layers.BatchNormalization) or calls[id(layer)] != 1:
            continue
        inputs = tf.nest.flatten(node.keras_inputs)
        if len(inputs) != 1:
            continue
        conv = inputs[0]._keras_history.layer
        if type(conv) not in (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D):
            continue
        if calls[id(conv)] != 1 or usage.get(str(id(inputs[0])), 0) != 1 or not _is_linear(conv.activation):
            continue
        pairs[id(conv)] = (conv, layer)
    return pairs


def _has_batch_norm(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            return True
        if isinstance(layer, tf.keras.Model) and _has_batch_norm(layer):
            return True
    return False


def fold_batch_norms(model):
    '''
    Fold every batch norm that directly follows a convolution into that
    convolution.

    The TFLite converter does the same before it quantizes weights. Doing it
    on the Keras model gives the weights that will actually be quantized,
    which `correct_bias` needs, and lets `equalize_cross_layer` work on
    them directly.

    Input
    ----------
        model: a functional `tf.keras.Model` without QAT wrappers, in
               inference state. Nested sub-models are folded too.

    Output
    ----------
        A new model computing the same function. Folded convolutions are
        new layers with a bias; the batch norms become identity layers of
        the same name; every other layer is shared with `model`. Batch
        norms that do not follow a convolution (pre-activation blocks) are
        left as they are.
    '''
    if has_qat_layers(model):
        raise ValueError('fold_batch_norms needs a model without QAT wrappers: call strip_qat first.')
    if not _has_batch_norm(model):
        return model

    pairs = _foldable_pairs(model)
    batch_norms = {id(bn) for _, bn in pairs.values()}
    replaced, pending = {}, []

    def substitute(layer):
        if id(layer) in replaced:
            return replaced[id(layer)]
        new = layer
        if id(layer) in pairs:
            conv, bn = pairs[id(layer)]
            config = conv.get_config()
            config['use_bias'] = True
            new = type(conv).from_config(config)
            pending.append((new,) + tuple(_fused_conv_bn(conv, bn)))
        elif id(layer) in batch_norms:
            new = tf.keras.layers.Activation('linear', name=layer.name)
        elif isinstance(layer, tf.keras.Model) and _has_batch_norm(layer):
            new = fold_batch_norms(layer)
        replaced[id(layer)] = new
        return new

    folded = _replay(model, substitute)
    for layer, kernel, bias in pending:
        layer.set_weights([kernel, bias])
    return folded


def _fused_conv_bn(conv, bn):
    from swiss_army_keras.reparam_utils import fuse_conv_bn_layers
    return fuse_conv_bn_layers(conv, bn)


def _quantized_weights(kernel, per_channel, channel_axis):
    '''Symmetric 8-bit rounding of a weight tensor, as the TFLite converter
    applies it: one scale for the tensor, or one per channel.'''
    if per_channel:
        axes = tuple(i for i in range(kernel.ndim) if i != channel_axis % kernel.ndim)
        limit = np.max(np.abs(kernel), axis=axes, keepdims=True)
    else:
        limit = np.max(np.abs(kernel))
    scale = np.where(limit > 0, limit / 127.0, 1.0)
    return np.clip(np.round(kernel / scale), -127, 127) * scale


@dataclass
class BiasCorrectionResult:
    '''Outcome of `correct_bias`: `layers` is the number of layers whose
    bias was changed, `skipped` those left alone (no bias), `max_shift` the
    largest absolute bias change, and `restore()` puts the biases back.
    '''
    layers: int
    skipped: int
    max_shift: float
    _saved: list = field(default_factory=list, repr=False)

    def restore(self):
        '''Put back the biases the model had before the correction.'''
        for layer, weights in self._saved:
            layer.set_weights(weights)
        self._saved = []


def _correct_bias_in(model, inputs, per_channel, saved, counters, batch_size):
    for node in _nodes(model):
        layer = node.layer
        node_inputs = tf.nest.flatten(node.keras_inputs)
        if isinstance(layer, tf.keras.Model):
            if len(node_inputs) == len(layer.inputs):
                feeder = tf.keras.Model(model.inputs, node_inputs if len(node_inputs) > 1 else node_inputs[0])
                _correct_bias_in(layer, feeder.predict(inputs, batch_size=batch_size, verbose=0),
                                 per_channel, saved, counters, batch_size)
            continue
        if type(layer) not in (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Dense):
            continue
        if not layer.use_bias:
            counters['skipped'] += 1
            continue
        weights = layer.get_weights()
        kernel = weights[0]
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            quantized = _quantized_weights(kernel, per_channel, channel_axis=2)
        elif isinstance(layer, tf.keras.layers.Dense):
            quantized = _quantized_weights(kernel, False, channel_axis=-1)
        else:
            quantized = _quantized_weights(kernel, per_channel, channel_axis=-1)

        # the mean output of the layer for the weight error alone: conv(x, W_q - W)
        config = layer.get_config()
        config.update(use_bias=False, activation='linear', name=layer.name + '_weight_error')
        probe = type(layer).from_config(config)
        feeder = tf.keras.Model(model.inputs, node_inputs[0])
        activations = feeder.predict(inputs, batch_size=batch_size, verbose=0)
        probe.build(activations.shape)
        probe.set_weights([quantized - kernel])
        totals, count = 0.0, 0
        for start in range(0, len(activations), batch_size):
            error = probe(activations[start:start + batch_size]).numpy()
            error = error.reshape(-1, error.shape[-1])
            totals = totals + error.sum(axis=0)
            count += error.shape[0]
        shift = totals / max(count, 1)

        saved.append((layer, weights))
        layer.set_weights([kernel, weights[1] - shift.astype(weights[1].dtype)] + weights[2:])
        counters['layers'] += 1
        counters['max_shift'] = max(counters['max_shift'], float(np.max(np.abs(shift))))


def correct_bias(model, inputs, per_channel=False, batch_size=8):
    '''
    Bias correction for post-training quantization (Nagel et al., 2019,
    "Data-Free Quantization Through Weight Equalization and Bias
    Correction", in its empirical form).

    Rounding weights to 8 bits shifts the mean output of a layer whenever
    its inputs have a non-zero mean, which is always the case after a ReLU.
    The shift is `E[conv(x, W_q - W)]` per output channel. This function
    measures it on calibration data and subtracts it from the bias, so the
    quantized layer has the same mean output as the float one. It matters
    most where the weight rounding is coarse: per-tensor quantization.

    Works in place. Run it on a model whose batch norms are folded
    (`fold_batch_norms`), after `equalize_cross_layer`, with the weight
    granularity of the target.

    Input
    ----------
        model: a functional `tf.keras.Model` without QAT wrappers. Layers
               without a bias are skipped (fold the batch norms first).
        inputs: calibration inputs, a float array `(n, H, W, C)` normalized
                the way the model expects. A few dozen images are enough.
        per_channel: weight quantization granularity of the target.
        batch_size: batch size of the forward passes.

    Output
    ----------
        A `BiasCorrectionResult`; call its `restore()` to undo the change.
        The activations are those of the float model, so quantization
        errors that build up along the network are not compensated.
    '''
    if has_qat_layers(model):
        raise ValueError('correct_bias needs a model without QAT wrappers: call strip_qat first.')
    inputs = np.asarray(inputs, dtype=np.float32)
    saved, counters = [], dict(layers=0, skipped=0, max_shift=0.0)
    _correct_bias_in(model, inputs, per_channel, saved, counters, batch_size)
    return BiasCorrectionResult(counters['layers'], counters['skipped'], counters['max_shift'], saved)


def quantization_debug_report(model, representative_dataset, profile='default', per_channel=None, top=None):
    '''
    Rank the tensors of a model by the error that quantization introduces
    in them, with `tf.lite.experimental.QuantizationDebugger`.

    Each operator of the quantized model is compared with its float
    counterpart on the calibration data, one operator at a time (the error
    of the layers before it is not included). The usual reading: sort by
    `rmse_over_scale`; values near 0.29 (the error of ideal rounding,
    `1 / sqrt(12)`) are healthy, values far above it point at a layer whose
    range or weights are a problem.

    Input
    ----------
        model: a `tf.keras.Model` (pass a stripped, fused model for an
               accelerator profile, as for `convert_to_tflite`).
        representative_dataset: the calibration generator callable.
        profile: deployment profile; it sets per-channel or per-tensor
                 weights.
        per_channel: override the profile's granularity.
        top: keep only this many entries.

    Output
    ----------
        A list of dicts sorted by decreasing `rmse_over_scale`, each with
        `op_name`, `tensor_name`, `scale`, `rmse`, `rmse_over_scale`,
        `max_abs_error`, `mean_error` and `num_elements`.
    '''
    import csv
    import io

    converter = _configured_converter(model, representative_dataset, profile, per_channel=per_channel,
                                      io_dtype='float32', static_batch=0)
    debugger = tf.lite.experimental.QuantizationDebugger(converter=converter, debug_dataset=representative_dataset)
    debugger.run()
    buffer = io.StringIO()
    debugger.layer_statistics_dump(buffer)
    rows = []
    for row in csv.DictReader(io.StringIO(buffer.getvalue())):
        scale = float(row['scale'])
        rmse = float(row['mean_squared_error']) ** 0.5
        rows.append({
            'op_name': row['op_name'],
            'tensor_name': row['tensor_name'],
            'scale': scale,
            'rmse': rmse,
            'rmse_over_scale': rmse / scale if scale > 0 else float('inf'),
            'max_abs_error': float(row['max_abs_error']),
            'mean_error': float(row['mean_error']),
            'num_elements': int(float(row['num_elements'])),
        })
    rows.sort(key=lambda r: r['rmse_over_scale'], reverse=True)
    return rows[:top] if top else rows


# ========================= #
# Whole-model quantization-aware training

def flatten_model(model):
    '''
    Inline every nested sub-model, so that the result is one flat graph of
    layers.

    The builders of this library keep the backbone as a nested sub-model.
    `tensorflow_model_optimization` cannot wrap layers inside a nested
    model, so whole-model quantization-aware training starts from the flat
    form.

    Input
    ----------
        model: a functional `tf.keras.Model`.

    Output
    ----------
        A new functional model with a dynamic batch size that shares every
        layer with `model` and computes the same function.
    '''
    new_inputs = [tf.keras.Input(shape=tensor.shape[1:], dtype=tensor.dtype, name=tensor._keras_history.layer.name)
                  for tensor in model.inputs]

    def run(graph, inputs):
        usage, tensor_dict = graph._tensor_usage_count, {}
        for tensor, new in zip(graph.inputs, inputs):
            tensor_dict[str(id(tensor))] = [new] * usage[str(id(tensor))]
        for node in _nodes(graph):
            if any(t_id not in tensor_dict for t_id in node.flat_input_ids):
                continue
            args, kwargs = node.map_arguments(tensor_dict)
            if isinstance(node.layer, tf.keras.Model):
                outputs = run(node.layer, tf.nest.flatten(args))
            else:
                outputs = node.layer(*args, **kwargs)
            for t_id, tensor in zip(node.flat_output_ids, tf.nest.flatten(outputs)):
                tensor_dict[t_id] = [tensor] * usage[t_id]
        return [tensor_dict[str(id(tensor))].pop() for tensor in graph.outputs]

    outputs = run(model, new_inputs)
    flat = tf.keras.Model(new_inputs if len(new_inputs) > 1 else new_inputs[0],
                          outputs if len(outputs) > 1 else outputs[0], name=model.name)
    return _carry_attributes(model, flat)


def _relu_kind(layer):
    '''`'relu'` or `'relu6'` for a layer that only applies that activation, else None.'''
    if isinstance(layer, tf.keras.layers.Activation):
        name = getattr(layer.activation, '__name__', '')
        return name if name in ('relu', 'relu6') else None
    if isinstance(layer, tf.keras.layers.ReLU) and not layer.negative_slope and not layer.threshold:
        if layer.max_value is None:
            return 'relu'
        if float(layer.max_value) == 6.0:
            return 'relu6'
    return None


def merge_activations(model):
    '''
    Give every convolution or dense layer that is followed only by a ReLU
    or ReLU6 layer that activation as its own, and turn the activation
    layer into an identity.

    The function is unchanged. The point is quantization-aware training:
    `tensorflow_model_optimization` does not fake-quantize the weights of a
    convolution that is followed by a separate activation layer.

    Input
    ----------
        model: a flat functional `tf.keras.Model` (see `flatten_model`).

    Output
    ----------
        A new model; merged layers are new layers with the weights copied,
        every other layer is shared with `model`.
    '''
    usage = model._tensor_usage_count
    calls = collections.Counter(id(node.layer) for node in _nodes(model))
    merges, identities = {}, set()
    for node in _nodes(model):
        kind = _relu_kind(node.layer)
        inputs = tf.nest.flatten(node.keras_inputs)
        if kind is None or len(inputs) != 1 or calls[id(node.layer)] != 1:
            continue
        source = inputs[0]._keras_history.layer
        if type(source) not in (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Dense):
            continue
        if not _is_linear(source.activation) or calls[id(source)] != 1 or usage.get(str(id(inputs[0])), 0) != 1:
            continue
        merges[id(source)] = kind
        identities.add(id(node.layer))

    if not merges:
        return model
    replaced, pending = {}, []

    def substitute(layer):
        if id(layer) in replaced:
            return replaced[id(layer)]
        new = layer
        if id(layer) in merges:
            config = layer.get_config()
            config['activation'] = merges[id(layer)]
            new = type(layer).from_config(config)
            pending.append((new, layer))
        elif id(layer) in identities:
            new = tf.keras.layers.Activation('linear', name=layer.name)
        replaced[id(layer)] = new
        return new

    merged = _replay(model, substitute)
    for new, old in pending:
        new.set_weights(old.get_weights())
    return merged


def untie_shared_layers(model, jitter=0.0):
    '''
    Give every call of a layer with weights that is called more than once
    its own copy, with the same weights.

    The detection heads share their output convolutions across pyramid
    levels. Untied, each level fine-tunes its own copy during
    quantization-aware training; the exported file holds one copy per
    level either way.

    Input
    ----------
        model: a flat functional `tf.keras.Model` (see `flatten_model`).
        jitter: copy `k` (from 0) has its kernel multiplied by
                `1 + k * jitter`. The TFLite converter merges weight
                constants that are exactly equal, which would tie the
                copies again in a file exported before any fine-tuning;
                `prepare_qat` uses 1e-6, far below the 8-bit rounding step
                of about 1/127.

    Output
    ----------
        `model` itself when nothing is shared, else a new model in which the
        copies are named `<layer>_untied<k>`; every other layer is shared
        with `model`.
    '''
    calls = collections.Counter(id(node.layer) for node in _nodes(model))
    shared = {id(layer) for layer in model.layers if calls[id(layer)] > 1 and layer.weights}
    if not shared:
        return model
    counters, pending = collections.Counter(), []

    def substitute(layer):
        if id(layer) not in shared:
            return layer
        config = layer.get_config()
        config['name'] = '{}_untied{}'.format(layer.name, counters[id(layer)])
        counters[id(layer)] += 1
        copy = type(layer).from_config(config)
        pending.append((copy, layer))
        return copy

    untied = _replay(model, substitute)
    for copy, layer in pending:
        weights = layer.get_weights()
        index = int(copy.name.rsplit('_untied', 1)[1])
        if jitter and index:
            weights = [weights[0] * (1.0 + index * jitter)] + weights[1:]
        copy.set_weights(weights)
    return untied


def batch_norms_to_affine(model):
    '''
    Replace every batch norm on a 4-D tensor by the equivalent 1x1
    depthwise convolution with bias: `y = x * scale + shift` per channel,
    with `scale = gamma / sqrt(variance + epsilon)` and
    `shift = beta - mean * scale`.

    Meant for the batch norms that `fold_batch_norms` cannot fold, those
    that do not follow a convolution (the pre-activation blocks of DDRNet
    and PIDNet). `tensorflow_model_optimization` cannot wrap a standalone
    batch norm, so it would stay in floating point in a whole-model QAT
    export; as a depthwise convolution it is quantized and keeps training
    as a per-channel scale and shift. Run `fold_batch_norms` first.

    Input
    ----------
        model: a flat functional `tf.keras.Model` (see `flatten_model`) in
               inference state.

    Output
    ----------
        A new model computing the same function, or `model` itself when it
        has no batch norm on a 4-D tensor.
    '''
    targets = {id(layer) for layer in model.layers
               if isinstance(layer, tf.keras.layers.BatchNormalization) and len(layer.input_shape) == 4
               and layer.axis in ([3], [-1], 3, -1)}
    if not targets:
        return model
    replaced, pending = {}, []

    def substitute(layer):
        if id(layer) in replaced:
            return replaced[id(layer)]
        new = layer
        if id(layer) in targets:
            new = tf.keras.layers.DepthwiseConv2D(1, use_bias=True, name=layer.name + '_affine')
            pending.append((new, layer))
        replaced[id(layer)] = new
        return new

    converted = _replay(model, substitute)
    for new, bn in pending:
        weights = dict(zip([w.name.rsplit('/', 1)[-1].split(':')[0] for w in bn.weights], bn.get_weights()))
        variance, mean = weights['moving_variance'], weights['moving_mean']
        gamma = weights.get('gamma', np.ones_like(mean))
        beta = weights.get('beta', np.zeros_like(mean))
        scale = gamma / np.sqrt(variance + bn.epsilon)
        new.set_weights([scale.reshape(1, 1, -1, 1).astype(np.float32), (beta - mean * scale).astype(np.float32)])
    return converted


_QAT_CONFIG_CLASS = []


def _weighted_layer_quantize_config():
    '''The quantize config used for convolutions and dense layers in
    whole-model QAT (created on first use: it needs the QAT library).'''
    if _QAT_CONFIG_CLASS:
        return _QAT_CONFIG_CLASS[0]
    import tensorflow_model_optimization as tfmot
    quantizers = tfmot.quantization.keras.quantizers

    class WeightedLayerQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
        '''Fake-quantizes the kernel (symmetric 8 bits, per tensor or per
        output channel) and the output of the layer after its own
        activation. Unlike the stock configuration it accepts any built-in
        activation, ReLU6 included, and always quantizes the weights.'''

        def __init__(self, per_channel=True):
            self.per_channel = per_channel

        @staticmethod
        def _attribute(layer):
            return 'depthwise_kernel' if isinstance(layer, tf.keras.layers.DepthwiseConv2D) else 'kernel'

        def get_weights_and_quantizers(self, layer):
            # per channel only where the last kernel axis is the output channel
            per_axis = self.per_channel and type(layer) is tf.keras.layers.Conv2D
            quantizer = quantizers.LastValueQuantizer(num_bits=8, per_axis=per_axis, symmetric=True, narrow_range=True)
            return [(getattr(layer, self._attribute(layer)), quantizer)]

        def get_activations_and_quantizers(self, layer):
            return []

        def set_quantize_weights(self, layer, quantize_weights):
            setattr(layer, self._attribute(layer), quantize_weights[0])

        def set_quantize_activations(self, layer, quantize_activations):
            pass

        def get_output_quantizers(self, layer):
            return [quantizers.MovingAverageQuantizer(num_bits=8, per_axis=False, symmetric=False, narrow_range=False)]

        def get_config(self):
            return {'per_channel': self.per_channel}

    class OutputOnlyQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
        '''Fake-quantizes only the output of a layer without weights that
        the QAT library has no configuration for (tensor products, the
        custom gate layers, `Resizing`), so that the converter can
        quantize the operator instead of leaving it in floating point.'''

        def get_weights_and_quantizers(self, layer):
            return []

        def get_activations_and_quantizers(self, layer):
            return []

        def set_quantize_weights(self, layer, quantize_weights):
            pass

        def set_quantize_activations(self, layer, quantize_activations):
            pass

        def get_output_quantizers(self, layer):
            return [quantizers.MovingAverageQuantizer(num_bits=8, per_axis=False, symmetric=False, narrow_range=False)]

        def get_config(self):
            return {}

    class FixedRangeOutputQuantizeConfig(OutputOnlyQuantizeConfig):
        '''For a layer without weights whose output range is fixed by its
        last operation, such as a gate ending in a sigmoid. A learned
        range would differ slightly from the scale the converter imposes
        on that operation (1/256 for a sigmoid), and the converter then
        inserts a floating-point requantization after it.'''

        def __init__(self, low=0.0, high=255.0 / 256.0):
            self.low, self.high = float(low), float(high)

        def get_output_quantizers(self, layer):
            return [quantizers.FixedQuantizer(num_bits=8, init_min=self.low, init_max=self.high, narrow_range=False)]

        def get_config(self):
            return {'low': self.low, 'high': self.high}

    _QAT_CONFIG_CLASS.extend([WeightedLayerQuantizeConfig, OutputOnlyQuantizeConfig, FixedRangeOutputQuantizeConfig])
    return WeightedLayerQuantizeConfig


def _output_only_quantize_config():
    _weighted_layer_quantize_config()
    return _QAT_CONFIG_CLASS[1]


def _fixed_range_quantize_config():
    _weighted_layer_quantize_config()
    return _QAT_CONFIG_CLASS[2]


def qat_scope():
    '''
    The custom-object scope needed to load a saved model that was prepared
    by `prepare_qat`::

        with deploy_utils.qat_scope():
            model = tf.keras.models.load_model('qat_model.h5')
    '''
    import tensorflow_model_optimization as tfmot
    return tfmot.quantization.keras.quantize_scope(
        {'WeightedLayerQuantizeConfig': _weighted_layer_quantize_config(),
         'OutputOnlyQuantizeConfig': _output_only_quantize_config(),
         'FixedRangeOutputQuantizeConfig': _fixed_range_quantize_config()})


def _layer_output_ranges(model, inputs, batch_size):
    '''`{layer name: (min, max)}` of every layer output over `inputs`.'''
    names, tensors = [], []
    for node in _nodes(model):
        names.append(node.layer.name)
        tensors.append(tf.nest.flatten(node.outputs)[0])
    probe = tf.keras.Model(model.inputs, tensors)
    ranges = {}
    for start in range(0, len(inputs), batch_size):
        outputs = probe(inputs[start:start + batch_size], training=False)
        for name, output in zip(names, outputs):
            low, high = float(tf.reduce_min(output)), float(tf.reduce_max(output))
            previous = ranges.get(name, (low, high))
            ranges[name] = (min(previous[0], low), max(previous[1], high))
    return ranges


def calibrate_qat_ranges(qat_model, float_model, inputs, batch_size=4, min_width=1e-3):
    '''
    Set the activation ranges of a model prepared by `prepare_qat` from
    data, the way post-training calibration does.

    The fake-quantization wrappers start with a range of -6 to 6 and move
    it slowly while training. Setting it from the float model first means
    the prepared model starts from the post-training result instead of
    from an arbitrary range. `prepare_qat` calls this; call it again after
    changing the weights without training.

    Input
    ----------
        qat_model: the model returned by `prepare_qat`.
        float_model: its float counterpart (`qat_model.float_model`).
        inputs: float array of calibration inputs, normalized as the model
                expects.
        batch_size: batch size of the forward passes.
        min_width: smallest range given to a tensor. A tensor that is
                constant on the calibration data (a dead activation, common
                in untrained models) would otherwise get a zero or
                near-zero scale, which the LiteRT runtime can refuse.

    Output
    ----------
        The number of activation ranges that were set.
    '''
    inputs = np.asarray(inputs, dtype=np.float32)
    # one pass in training mode records the weight ranges (they follow the weights)
    qat_model(inputs[:batch_size], training=True)
    ranges = {}
    for name, (low, high) in _layer_output_ranges(float_model, inputs, batch_size).items():
        low, high = min(low, 0.0), max(high, 0.0)
        if high - low < min_width:
            high = low + min_width
        ranges[name] = (low, high)
    count = 0
    for layer in qat_model.layers:
        base = layer.name[len('quant_'):] if layer.name.startswith('quant_') else None
        for variable in layer.weights:
            tail = variable.name.rsplit('/', 1)[-1]
            if 'kernel' in tail:
                continue
            if tail == 'quantize_layer_min:0':
                variable.assign(min(float(inputs.min()), 0.0))
            elif tail == 'quantize_layer_max:0':
                variable.assign(max(float(inputs.max()), 0.0))
            elif base in ranges and tail.endswith('min:0'):
                variable.assign(ranges[base][0])
                count += 1
            elif base in ranges and tail.endswith('max:0'):
                variable.assign(ranges[base][1])
    return count


def prepare_qat(model, calibration_inputs, per_channel=True, equalize=None, batch_size=4):
    '''
    Turn a trained model into one that can be fine-tuned with
    quantization-aware training of the **whole network**, backbone
    included, on the graph that will be deployed.

    The builders of this library wrap only the head for QAT, and for
    accelerator profiles those wrappers are stripped at export, so their
    effect is small (see the deployment guide). Post-training quantization
    is enough for backbones that quantize well. For those that do not
    (fused reparameterized backbones, ReLU6 networks per tensor), the
    remedy is to fine-tune every weight against the quantization it will
    meet. This function prepares that:

    1. fuse reparameterizable backbones, strip the head's wrappers;
    2. fold batch norms into their convolutions;
    3. inline the nested backbone, give every layer that is called more
       than once (the output convolutions the detection heads share across
       pyramid levels) one copy per call (`untie_shared_layers`), turn the
       remaining batch norms into 1x1 depthwise convolutions
       (`batch_norms_to_affine`), and merge ReLU / ReLU6 layers into the
       convolutions before them. A shared layer with QAT wrappers is
       exported with per-channel weights whatever the converter is told;
       untied, each pyramid level fine-tunes its own copy;
    4. for per-tensor targets, equalize consecutive layers;
    5. wrap every layer: convolutions and dense layers get their kernel
       and output fake-quantized, per channel or per tensor; other layers
       get the library's default configuration, or an output quantizer
       when they have none (tensor products, the custom gate layers);
    6. set the activation ranges from `calibration_inputs`.

    The result starts at about the accuracy of the post-training export.
    Fine-tune it for a few epochs with a small learning rate (a tenth to a
    hundredth of the training rate), then convert it with its wrappers:
    `Quantizer` keeps them for a prepared model, and
    `convert_to_tflite(prepared, ...)` works directly. Its simulated
    predictions match those of the exported file closely, which is what
    makes the fine-tuning meaningful.

    Input
    ----------
        model: a trained functional `tf.keras.Model` of this library (QAT
               wrappers on the head or not).
        calibration_inputs: float array `(n, H, W, C)` of normalized
               inputs; a few dozen images.
        per_channel: weight quantization granularity of the target:
               `True` for one scale per output channel (ARTPEC-9, CPU),
               `False` for one per tensor (ARTPEC-8). Depthwise and dense
               kernels are always quantized per tensor during training.
        equalize: apply cross-layer equalization before wrapping. `None`
               does it for per-tensor targets.
        batch_size: batch size of the calibration passes.

    Output
    ----------
        A new model with independent weights (the input model is not
        modified) and a dynamic batch size. Attributes: `float_model`, the
        float network it was wrapped from, with identical weights at this
        point (usable as a distillation teacher); `preprocessing`;
        `qat_per_channel`, the granularity it was prepared for;
        `unwrapped_layers`, the names of layers that could not be wrapped
        (layers with weights the QAT library does not know, batch norms on
        2-D tensors, and raw `tf` operations such as `x * 2.0`); the
        converter quantizes the latter post-training when it can, and the
        lint reports any operator left in floating point.

    To keep fine-tuned weights, `save_weights` and later `load_weights`
    into a model prepared the same way from the same trained model. To
    load a whole saved model, wrap the call in `qat_scope()`.
    '''
    import tensorflow_model_optimization as tfmot
    from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import (
        default_8bit_quantize_registry, default_8bit_quantize_scheme)

    calibration_inputs = np.asarray(calibration_inputs, dtype=np.float32)
    plain = fold_batch_norms(strip_qat(reparameterize_model(model)))
    plain = flatten_model(plain)
    plain = merge_activations(batch_norms_to_affine(untie_shared_layers(plain, jitter=1e-6)))

    # an independent float copy: nothing below may touch the caller's weights
    try:
        float_model = tf.keras.models.clone_model(plain)
    except Exception as error:
        raise ValueError('prepare_qat cannot copy this model (it contains operations Keras cannot rebuild from '
                         'their configuration): {}'.format(error))
    float_model.set_weights(plain.get_weights())
    if (not per_channel) if equalize is None else equalize:
        equalize_cross_layer(float_model)

    registry = default_8bit_quantize_registry.Default8BitQuantizeRegistry()
    config_class = _weighted_layer_quantize_config()
    output_only = _output_only_quantize_config()
    weighted = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Dense)
    unwrapped = []

    def annotate(layer):
        copy = type(layer).from_config(layer.get_config())
        if isinstance(layer, tf.keras.layers.InputLayer):
            return copy
        if type(layer) in weighted:
            return tfmot.quantization.keras.quantize_annotate_layer(copy, quantize_config=config_class(per_channel))
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            unwrapped.append(layer.name)
            return copy
        if registry.supports(layer):
            return tfmot.quantization.keras.quantize_annotate_layer(copy)
        if type(layer).__name__ in ('TFOpLambda', 'SlicingOpLambda'):
            # raw `tf` operations: the wrapper cannot infer their output shape
            unwrapped.append(layer.name)
            return copy
        if not layer.weights:
            # a layer can declare the fixed range of its output (`qat_output_range`),
            # as the sigmoid gates of PIDNet do
            fixed = getattr(layer, 'qat_output_range', None)
            config = output_only() if fixed is None else _fixed_range_quantize_config()(*fixed)
            return tfmot.quantization.keras.quantize_annotate_layer(copy, quantize_config=config)
        unwrapped.append(layer.name)
        return copy

    annotated = tf.keras.models.clone_model(float_model, clone_function=annotate)
    annotated.set_weights(float_model.get_weights())
    with qat_scope():
        prepared = tfmot.quantization.keras.quantize_apply(
            annotated, scheme=default_8bit_quantize_scheme.Default8BitQuantizeScheme(disable_per_axis=not per_channel))

    calibrate_qat_ranges(prepared, float_model, calibration_inputs, batch_size=batch_size)
    _carry_attributes(model, prepared)
    if hasattr(model, 'preprocessing'):
        float_model.preprocessing = model.preprocessing
    # plain attributes: Keras would otherwise track the float model as a
    # sub-layer of the prepared one and list its weights as trainable
    object.__setattr__(prepared, 'float_model', float_model)
    object.__setattr__(prepared, 'unwrapped_layers', tuple(unwrapped))
    object.__setattr__(prepared, 'qat_per_channel', bool(per_channel))
    prepared._sak_qat_prepared = True
    prepared._sak_reparameterized = True
    return prepared


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


def detection_export_spec(model, tflite_model=None):
    '''
    Describe the exported file of a base detector for the reference decoder
    (`swiss_army_keras.detection_utils.decode_detections`): which output
    of the file is which prediction of which pyramid level, and how to
    turn its integers into real values.

    The converter does not keep the output order of the Keras model, so
    the application cannot rely on it; this function matches the outputs
    by name.

    Input
    ----------
        model: the Keras model the file was converted from: a base
               detector (`fcos_lite_detection_base`,
               `gfl_lite_detection_base`) or what the export steps made of
               it (`Quantizer.export_model`, a model from `prepare_qat`).
        tflite_model: the flatbuffer as `bytes`, or the path of the file.
               `None` describes the outputs of the Keras model itself
               (float, in its own order), to check the decoder against
               `model.predict` on the workstation.

    Output
    ----------
        A dict that serializes to JSON: `kind` (`'fcos'` or `'gfl'`),
        `n_classes`, `strides`, `reg_max` (GFL), `input_size` `[height,
        width]`, and `outputs`, one entry per output in file order with
        `position`, `role` (`'cls'`, `'reg'` or `'centerness'`), `level`,
        `shape`, `dtype`, `scale` and `zero_point`.
    '''
    detection = getattr(model, 'detection', None)
    if not detection:
        raise ValueError('The model is not a base detector of this library (it has no `detection` attribute). '
                         'For a full detector, export `model.base_model`.')
    roles = list(detection['roles'])
    if len(model.outputs) != len(roles) * len(detection['strides']):
        raise ValueError('The model has {} outputs; a {} detector with {} levels has {}.'.format(
            len(model.outputs), detection['kind'], len(detection['strides']), len(roles) * len(detection['strides'])))
    by_name = {name: (roles[index % len(roles)], index // len(roles))
               for index, name in enumerate(model.output_names)}

    if tflite_model is None:
        described = [{'position': position, 'name': name, 'shape': tuple(int(d or 1) for d in tensor.shape),
                      'dtype': 'float32', 'scale': 0.0, 'zero_point': 0}
                     for position, (name, tensor) in enumerate(zip(model.output_names, model.outputs))]
    else:
        described = tflite_outputs(tflite_model)
    outputs = []
    for entry in described:
        if entry['name'] not in by_name:
            raise ValueError("Output '{}' of the file is not an output of the model ({}); pass the Keras model the "
                             'file was converted from.'.format(entry['name'], ', '.join(model.output_names)))
        role, level = by_name[entry['name']]
        outputs.append({'position': entry['position'], 'role': role, 'level': level, 'name': entry['name'],
                        'shape': list(entry['shape']), 'dtype': entry['dtype'], 'scale': entry['scale'],
                        'zero_point': entry['zero_point']})
    if len({(o['role'], o['level']) for o in outputs}) != len(by_name):
        raise ValueError('The file has {} outputs; the model has {}.'.format(len(outputs), len(by_name)))

    spec = {key: value for key, value in detection.items() if key != 'roles'}
    spec['strides'] = [int(v) for v in spec['strides']]
    spec['input_size'] = [int(model.inputs[0].shape[1]), int(model.inputs[0].shape[2])]
    spec['outputs'] = outputs
    return spec


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

    # degenerate activation ranges
    produced = set(inputs)
    for op in ops:
        produced.update(i for i in op['outputs'] if i >= 0)
    zero_scale = [tensors[i]['name'] for i in sorted(produced)
                  if tensors[i]['dtype'] not in _FLOAT_DTYPES
                  and len(tensors[i]['quantization_parameters']['scales'])
                  and not np.all(tensors[i]['quantization_parameters']['scales'] > 0)]
    if zero_scale:
        report.errors.append(
            '{} tensor(s) have a zero quantization scale, e.g. {!r}: they were constant during calibration '
            '(an untrained model, a dead activation, or unrepresentative calibration data). The LiteRT runtime '
            'can abort when it loads such a file'.format(len(zero_scale), zero_scale[0]))
    tiny_scale = [tensors[i]['name'] for i in sorted(produced)
                  if tensors[i]['dtype'] not in _FLOAT_DTYPES
                  and len(tensors[i]['quantization_parameters']['scales'])
                  and np.all(tensors[i]['quantization_parameters']['scales'] > 0)
                  and np.any(tensors[i]['quantization_parameters']['scales'] < 1e-7)]
    if tiny_scale:
        report.warnings.append(
            '{} tensor(s) have a near-zero quantization scale, e.g. {!r}: they were almost constant during '
            'calibration (a saturated gate or a dead activation, common in untrained models). Some runtimes '
            'refuse such files'.format(len(tiny_scale), tiny_scale[0]))

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
