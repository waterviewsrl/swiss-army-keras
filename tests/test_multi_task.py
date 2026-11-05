"""`multi_task`: several heads on one shared backbone.

Every other builder creates its own backbone, so two tasks cost two
backbone passes. These tests check that `multi_task` builds exactly one,
names each output after its head, and stays compatible with the helpers
that work on single-head models (embedding output, activation drop).
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras import deploy_utils
from swiss_army_keras._model_multi_task import HEAD_TYPES
from swiss_army_keras.layer_utils import add_embedding_output, qat_annotate_layer
from swiss_army_keras.models import multi_task

INPUT_SIZE = 96


def _heads():
    return {
        'seg': dict(type='lr_aspp', n_labels=4, gate=False),
        'depth': dict(type='fastdepth'),
        'inverse_depth': dict(type='lr_aspp_depth', gate=False),
        'cls': dict(type='classifier', n_classes=3, size=32, activation='relu'),
        'quality': dict(type='iqa', pooling='avg'),
    }


@pytest.fixture(scope='module')
def model():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    return multi_task(inp, _heads(), backbone='MobileNetV4-Conv-S', weights=None, freeze_backbone=False)


def test_one_backbone_and_one_named_output_per_head(model):
    assert [l.name for l in model.layers if l.name.endswith('_backbone')] == ['MobileNetV4-Conv-S_backbone']
    assert model.output_names == list(_heads())
    assert model.head_names == list(_heads())
    assert callable(model.preprocessing)

    outputs = model.predict(np.random.rand(2, INPUT_SIZE, INPUT_SIZE, 3).astype('float32'), verbose=0)
    assert [o.shape for o in outputs] == [
        (2, INPUT_SIZE, INPUT_SIZE, 4), (2, INPUT_SIZE, INPUT_SIZE, 1), (2, INPUT_SIZE, INPUT_SIZE, 1), (2, 3), (2, 1)]
    for index in (1, 2, 4):  # sigmoid-bounded heads
        assert outputs[index].min() >= 0.0 and outputs[index].max() <= 1.0
    np.testing.assert_allclose(outputs[3].sum(axis=-1), 1.0, atol=1e-3)


def test_losses_can_be_given_by_head_name(model):
    model.compile(optimizer='adam', loss={
        'seg': 'mse', 'depth': 'mse', 'inverse_depth': 'mse', 'cls': 'categorical_crossentropy', 'quality': 'mse'})
    x = np.random.rand(2, INPUT_SIZE, INPUT_SIZE, 3).astype('float32')
    targets = {name: np.zeros((2,) + tuple(shape[1:]), dtype='float32')
               for name, shape in zip(model.output_names, model.output_shape)}
    losses = model.evaluate(x, targets, verbose=0, return_dict=True)
    assert {f'{name}_loss' for name in _heads()} <= set(losses)


def test_shares_fewer_weights_than_separate_models(model):
    backbone = model.get_layer('MobileNetV4-Conv-S_backbone')
    assert model.count_params() < 2 * backbone.count_params()


def test_embedding_and_activation_drop_work_on_a_multi_task_model(model):
    wrapped = add_embedding_output(model, output_index=3)
    assert tuple(wrapped.outputs[-1].shape[1:]) == (32,)
    assert wrapped.head_names == list(_heads())

    plain = deploy_utils.drop_output_activations(deploy_utils.strip_qat(model))
    assert plain.dropped_output_activations == [None, 'sigmoid', 'sigmoid', 'softmax', 'sigmoid']
    assert plain.output_names[:4] == ['seg', 'depth', 'inverse_depth', 'cls']  # head names survive


def test_classifier_head_can_return_logits_and_use_another_level():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = multi_task(inp, {'cls': dict(type='classifier', n_classes=3, size=16, logits=True, feature_level=4)},
                       backbone='MobileNetV2', weights=None)
    out = model.predict(np.random.rand(2, INPUT_SIZE, INPUT_SIZE, 3).astype('float32'), verbose=0)
    assert out.shape == (2, 3)
    assert model.get_layer('quant_cls_pool').input_shape[1] == INPUT_SIZE // 16


@pytest.mark.parametrize('backbone', ['RepViT-M0.9', 'StrideFormer-Tiny'])
def test_builds_on_backbones_with_fewer_feature_levels(backbone):
    size = 128  # StrideFormer needs an input that stays even down to stride 32
    inp = tf.keras.Input((size, size, 3))
    model = multi_task(inp, {'seg': dict(type='lr_aspp', n_labels=2), 'depth': dict(type='fastdepth')},
                       backbone=backbone, weights=None)
    assert [tuple(s[1:]) for s in model.output_shape] == [(size, size, 2), (size, size, 1)]


def test_a_callable_is_accepted_as_a_head_type():
    def edge_head(features, input_tensor, name, channels=1):
        x = qat_annotate_layer(tf.keras.layers.Conv2D(channels, 1, name=f'{name}_conv'))(features[0])
        return x

    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = multi_task(inp, {'edges': dict(type=edge_head, channels=2)}, backbone='MobileNetV2', weights=None)
    assert model.output_shape == (None, INPUT_SIZE // 2, INPUT_SIZE // 2, 2)
    assert model.output_names == ['edges']


def test_bad_specifications_are_rejected():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    with pytest.raises(ValueError, match='at least one head'):
        multi_task(inp, {}, backbone='MobileNetV2', weights=None)
    with pytest.raises(ValueError, match="no 'type'"):
        multi_task(inp, {'seg': dict(n_labels=2)}, backbone='MobileNetV2', weights=None)
    with pytest.raises(ValueError, match='Unknown head type'):
        multi_task(inp, {'seg': dict(type='unet', n_labels=2)}, backbone='MobileNetV2', weights=None)
    with pytest.raises(ValueError, match='feature_level'):
        multi_task(inp, {'q': dict(type='iqa', feature_level=9)}, backbone='MobileNetV2', weights=None)
    assert set(HEAD_TYPES) == {'lr_aspp', 'lr_aspp_depth', 'fastdepth', 'classifier', 'iqa'}
