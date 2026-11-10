"""Plain convolutional backbones for accelerators (`_backbone_plain.py`):
ResNet-18/34, STDC1/2, QARepVGG and the ReLU6 cross-stage-partial family,
plus the MobileNetV3Small registry entry.

No reference weights exist for them in this repository, so the checks are
structural: feature strides, parameter counts where an authoritative count
is known (torchvision's ResNets), exactness of the QARepVGG fusion, and the
absence of the operators the families were added to avoid.
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras import deploy_utils
from swiss_army_keras._backbone_plain import (
    CSP_N,
    QARepVGG_A0,
    ResNet18,
    ResNet34,
    STDC1,
    reparameterize_qarepvgg,
)
from swiss_army_keras._backbone_zoo import BACKBONE_REGISTRY, backbone_num_levels, backbone_zoo

SIZE = 64
PLAIN_KEYS = ["ResNet18", "ResNet34", "STDC1", "STDC2", "QARepVGG-A0", "QARepVGG-A1", "QARepVGG-B0",
              "CSP-N", "CSP-S", "CSP-M", "MobileNetV3Small"]


@pytest.mark.parametrize("key", PLAIN_KEYS)
def test_registered_with_five_levels_at_strides_2_to_32(key):
    assert key in BACKBONE_REGISTRY and backbone_num_levels(key) == 5
    inp = tf.keras.Input((SIZE, SIZE, 3))
    backbone = backbone_zoo(key, None, inp, 5, False, False)
    assert [SIZE // int(t.shape[1]) for t in backbone.outputs] == [2, 4, 8, 16, 32]
    assert callable(backbone.preprocessing)


@pytest.mark.parametrize("constructor, expected", [(ResNet18, 11_176_512), (ResNet34, 21_284_672)],
                         ids=["ResNet18", "ResNet34"])
def test_resnet_parameter_counts_match_torchvision(constructor, expected):
    """torchvision's totals (11,689,512 and 21,797,672) minus the 513,000
    parameters of the 1000-class layer.
    """
    model = constructor(input_shape=(SIZE, SIZE, 3))
    assert int(sum(np.prod(w.shape) for w in model.trainable_weights)) == expected


def _only_these_layers(model, allowed):
    return {type(layer).__name__ for layer in model.layers} <= set(allowed)


def test_families_use_plain_operators_only():
    common = {"InputLayer", "Conv2D", "BatchNormalization", "Activation", "Add", "Concatenate"}
    assert _only_these_layers(ResNet18(input_shape=(SIZE, SIZE, 3)), common | {"MaxPooling2D"})
    assert _only_these_layers(STDC1(input_shape=(SIZE, SIZE, 3)), common | {"DepthwiseConv2D", "AveragePooling2D"})
    assert _only_these_layers(CSP_N(input_shape=(SIZE, SIZE, 3)), common | {"ReLU", "MaxPooling2D"})
    assert _only_these_layers(QARepVGG_A0(input_shape=(SIZE, SIZE, 3)), common | {"Cropping2D"})


def test_csp_channels_are_multiples_of_16_and_activation_is_relu6():
    model = CSP_N(input_shape=(SIZE, SIZE, 3))
    convs = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    assert all(layer.filters % 16 == 0 for layer in convs)
    relus = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.ReLU)]
    assert relus and all(layer.max_value == 6.0 for layer in relus)


def test_torch_padding_option_and_missing_weights_message():
    aligned = ResNet18(input_shape=(SIZE, SIZE, 3), torch_padding=True)
    assert any(isinstance(layer, tf.keras.layers.ZeroPadding2D) for layer in aligned.layers)
    default = ResNet18(input_shape=(SIZE, SIZE, 3))
    assert not any(isinstance(layer, tf.keras.layers.ZeroPadding2D) for layer in default.layers)
    assert aligned.output_shape == default.output_shape
    with pytest.raises(NotImplementedError, match="no pretrained weights"):
        ResNet18(input_shape=(SIZE, SIZE, 3), weights="imagenet")


def _randomize_batch_norms(model, seed=0):
    rng = np.random.default_rng(seed)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            shape = layer.get_weights()[0].shape
            layer.set_weights([rng.uniform(0.8, 1.2, shape), rng.normal(0, 0.1, shape),
                               rng.normal(0, 0.1, shape), rng.uniform(0.8, 1.2, shape)])


@pytest.mark.parametrize("size", [(64, 64), (96, 80), (100, 76)], ids=str)
def test_qarepvgg_fusion_is_exact_for_even_and_odd_sizes(size):
    """Stride-2 blocks are the delicate part: with Keras 'same' padding the
    1x1 branch must sample the pixels under the centre of the 3x3 windows.
    """
    tf.keras.utils.set_random_seed(0)
    model = QARepVGG_A0(input_shape=(size[0], size[1], 3))
    _randomize_batch_norms(model)
    fused = reparameterize_qarepvgg(model)

    assert len(fused.layers) == 45                       # input, 22 convolutions, 22 ReLUs
    assert not any(isinstance(layer, tf.keras.layers.BatchNormalization) for layer in fused.layers)
    x = np.random.default_rng(1).random((2, size[0], size[1], 3), dtype=np.float32)
    reference, result = model.predict(x, verbose=0), fused.predict(x, verbose=0)
    assert np.abs(reference - result).max() / np.abs(reference).max() < 1e-4


def test_qarepvgg_is_fused_inside_a_full_model_at_export():
    from swiss_army_keras.models import lr_aspp

    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input((SIZE, SIZE, 3))
    model = lr_aspp(inp, n_labels=3, gate=False, backbone="QARepVGG-A0", weights=None, freeze_backbone=False)
    _randomize_batch_norms(model.get_layer("QARepVGG-A0_backbone"))
    fused = deploy_utils.reparameterize_model(model)

    backbone = fused.get_layer("QARepVGG-A0_backbone")
    assert not any(isinstance(layer, tf.keras.layers.Add) for layer in backbone.layers)
    x = np.random.default_rng(1).random((2, SIZE, SIZE, 3), dtype=np.float32)
    plain, plain_fused = deploy_utils.strip_qat(model), deploy_utils.strip_qat(fused)
    reference, result = plain.predict(x, verbose=0), plain_fused.predict(x, verbose=0)
    assert np.abs(reference - result).max() / np.abs(reference).max() < 1e-3


def test_reparameterize_qarepvgg_rejects_other_models():
    with pytest.raises(ValueError, match="_qarepvgg_config"):
        reparameterize_qarepvgg(ResNet18(input_shape=(SIZE, SIZE, 3)))
