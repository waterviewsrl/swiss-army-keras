"""Golden-output verification for `swiss_army_keras.reparam_utils`.

The real test of any reparameterization utility isn't "does it run" - it's
"does the fused single conv produce numerically identical output to the
original multi-branch block". Both tests below build an actual small
RepVGG-style multi-branch block (3x3 conv+BN branch + 1x1 conv+BN branch +
identity+BN branch, summed, then an activation - the pattern MobileOne §2.1/
FastViT's RepMixer §2.3/RepViT §2.4 all share), fuse it via
`reparam_utils`, and diff the two models' outputs on the same random input.
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras.reparam_utils import (
    fuse_conv_bn,
    fuse_conv_bn_layers,
    identity_kernel,
    pad_kernel_to,
    scale_kernel_bias,
    sum_branches,
)


def _build_multibranch_block(channels=4, size=8, depthwise=False):
    inp = tf.keras.Input((size, size, channels))

    if depthwise:
        conv3x3 = tf.keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)
        conv1x1 = tf.keras.layers.DepthwiseConv2D(1, padding="same", use_bias=False)
    else:
        conv3x3 = tf.keras.layers.Conv2D(channels, 3, padding="same", use_bias=False)
        conv1x1 = tf.keras.layers.Conv2D(channels, 1, padding="same", use_bias=False)

    branch_3x3 = tf.keras.layers.BatchNormalization()(conv3x3(inp))
    branch_1x1 = tf.keras.layers.BatchNormalization()(conv1x1(inp))
    branch_identity = tf.keras.layers.BatchNormalization()(inp)

    summed = tf.keras.layers.Add()([branch_3x3, branch_1x1, branch_identity])
    out = tf.keras.layers.ReLU()(summed)
    model = tf.keras.Model(inp, out)

    # Put BatchNormalization in a non-trivial (not identity) state, as if
    # trained - fusion correctness at gamma=1/beta=0/mean=0/var=1 is a much
    # weaker test since scale reduces to ~1 and offset to ~0.
    rng = np.random.default_rng(0)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            gamma, beta, mean, var = layer.get_weights()
            layer.set_weights([
                rng.uniform(0.5, 2.0, size=gamma.shape).astype("float32"),
                rng.uniform(-1.0, 1.0, size=beta.shape).astype("float32"),
                rng.uniform(-1.0, 1.0, size=mean.shape).astype("float32"),
                rng.uniform(0.5, 2.0, size=var.shape).astype("float32"),
            ])

    return model, conv3x3, conv1x1


def _fuse_and_predict(model, conv3x3, conv1x1, channels, depthwise, x):
    bn3x3, bn1x1, bn_id = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]

    k3x3, b3x3 = fuse_conv_bn_layers(conv3x3, bn3x3)
    k1x1, b1x1 = fuse_conv_bn_layers(conv1x1, bn1x1)
    k1x1_padded = pad_kernel_to(k1x1, target_size=3)

    id_kernel = identity_kernel(channels, kernel_size=3, depthwise=depthwise)
    gamma, beta, mean, var = bn_id.get_weights()
    channel_axis = -2 if depthwise else -1
    k_id, b_id = fuse_conv_bn(id_kernel, None, gamma, beta, mean, var, epsilon=bn_id.epsilon, channel_axis=channel_axis)

    fused_kernel, fused_bias = sum_branches([(k3x3, b3x3), (k1x1_padded, b1x1), (k_id, b_id)])

    inp = tf.keras.Input(model.input_shape[1:])
    if depthwise:
        fused_conv = tf.keras.layers.DepthwiseConv2D(3, padding="same", use_bias=True)
        y = fused_conv(inp)
        fused_conv.set_weights([fused_kernel, fused_bias])
    else:
        fused_conv = tf.keras.layers.Conv2D(channels, 3, padding="same", use_bias=True)
        y = fused_conv(inp)
        fused_conv.set_weights([fused_kernel, fused_bias])
    y = tf.keras.layers.ReLU()(y)
    fused_model = tf.keras.Model(inp, y)

    return fused_model.predict(x, verbose=0)


@pytest.mark.parametrize("depthwise", [False, True])
def test_fused_block_matches_multibranch_output(depthwise):
    channels, size = 4, 8
    model, conv3x3, conv1x1 = _build_multibranch_block(channels, size, depthwise=depthwise)

    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, size, size, channels)).astype("float32")

    y_multibranch = model.predict(x, verbose=0)
    y_fused = _fuse_and_predict(model, conv3x3, conv1x1, channels, depthwise, x)

    np.testing.assert_allclose(y_multibranch, y_fused, rtol=1e-4, atol=1e-4)


def test_pad_kernel_to_centers_and_zero_pads():
    k = np.arange(1 * 1 * 2 * 3, dtype="float32").reshape(1, 1, 2, 3)
    padded = pad_kernel_to(k, target_size=3)
    assert padded.shape == (3, 3, 2, 3)
    np.testing.assert_array_equal(padded[1, 1], k[0, 0])
    assert np.count_nonzero(padded) == np.count_nonzero(k)


def test_pad_kernel_to_rejects_odd_size_difference():
    k = np.zeros((1, 1, 2, 2), dtype="float32")
    with pytest.raises(ValueError):
        pad_kernel_to(k, target_size=2)


def test_identity_kernel_plain_conv_reproduces_input():
    channels, size = 3, 5
    kernel = identity_kernel(channels, kernel_size=3)
    conv = tf.keras.layers.Conv2D(channels, 3, padding="same", use_bias=False)
    conv.build((None, size, size, channels))
    conv.set_weights([kernel])

    x = np.random.default_rng(2).standard_normal((1, size, size, channels)).astype("float32")
    y = conv(x).numpy()
    np.testing.assert_allclose(y, x, rtol=1e-5, atol=1e-5)


def test_identity_kernel_depthwise_reproduces_input():
    channels, size = 3, 5
    kernel = identity_kernel(channels, kernel_size=3, depthwise=True)
    conv = tf.keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)
    conv.build((None, size, size, channels))
    conv.set_weights([kernel])

    x = np.random.default_rng(3).standard_normal((1, size, size, channels)).astype("float32")
    y = conv(x).numpy()
    np.testing.assert_allclose(y, x, rtol=1e-5, atol=1e-5)


def test_identity_kernel_grouped():
    channels, groups = 4, 2
    kernel = identity_kernel(channels, kernel_size=1, groups=groups)
    # Each group should only mix within itself, never across groups.
    group_size = channels // groups
    for g in range(groups):
        block = kernel[0, 0, g * group_size:(g + 1) * group_size, g * group_size:(g + 1) * group_size]
        np.testing.assert_array_equal(block, np.eye(group_size, dtype="float32"))
    off_diagonal = kernel[0, 0, 0:group_size, group_size:]
    np.testing.assert_array_equal(off_diagonal, np.zeros_like(off_diagonal))


def test_fuse_conv_bn_with_none_bias_treated_as_zero():
    kernel = np.ones((3, 3, 2, 2), dtype="float32")
    gamma = np.array([2.0, 3.0], dtype="float32")
    beta = np.array([0.5, -0.5], dtype="float32")
    mean = np.array([0.0, 0.0], dtype="float32")
    var = np.array([1.0, 1.0], dtype="float32")

    fused_kernel, fused_bias = fuse_conv_bn(kernel, None, gamma, beta, mean, var, epsilon=0.0)
    fused_kernel_zero_bias, fused_bias_zero_bias = fuse_conv_bn(
        kernel, np.zeros_like(gamma), gamma, beta, mean, var, epsilon=0.0
    )
    np.testing.assert_array_equal(fused_kernel, fused_kernel_zero_bias)
    np.testing.assert_array_equal(fused_bias, fused_bias_zero_bias)


def test_sum_branches():
    pairs = [
        (np.array([1.0, 2.0]), np.array([0.1])),
        (np.array([3.0, 4.0]), np.array([0.2])),
    ]
    kernel, bias = sum_branches(pairs)
    np.testing.assert_allclose(kernel, [4.0, 6.0])
    np.testing.assert_allclose(bias, [0.3])


def test_scale_kernel_bias_matches_explicit_multiply():
    channels, size = 3, 5
    conv = tf.keras.layers.Conv2D(channels, 3, padding="same", use_bias=True)
    conv.build((None, size, size, channels))
    rng = np.random.default_rng(4)
    kernel = rng.standard_normal((3, 3, channels, channels)).astype("float32")
    bias = rng.standard_normal((channels,)).astype("float32")
    conv.set_weights([kernel, bias])

    scale = rng.uniform(0.1, 3.0, size=(channels,)).astype("float32")
    scaled_kernel, scaled_bias = scale_kernel_bias(kernel, bias, scale, channel_axis=-1)

    scaled_conv = tf.keras.layers.Conv2D(channels, 3, padding="same", use_bias=True)
    scaled_conv.build((None, size, size, channels))
    scaled_conv.set_weights([scaled_kernel, scaled_bias])

    x = rng.standard_normal((1, size, size, channels)).astype("float32")
    y_manual = conv(x).numpy() * scale
    y_scaled_conv = scaled_conv(x).numpy()
    np.testing.assert_allclose(y_manual, y_scaled_conv, rtol=1e-5, atol=1e-5)
