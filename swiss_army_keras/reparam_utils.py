'''Structural reparameterization primitives.

Shared by every backbone in plan.md §2 that trains a multi-branch block and
collapses it into a single `Conv2D`/`DepthwiseConv2D` at inference time
(MobileOne §2.1, FastViT's RepMixer §2.3, RepViT §2.4 — the "RepVGG-style"
reparameterization trick). This module holds only the framework-independent
math (conv+BN fusion, kernel padding, identity-as-kernel, branch summation),
verified once here against a synthetic multi-branch block; the per-block-class
orchestration (walking a specific `MobileOneBlock`/etc. and calling these
functions in the right order) belongs with each backbone's own implementation
once those block classes exist, not in this shared module.

All functions operate on plain `numpy` arrays with Keras's own weight-array
conventions:
    - Conv2D kernel:            (kh, kw, cin, cout)
    - DepthwiseConv2D kernel:   (kh, kw, cin, depth_multiplier)
    - bias:                     (cout,) or (cin * depth_multiplier,)
'''

from __future__ import absolute_import

import numpy as np


def fuse_conv_bn(kernel, bias, gamma, beta, moving_mean, moving_variance, epsilon=1e-3, channel_axis=-1):
    '''Fuse a Conv2D/DepthwiseConv2D immediately followed by a BatchNormalization
    into a single equivalent (kernel, bias) pair.

    BN(conv(x)) == conv'(x) for a single conv' with:
        scale = gamma / sqrt(moving_variance + epsilon)
        kernel' = kernel * scale            (broadcast over `channel_axis`)
        bias'   = (bias - moving_mean) * scale + beta

    Input
    ----------
        kernel: conv kernel. For a regular `Conv2D`, shape
                (kh, kw, cin, cout) — BN normalizes `cout`, the *last* axis
                (`channel_axis=-1`, the default). For a `DepthwiseConv2D`,
                shape (kh, kw, cin, depth_multiplier) — BN normalizes the
                *input* channels (`cin`, axis -2 for depth_multiplier=1),
                since depthwise conv's output-channel count is
                `cin * depth_multiplier`, not the kernel's last axis. Pass
                `channel_axis=-2` explicitly for the depthwise case (or use
                `fuse_conv_bn_layers`, which detects this automatically).
        bias: conv bias, matching `gamma`'s shape, or None (treated as
              zeros — Conv2D layers built with `use_bias=False`, the norm in
              this codebase for conv-then-BN blocks, have no bias to fuse).
        gamma, beta, moving_mean, moving_variance: the BN layer's 4 weight
            arrays, each shape (num_channels,) where num_channels is the
            size of `kernel`'s `channel_axis`.
        epsilon: the BN layer's `epsilon` (must match what the BN layer was
                 actually built/trained with — mismatches here are silent).
        channel_axis: which axis of `kernel` BN's per-channel scale/offset
                      broadcasts over. `-1` for regular `Conv2D`, `-2` for
                      `DepthwiseConv2D` with `depth_multiplier=1` (BN's
                      channel count then equals `cin`). For
                      `DepthwiseConv2D` with `depth_multiplier > 1`, BN's
                      channel count is `cin * depth_multiplier` — the
                      *flattened* `(cin, depth_multiplier)` pair, not either
                      axis alone — pass the tuple `(-2, -1)` for this case;
                      `gamma`/etc. are then reshaped directly onto the
                      kernel's trailing `(cin, depth_multiplier)` shape,
                      which is exactly how Keras already orders a depthwise
                      conv's output channels (input channel `i`'s
                      `depth_multiplier` outputs occupy consecutive
                      positions `i*depth_multiplier .. i*depth_multiplier +
                      depth_multiplier - 1`), so this reshape needs no
                      further permutation.

    Output
    ----------
        (fused_kernel, fused_bias): `fused_kernel` has `kernel`'s shape,
        `fused_bias` has `gamma`'s shape.
    '''
    if bias is None:
        bias = np.zeros_like(gamma)

    scale = gamma / np.sqrt(moving_variance + epsilon)
    broadcast_shape = [1] * kernel.ndim
    if isinstance(channel_axis, tuple):
        for axis in channel_axis:
            broadcast_shape[axis] = kernel.shape[axis]
    else:
        broadcast_shape[channel_axis] = scale.shape[0]
    fused_kernel = kernel * scale.reshape(broadcast_shape)
    fused_bias = (bias - moving_mean) * scale + beta
    return fused_kernel, fused_bias


def fuse_conv_bn_layers(conv_layer, bn_layer):
    '''Convenience wrapper around `fuse_conv_bn` that pulls weights directly
    from a built Keras `Conv2D`/`DepthwiseConv2D` layer and the
    `BatchNormalization` layer that immediately follows it, auto-detecting
    the right `channel_axis` (see `fuse_conv_bn`) from the conv layer type.
    '''
    import tensorflow as tf

    weights = conv_layer.get_weights()
    kernel = weights[0]
    bias = weights[1] if len(weights) > 1 else None
    gamma, beta, moving_mean, moving_variance = bn_layer.get_weights()
    if isinstance(conv_layer, tf.keras.layers.DepthwiseConv2D):
        channel_axis = (-2, -1) if conv_layer.depth_multiplier > 1 else -2
    else:
        channel_axis = -1
    return fuse_conv_bn(
        kernel, bias, gamma, beta, moving_mean, moving_variance,
        epsilon=bn_layer.epsilon, channel_axis=channel_axis,
    )


def pad_kernel_to(kernel, target_size):
    '''Zero-pad a smaller square spatial kernel (e.g. a 1x1 branch) up to a
    larger target spatial size (e.g. 3x3), centered, so it can be summed
    with a same-shaped branch.

    **Only valid if the branch this kernel came from, and the branch it's
    being summed with, were both built with explicit *symmetric* padding
    (`tf.keras.layers.ZeroPadding2D(kernel_size // 2)` + `padding='valid'`)
    — not Keras's `padding='same'`, if `stride > 1` is involved.** This
    function assumes both branches sample the *same* input window, which is
    what symmetric centering gives you; `'same'` padding for a strided conv
    can be asymmetric in a way that depends on kernel size (`total_pad =
    max((out-1)*stride+k-in, 0)`, with any odd remainder dumped on one
    side), so two different kernel sizes at `stride>1` can end up sampling
    *different* windows despite both using `'same'`. At `stride == 1`,
    `'same'` for an odd kernel is always exactly symmetric, so this concern
    doesn't apply there. This was a real bug (not a theoretical one) in
    MobileOne's and FastViT's first implementations in this repo — see
    plan.md §2.1 for the full account, including how a too-small
    golden-output test input can hide it entirely (the network's final
    output can collapse to a single spatial pixel, at which point a
    mid-network sampling-window shift may not change the final comparison
    at all).

    Input
    ----------
        kernel: shape (kh, kw, cin, cout), kh == kw, kh <= target_size.
        target_size: the target (odd) spatial size.

    Output
    ----------
        A (target_size, target_size, cin, cout) kernel, equal to `kernel` at
        the center and zero elsewhere.
    '''
    kh = kernel.shape[0]
    if kh == target_size:
        return kernel
    if (target_size - kh) % 2 != 0:
        raise ValueError(
            f"Cannot center a {kh}x{kh} kernel inside a {target_size}x{target_size} "
            "one - the size difference must be even."
        )
    pad = (target_size - kh) // 2
    return np.pad(kernel, ((pad, pad), (pad, pad), (0, 0), (0, 0)))


def identity_kernel(channels, kernel_size, depthwise=False, groups=1, dtype=np.float32):
    '''Build the conv kernel equivalent to an identity (skip-connection) mapping.

    A plain `y = x` skip connection is itself a valid "branch" in a
    RepVGG-style multi-branch block whenever stride=1 and cin==cout — folding
    it into the branch-sum requires expressing it as a conv kernel first.

    Input
    ----------
        channels: number of channels the identity maps (cin == cout).
        kernel_size: target spatial size (matching the block's other branches).
        depthwise: if True, build a `DepthwiseConv2D`-shaped kernel
                   (kernel_size, kernel_size, channels, 1) instead of a full
                   `Conv2D`-shaped one.
        groups: for a *grouped* (not depthwise) `Conv2D` identity — must
                evenly divide `channels`; each group's block of the kernel
                gets its own identity submatrix. Ignored if `depthwise=True`.

    Output
    ----------
        A (kernel_size, kernel_size, cin, cout_or_1) kernel with a 1 at the
        spatial center for each matching in/out channel pair, 0 elsewhere.
    '''
    center = kernel_size // 2
    if depthwise:
        kernel = np.zeros((kernel_size, kernel_size, channels, 1), dtype=dtype)
        kernel[center, center, :, 0] = 1.0
        return kernel

    if channels % groups != 0:
        raise ValueError("`channels` must be evenly divisible by `groups`")
    group_size = channels // groups
    kernel = np.zeros((kernel_size, kernel_size, channels, channels), dtype=dtype)
    for g in range(groups):
        for i in range(group_size):
            c = g * group_size + i
            kernel[center, center, c, c] = 1.0
    return kernel


def scale_kernel_bias(kernel, bias, scale, channel_axis=-1):
    '''Fold a per-output-channel learnable scale (a "LayerScale"/"gamma"
    vector, as used by FastViT §2.3 — a channel-wise multiply applied after
    a block's linear branch-sum and before its residual add) into a
    (kernel, bias) pair — `scale * (conv(x) + bias) == conv'(x) + bias'`
    with `kernel' = kernel * scale`, `bias' = bias * scale`, exactly the
    same broadcast rule as `fuse_conv_bn`'s scale step (a LayerScale is
    algebraically a degenerate BN with beta=mean=0, var=1).

    Input
    ----------
        kernel, bias: a (kernel, bias) pair, as produced by `fuse_conv_bn`/
                      `sum_branches`.
        scale: the per-channel scale vector, shape (num_channels,) matching
               `kernel`'s `channel_axis` size.
        channel_axis: which axis of `kernel` the scale broadcasts over —
                      same convention as `fuse_conv_bn` (`-1` for `Conv2D`,
                      `-2` for `DepthwiseConv2D`).

    Output
    ----------
        (scaled_kernel, scaled_bias).
    '''
    broadcast_shape = [1] * kernel.ndim
    broadcast_shape[channel_axis] = scale.shape[0]
    return kernel * scale.reshape(broadcast_shape), bias * scale


def sum_branches(kernel_bias_pairs):
    '''Sum a list of same-shaped (kernel, bias) pairs into one fused pair —
    the final step of collapsing a multi-branch block: every branch has
    already been reduced to a single equivalent conv (via `fuse_conv_bn`,
    padded to a common spatial size via `pad_kernel_to`, with the identity
    branch turned into a kernel via `identity_kernel` if present), and a
    multi-branch block's output is defined as the *sum* of its branches, so
    the fused single-conv's weights are just that same sum.

    Input
    ----------
        kernel_bias_pairs: a non-empty list of (kernel, bias) numpy-array
            pairs, all with identical shapes.

    Output
    ----------
        (kernel, bias): the elementwise sum across all pairs.
    '''
    kernels, biases = zip(*kernel_bias_pairs)
    return np.sum(kernels, axis=0), np.sum(biases, axis=0)
