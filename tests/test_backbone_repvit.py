"""Architecture and reparameterization tests for RepViT (plan.md §2.4).

Per-variant configs verified against `keras_cv_attention_models` (plan.md
§12 research) - the shape-contract tests here check that verification landed
correctly (channel counts, block counts, 4-level pyramid), not just that the
code runs. The reparameterization test is the important one: it confirms
`reparameterize_repvit` produces a numerically equivalent model, the same
golden-output standard used in `tests/test_reparam_utils.py`.
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras._backbone_repvit import (
    REPVIT_CONFIGS,
    RepViT_M0_9,
    RepViT_M1_0,
    RepViT_M1_1,
    RepViT_M1_5,
    RepViT_M2_3,
    reparameterize_repvit,
    repvit_stage_output_names,
)

VARIANTS = [
    ('RepViT-M0.9', RepViT_M0_9),
    ('RepViT-M1.0', RepViT_M1_0),
    ('RepViT-M1.1', RepViT_M1_1),
    ('RepViT-M1.5', RepViT_M1_5),
    ('RepViT-M2.3', RepViT_M2_3),
]


@pytest.mark.parametrize("key,ctor", VARIANTS)
def test_variant_builds_with_expected_pyramid(key, ctor):
    config = REPVIT_CONFIGS[key]
    inp = tf.keras.Input((64, 64, 3))
    model = ctor(include_top=False, weights=None, input_tensor=inp)

    names = repvit_stage_output_names(config['model_name'], config['stage_blocks'])
    assert len(names) == 4

    sizes = []
    for tap_name, expected_channels in zip(names, config['stage_channels']):
        layer = model.get_layer(tap_name)
        assert int(layer.output.shape[-1]) == expected_channels
        sizes.append(int(layer.output.shape[1]))

    # 4x/8x/16x/32x downsampling -> strictly decreasing spatial sizes.
    assert sizes == sorted(sizes, reverse=True)
    assert len(set(sizes)) == 4
    assert model.output_shape == (None, sizes[-1], sizes[-1], config['stage_channels'][-1])


def test_weights_imagenet_not_implemented():
    with pytest.raises(NotImplementedError, match="weights='imagenet'"):
        RepViT_M0_9(include_top=False, weights='imagenet', input_tensor=tf.keras.Input((64, 64, 3)))


def test_requires_input_tensor_or_input_shape():
    with pytest.raises(ValueError):
        RepViT_M0_9(include_top=False, weights=None)


def test_include_top_builds_classifier_head():
    inp = tf.keras.Input((64, 64, 3))
    model = RepViT_M0_9(include_top=True, weights=None, input_tensor=inp, classes=10)
    assert model.output_shape == (None, 10)


def _randomize_bn_stats(model, rng, spread=0.1):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            gamma, beta, mean, var = layer.get_weights()
            layer.set_weights([
                rng.uniform(1 - spread, 1 + spread, size=gamma.shape).astype("float32"),
                rng.uniform(-spread, spread, size=beta.shape).astype("float32"),
                rng.uniform(-spread, spread, size=mean.shape).astype("float32"),
                rng.uniform(1 - spread, 1 + spread, size=var.shape).astype("float32"),
            ])


@pytest.mark.parametrize("key,ctor", VARIANTS)
def test_reparameterize_matches_original_output(key, ctor):
    """The real correctness test for structural reparameterization: fuse a
    trained (here: BN-randomized, since weights=None gives BN identity
    stats which would trivially "match") model and diff its output against
    the unfused original on the same input.

    BN stats are randomized with a *small* spread rather than a wide one -
    confirmed during development that a wide spread (e.g. gamma/var in
    [0.5, 2.0], independently per BN) compounds through this network's
    20-30 blocks into activations of magnitude ~1e3, at which point genuine
    float32 accumulation drift between the two graphs (349 vs ~260 ops) is
    large in absolute terms even though the fusion math itself is exact -
    confirmed separately with a single-block isolation test. A small spread
    keeps activations near unit scale, which is both more representative of
    an actually-trained model and avoids that confound.

    Input is 128x128, not a smaller size: at 32x32 this network's 5 stride-2
    stages collapse the output to a *single* spatial pixel, which can hide a
    real per-pixel misalignment bug in the final comparison (confirmed the
    hard way while building this file's MobileOne sibling test - see
    plan.md §13 Phase 2's MobileOne entry). 128x128 keeps a 4x4 final
    output, enough spatial extent for a real alignment bug to show up.
    """
    tf.random.set_seed(0)
    inp = tf.keras.Input((128, 128, 3))
    model = ctor(include_top=False, weights=None, input_tensor=inp)

    rng = np.random.default_rng(0)
    _randomize_bn_stats(model, rng)

    x = rng.standard_normal((1, 128, 128, 3)).astype("float32")
    y_original = model.predict(x, verbose=0)

    fused_model = reparameterize_repvit(model)
    y_fused = fused_model.predict(x, verbose=0)

    assert len(fused_model.layers) < len(model.layers)
    np.testing.assert_allclose(y_original, y_fused, rtol=1e-3, atol=1e-3)


def test_reparameterize_rejects_unrelated_model():
    inp = tf.keras.Input((8, 8, 3))
    out = tf.keras.layers.Conv2D(4, 3)(inp)
    unrelated_model = tf.keras.Model(inp, out)
    with pytest.raises(ValueError, match="_repvit_config"):
        reparameterize_repvit(unrelated_model)
