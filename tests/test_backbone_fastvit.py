"""Architecture and reparameterization tests for FastViT (plan.md §2.3).

Per-variant configs verified against `keras_cv_attention_models` (plan.md
§12 research). The reparameterization test is the important one - FastViT's
RepMixer fusion is more involved than RepViT's (a *subtracted* branch, a
learnable per-channel layer-scale, and a separate residual, all folded into
one conv - see `_backbone_fastvit.py`'s module docstring), so this is where
a subtle sign or axis error would show up as wrong numbers, not a crash.
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras._backbone_fastvit import (
    FASTVIT_CONFIGS,
    FastViT_MA36,
    FastViT_S12,
    FastViT_SA12,
    FastViT_SA24,
    FastViT_SA36,
    FastViT_T8,
    FastViT_T12,
    reparameterize_fastvit,
    fastvit_stage_output_names,
)

VARIANTS = [
    ('FastViT-T8', FastViT_T8),
    ('FastViT-T12', FastViT_T12),
    ('FastViT-S12', FastViT_S12),
    ('FastViT-SA12', FastViT_SA12),
    ('FastViT-SA24', FastViT_SA24),
    ('FastViT-SA36', FastViT_SA36),
    ('FastViT-MA36', FastViT_MA36),
]

# The 3 conv-only variants (no attention stage) vs. the 4 SA/MA variants
# (attention in stage 4) - some tests only make sense for one group.
CONV_ONLY_VARIANTS = [('FastViT-T8', FastViT_T8), ('FastViT-T12', FastViT_T12), ('FastViT-S12', FastViT_S12)]
ATTENTION_VARIANTS = [
    ('FastViT-SA12', FastViT_SA12), ('FastViT-SA24', FastViT_SA24),
    ('FastViT-SA36', FastViT_SA36), ('FastViT-MA36', FastViT_MA36),
]


@pytest.mark.parametrize("key,ctor", VARIANTS)
def test_variant_builds_with_expected_pyramid(key, ctor):
    config = FASTVIT_CONFIGS[key]
    inp = tf.keras.Input((64, 64, 3))
    model = ctor(include_top=False, weights=None, input_tensor=inp)

    names = fastvit_stage_output_names(config['model_name'])
    assert len(names) == 4

    sizes = []
    for tap_name, expected_channels in zip(names, config['stage_channels']):
        layer = model.get_layer(tap_name)
        assert int(layer.output.shape[-1]) == expected_channels
        sizes.append(int(layer.output.shape[1]))

    assert sizes == sorted(sizes, reverse=True)
    assert len(set(sizes)) == 4
    assert model.output_shape == (None, sizes[-1], sizes[-1], config['stage_channels'][-1])


@pytest.mark.parametrize("key,ctor", ATTENTION_VARIANTS)
def test_attention_variant_has_mhsa_only_in_stage4(key, ctor):
    inp = tf.keras.Input((64, 64, 3))
    model = ctor(include_top=False, weights=None, input_tensor=inp)
    mhsa_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.MultiHeadAttention)]
    config = FASTVIT_CONFIGS[key]
    assert len(mhsa_layers) == config['stage_blocks'][-1]
    assert all(f"{config['model_name']}_stage4_" in l.name for l in mhsa_layers)


@pytest.mark.parametrize("key,ctor", CONV_ONLY_VARIANTS)
def test_conv_only_variant_has_no_attention(key, ctor):
    inp = tf.keras.Input((64, 64, 3))
    model = ctor(include_top=False, weights=None, input_tensor=inp)
    assert not any(isinstance(l, tf.keras.layers.MultiHeadAttention) for l in model.layers)


def test_weights_imagenet_not_implemented():
    with pytest.raises(NotImplementedError, match="weights='imagenet'"):
        FastViT_T8(include_top=False, weights='imagenet', input_tensor=tf.keras.Input((64, 64, 3)))


def test_requires_input_tensor_or_input_shape():
    with pytest.raises(ValueError):
        FastViT_T8(include_top=False, weights=None)


def test_include_top_builds_classifier_head():
    inp = tf.keras.Input((64, 64, 3))
    model = FastViT_T8(include_top=True, weights=None, input_tensor=inp, classes=10)
    assert model.output_shape == (None, 10)


def _randomize_weights(model, rng, bn_spread=0.1, scale_range=(0.5, 1.5), attn_std=0.05):
    """BN stats get a *small* spread (see the identical rationale in
    tests/test_backbone_repvit.py - a wide spread compounds through many
    blocks into activations large enough that genuine float32 accumulation
    drift, not a fusion bug, dominates the diff). LayerScale gammas are
    pulled *away* from their tiny init default (1e-5/1e-6) - at init, the
    scale step is numerically almost a no-op, which would make a sign error
    in the fusion math (e.g. treating a subtract as an add) invisible.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            gamma, beta, mean, var = layer.get_weights()
            layer.set_weights([
                rng.uniform(1 - bn_spread, 1 + bn_spread, size=gamma.shape).astype("float32"),
                rng.uniform(-bn_spread, bn_spread, size=beta.shape).astype("float32"),
                rng.uniform(-bn_spread, bn_spread, size=mean.shape).astype("float32"),
                rng.uniform(1 - bn_spread, 1 + bn_spread, size=var.shape).astype("float32"),
            ])
        elif type(layer).__name__ == "LayerScale":
            (gamma,) = layer.get_weights()
            layer.set_weights([rng.uniform(*scale_range, size=gamma.shape).astype("float32")])
        elif isinstance(layer, tf.keras.layers.MultiHeadAttention):
            layer.set_weights([rng.standard_normal(w.shape).astype("float32") * attn_std for w in layer.get_weights()])


@pytest.mark.parametrize("key,ctor", VARIANTS)
def test_reparameterize_matches_original_output(key, ctor):
    """Input is 128x128, not a smaller size: at 32x32 this network's 5
    stride-2 stages collapse the output to a *single* spatial pixel, which
    hid a real bug here during development - `rep_conv_block`'s stem calls
    and `rep_downsample`'s 7x7/3x3 fusion (both stride=2) used to rely on
    `padding='same'`, whose asymmetric-padding behavior at stride>1 can
    desynchronize two different-kernel-size branches' sampling grids
    (`pad_kernel_to` assumes symmetric centering). A 32x32 input's 1x1
    final output couldn't reveal this (aggregating the whole image into one
    pixel masks a geometric misalignment); a 128x128 input's 4x4 final
    output can. Fixed by switching to explicit symmetric `ZeroPadding2D` +
    `'valid'` convs - see plan.md §13 Phase 2's MobileOne entry, where this
    was found first, then audited and fixed here too.
    """
    tf.random.set_seed(0)
    inp = tf.keras.Input((128, 128, 3))
    model = ctor(include_top=False, weights=None, input_tensor=inp)

    rng = np.random.default_rng(0)
    _randomize_weights(model, rng)

    x = rng.standard_normal((1, 128, 128, 3)).astype("float32")
    y_original = model.predict(x, verbose=0)

    fused_model = reparameterize_fastvit(model)
    y_fused = fused_model.predict(x, verbose=0)

    assert len(fused_model.layers) < len(model.layers)
    np.testing.assert_allclose(y_original, y_fused, rtol=1e-3, atol=1e-3)


def test_reparameterize_rejects_unrelated_model():
    inp = tf.keras.Input((8, 8, 3))
    out = tf.keras.layers.Conv2D(4, 3)(inp)
    unrelated_model = tf.keras.Model(inp, out)
    with pytest.raises(ValueError, match="_fastvit_config"):
        reparameterize_fastvit(unrelated_model)
