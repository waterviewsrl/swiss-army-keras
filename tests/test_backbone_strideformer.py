"""Architecture tests for StrideFormer (plan.md §3.5, Phase 9).

Verified directly against `PaddlePaddle/PaddleSeg`'s
`paddleseg/models/backbones/strideformer.py` source and its two per-variant
training configs (`configs/pp_mobileseg/pp_mobileseg_{tiny,base}_
ade20k_512x512_80k.yml`) - not the earlier research pass's summary, which
had flagged its own uncertainty, and which this implementation's first
draft trusted for `STRIDEFORMER_CONFIGS`'s `cfg1` (missed 2 of its 3
blocks, cascading into wrong channel counts for every later stage) before
being caught by `test_variant_channel_counts_match_reference` below.

Unlike MobileOne/FastViT/RepViT (§2.1/§2.3/§2.4), no structural
reparameterization exists here either (same as MobileNetV4, §2.2) - no
fusion golden-output test. What replaces "the important test" here is
`test_variant_channel_counts_match_reference`, the regression test for the
real `cfg1` transcription bug found while building this.
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras._backbone_strideformer import (
    STRIDEFORMER_CONFIGS,
    StrideFormer_Tiny,
    StrideFormer_Base,
    strideformer_stage_output_names,
)

VARIANTS = [
    ('StrideFormer-Tiny', StrideFormer_Tiny, [32, 64, 128]),
    ('StrideFormer-Base', StrideFormer_Base, [64, 128, 192]),
]

# StrideFormer's SEA attention blocks (trans1/trans2) apply their own
# internal stride-2 downsample (`stride_attention=True` in both released
# variants) on top of the backbone's own stride32 deepest tap - an
# effective stride64 internal resolution, the same "needs real spatial
# extent, not 1x1" reasoning DDRNet's/PIDNet's own `DDRNET_INPUT_SIZE=256`
# comments already document for their own stride64 taps.
INPUT_SIZE = 256


def _random_batch(batch_size=1):
    return np.random.rand(batch_size, INPUT_SIZE, INPUT_SIZE, 3).astype("float32")


@pytest.mark.parametrize("key,ctor,expected_channels", VARIANTS)
def test_variant_builds_predicts_and_channel_counts_match_reference(key, ctor, expected_channels):
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = ctor(include_top=False, weights=None, input_tensor=inp)

    assert len(model.outputs) == 3
    assert [int(o.shape[-1]) for o in model.outputs] == expected_channels

    names = strideformer_stage_output_names(STRIDEFORMER_CONFIGS[key]['model_name'])
    sizes = [int(model.get_layer(n).output.shape[1]) for n in names]
    assert sizes == sorted(sizes, reverse=True)
    assert len(set(sizes)) == 3, "the 3 taps must be at genuinely different (stride8/16/32) resolutions"

    preds = model.predict(_random_batch(), verbose=0)
    for p in preds:
        assert np.all(np.isfinite(p))


def test_only_stem_stage_has_a_stem_conv():
    """Regression test for a real, easy-to-get-wrong detail: only `smb1`
    (the very first `StackedMV3Block` stage) builds a stem conv - `smb2`/
    `smb3`/`smb4` start directly with their own `ResidualUnit`s, confirmed
    from `StackedMV3Block.__init__`'s `if self.stem:` guard.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = StrideFormer_Tiny(include_top=False, weights=None, input_tensor=inp)
    assert any(layer.name == 'StrideFormer_Tiny_smb1_stem_conv' for layer in model.layers)
    assert not any(layer.name == 'StrideFormer_Tiny_smb2_stem_conv' for layer in model.layers)
    assert not any(layer.name == 'StrideFormer_Tiny_smb3_stem_conv' for layer in model.layers)
    assert not any(layer.name == 'StrideFormer_Tiny_smb4_stem_conv' for layer in model.layers)


def test_axial_attention_output_channel_contract():
    """Regression test for `AxialAttention`'s output shape contract: it
    must always return a `(B, H, 1, dh)` (row) or `(B, 1, W, dh)` (column)
    tensor - `dh = value_dim * num_heads`, read from `v`'s own channel
    count at call time, not assumed equal to `q`/`k`'s `nh_kd` width (which
    can differ whenever `attn_ratio != 1`, the actual case here -
    `attn_ratio=2` means `v` is twice as wide as `q`/`k`).
    """
    from swiss_army_keras._backbone_strideformer import AxialAttention

    num_heads, key_dim, attn_ratio = 4, 16, 2
    nh_kd = num_heads * key_dim
    dh = num_heads * key_dim * attn_ratio

    q = tf.keras.Input((8, 8, nh_kd))
    k = tf.keras.Input((8, 8, nh_kd))
    v = tf.keras.Input((8, 8, dh))

    row_out = AxialAttention(num_heads, key_dim, axis='row', name='row_test')([q, k, v])
    assert row_out.shape[1:] == (8, 1, dh)

    col_out = AxialAttention(num_heads, key_dim, axis='column', name='col_test')([q, k, v])
    assert col_out.shape[1:] == (1, 8, dh)


def test_hard_swish_and_relu6_match_expected_formulas():
    """Regression test for why these exist as `Layer` subclasses instead of
    `Activation('hard_swish'/'relu6', ...)` strings: `tf_keras` (this
    repo's forced legacy-Keras-2 backend, `TF_USE_LEGACY_KERAS=1`) doesn't
    register either name - confirmed empirically while building this
    (`ValueError: Unknown activation function 'hard_swish'`) - and checks
    the actual formulas against hand-computed values.
    """
    from swiss_army_keras._backbone_strideformer import HardSwish, ReLU6, HardSigmoid

    x = tf.constant([-5.0, -3.0, 0.0, 3.0, 6.0])

    relu6_out = ReLU6()(x).numpy()
    np.testing.assert_allclose(relu6_out, [0.0, 0.0, 0.0, 3.0, 6.0])

    hard_swish_out = HardSwish()(x).numpy()
    expected_hard_swish = x.numpy() * np.clip(x.numpy() + 3.0, 0.0, 6.0) / 6.0
    np.testing.assert_allclose(hard_swish_out, expected_hard_swish, rtol=1e-5)

    # slope=0.2, offset=0.5 - not the more common relu6(x+3)/6 (slope 1/6),
    # confirmed from `paddleseg/models/backbones/mobilenetv3.py::Hardsigmoid`.
    hsig_out = HardSigmoid()(x).numpy()
    expected_hsig = np.clip(0.2 * x.numpy() + 0.5, 0.0, 1.0)
    np.testing.assert_allclose(hsig_out, expected_hsig, rtol=1e-5)


def test_weights_imagenet_not_implemented():
    with pytest.raises(NotImplementedError, match="weights='imagenet'"):
        StrideFormer_Tiny(include_top=False, weights='imagenet', input_tensor=tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3)))


def test_requires_input_tensor_or_input_shape():
    with pytest.raises(ValueError):
        StrideFormer_Tiny(include_top=False, weights=None)


def test_include_top_builds_classifier_head():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = StrideFormer_Tiny(include_top=True, weights=None, input_tensor=inp, classes=10)
    assert model.output_shape == (None, 10)
