"""Tests for `swiss_army_keras._model_depth` (plan.md §13.5, Phase 11) -
two genuinely different monocular depth architectures: `lr_aspp_depth`
(a thin, post-hoc-wrapped reuse of the existing `lr_aspp` decoder) and
`fastdepth_head` (a from-scratch FastDepth-style NNConv5 cascade with
additive skips).

Numeric-property assertions use `MobileOne-S0` (this repo's own backbone,
confirmed unaffected by plan.md §10's `MobileNetV2`+`weights=None`
degenerate-output finding), `MobileNetV2` only for broad build/predict/
shape coverage - same convention `pose_regression_head`/
`reid_embedding_head`'s own tests already established.

`fastdepth_head`'s own sigmoid output is deliberately only checked for
shape/bounds/finiteness, not "genuinely varies" - a freshly-initialized
(`weights=None`), untrained NNConv5 cascade's raw pre-activation output can
have a very small amplitude (confirmed empirically: 0.0-0.05 in one probe),
and int8-style fake-quantization of a low-amplitude signal can legitimately
look near-constant post-sigmoid without this being a bug - the same
"quantization can't represent finer detail than its own step size"
characteristic, not the severe all-exactly-zero collapse §10 documents.
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras.models import fastdepth_head, lr_aspp_depth

INPUT_SIZE = 96


def _random_batch(batch_size=2):
    return np.random.rand(batch_size, INPUT_SIZE, INPUT_SIZE, 3).astype("float32")


class TestLrAsppDepth:
    def test_sigmoid_mode_builds_predicts_bounded_and_varies_with_mobileone(self):
        inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
        model = lr_aspp_depth(
            inp, output_mode='sigmoid', backbone="MobileOne-S0", weights=None, freeze_backbone=False,
        )
        out = model.predict(_random_batch(), verbose=0)
        assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, 1)
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert out.std() > 1e-6, "output degenerated to a constant"

    def test_linear_mode_builds_and_predicts_unbounded_finite(self):
        inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
        model = lr_aspp_depth(
            inp, output_mode='linear', backbone="MobileNetV2", weights=None, freeze_backbone=False,
        )
        out = model.predict(_random_batch(), verbose=0)
        assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, 1)
        assert np.all(np.isfinite(out))

    def test_linear_mode_returns_lr_aspp_unwrapped(self):
        """`output_mode='linear'` must return `lr_aspp`'s own model
        unchanged - no extra wrapping layer at all - confirmed by checking
        `'sigmoid'` mode's model has exactly one more layer than
        `'linear'` mode's (the single added post-hoc wrap layer), same
        backbone/input otherwise. (Not checked via "no sigmoid-named layer
        in linear mode" - `lr_aspp`'s own internal LR-ASPP module already
        has one, `_model_lr_aspp.py`'s `{name}_gap_sigmoid` gate, unrelated
        to this wrapper's own output activation.)
        """
        inp1 = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
        linear_model = lr_aspp_depth(
            inp1, output_mode='linear', backbone="MobileOne-S0", weights=None, freeze_backbone=False,
        )
        inp2 = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
        sigmoid_model = lr_aspp_depth(
            inp2, output_mode='sigmoid', backbone="MobileOne-S0", weights=None, freeze_backbone=False,
        )
        assert len(sigmoid_model.layers) == len(linear_model.layers) + 1

    def test_sigmoid_wrap_needs_no_second_quantize_apply(self):
        """The post-hoc sigmoid wrap must build/predict correctly without
        re-running `quantize_apply` - mirrors
        `TestUint8InputWrapper.test_no_second_quantize_apply_needed`
        (`tests/test_layer_utils.py`), the same established pattern this
        wrapper follows.
        """
        inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
        model = lr_aspp_depth(
            inp, output_mode='sigmoid', backbone="MobileOne-S0", weights=None, freeze_backbone=False,
        )
        out = model.predict(_random_batch(), verbose=0)
        assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, 1)

    def test_invalid_output_mode_raises(self):
        inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
        with pytest.raises(ValueError, match="output_mode"):
            lr_aspp_depth(inp, output_mode='bogus', backbone="MobileOne-S0", weights=None)


class TestFastDepthHead:
    def test_sigmoid_mode_builds_and_predicts_bounded_with_mobileone(self):
        inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
        model = fastdepth_head(
            inp, output_mode='sigmoid', backbone="MobileOne-S0", weights=None, freeze_backbone=False,
        )
        out = model.predict(_random_batch(), verbose=0)
        assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, 1)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_linear_mode_builds_and_predicts_finite_with_mobilenetv2(self):
        inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
        model = fastdepth_head(
            inp, output_mode='linear', backbone="MobileNetV2", weights=None, freeze_backbone=False,
        )
        out = model.predict(_random_batch(), verbose=0)
        assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, 1)
        assert np.all(np.isfinite(out))

    def test_output_resolution_matches_input_regardless_of_backbone_stride(self):
        """The final-resolution safety net (real-shape check, not an
        assumed `2**5` stride product) must land exactly at the input's
        own spatial size - confirmed at a non-default input size too.
        """
        size = 128
        inp = tf.keras.Input((size, size, 3))
        model = fastdepth_head(
            inp, backbone="MobileOne-S0", weights=None, freeze_backbone=False,
        )
        out = model.predict(np.random.rand(1, size, size, 3).astype("float32"), verbose=0)
        assert out.shape == (1, size, size, 1)

    def test_custom_decoder_channels_respected(self):
        inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
        model = fastdepth_head(
            inp, decoder_channels=(64, 32, 16, 8, 4), backbone="MobileOne-S0",
            weights=None, freeze_backbone=False,
        )
        out = model.predict(_random_batch(), verbose=0)
        assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, 1)

    def test_invalid_output_mode_raises(self):
        inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
        with pytest.raises(ValueError, match="output_mode"):
            fastdepth_head(inp, output_mode='bogus', backbone="MobileOne-S0", weights=None)

    def test_sigmoid_is_the_last_layer_so_it_can_be_dropped_at_export(self):
        from swiss_army_keras import deploy_utils

        inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
        model = fastdepth_head(inp, backbone="MobileNetV2", weights=None, freeze_backbone=False)
        plain = deploy_utils.strip_qat(model)
        logits_model = deploy_utils.drop_output_activations(plain)
        assert logits_model.dropped_output_activations == ['sigmoid']

        x = _random_batch()
        np.testing.assert_allclose(
            tf.sigmoid(logits_model.predict(x, verbose=0)).numpy(), plain.predict(x, verbose=0), atol=1e-6)


# Backbones that expose fewer than five feature levels: RepViT (four,
# strides 4 to 32) and StrideFormer (three, strides 8 to 32). Both heads
# used to fail on them with their default arguments.
@pytest.mark.parametrize("backbone,n_skips", [("RepViT-M0.9", 3), ("StrideFormer-Tiny", 2)])
def test_depth_heads_adapt_to_backbones_with_fewer_feature_levels(backbone, n_skips):
    size = 128  # StrideFormer needs an input that stays even down to stride 32
    inp = tf.keras.Input((size, size, 3))
    model = lr_aspp_depth(inp, backbone=backbone, weights=None, freeze_backbone=False)
    assert model.output_shape == (None, size, size, 1)

    inp = tf.keras.Input((size, size, 3))
    model = fastdepth_head(inp, backbone=backbone, weights=None, freeze_backbone=False)
    out = model.predict(np.random.rand(2, size, size, 3).astype("float32"), verbose=0)
    assert out.shape == (2, size, size, 1)
    assert out.min() >= 0.0 and out.max() <= 1.0
    skip_adds = [l.name for l in model.layers if l.name.endswith('_add') and 'skip' in l.name]
    assert len(skip_adds) == n_skips
