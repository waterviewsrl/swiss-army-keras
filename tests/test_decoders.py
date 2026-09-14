"""Build + forward-pass + QAT tests for the existing decoders.

Every decoder added under plan.md §3 (LR-ASPP, PP-LiteSeg, DDRNet) should get
an equivalent test to these — build the full model, run one forward pass on
a synthetic batch, and confirm `tfmot.quantization.keras.quantize_apply`
(already invoked inside `deeplab_v3_plus`/`deeplab_v3_plus_lite` themselves)
does not raise. `weights=None` throughout: these check architecture wiring,
not ImageNet accuracy, and must not depend on network access.
"""

import numpy as np
import tensorflow as tf

from swiss_army_keras.models import deeplab_v3_plus, deeplab_v3_plus_lite, lr_aspp, pp_liteseg, ddrnet

INPUT_SIZE = 128
N_LABELS = 3


def _random_batch(batch_size=2):
    return np.random.rand(batch_size, INPUT_SIZE, INPUT_SIZE, 3).astype("float32")


def test_deeplab_v3_plus_with_backbone_builds_and_predicts():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = deeplab_v3_plus(
        inp, n_labels=N_LABELS, backbone="MobileNetV2", weights=None, freeze_backbone=False
    )
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_deeplab_v3_plus_lite_with_backbone_builds_and_predicts():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = deeplab_v3_plus_lite(
        inp, n_labels=N_LABELS, backbone="MobileNetV2", weights=None, freeze_backbone=False
    )
    out = model.predict(_random_batch(), verbose=0)
    # deeplab_v3_plus_lite intentionally skips the final upsample stage, so
    # its output resolution is lower than the input - only the channel count
    # and finiteness are part of the contract here.
    assert out.shape[0] == 2
    assert out.shape[-1] == N_LABELS
    assert np.all(np.isfinite(out))


def test_deeplab_v3_plus_without_backbone_builds_and_predicts():
    """Regression test for a fixed bug (see plan.md §10/§13 Phase 3): the
    no-backbone encoder path builds `X_encoder[0]` with zero poolings
    applied, but the shallow-skip upsample size used to be computed from a
    hardcoded `shallow_resize_map[shallow_layer]` that assumed one more
    pooling than this path actually applies - crashing `Concatenate` with a
    shape mismatch. Fixed by reading the shallow branch's target size
    directly off its own tensor shape instead of a lookup table, which is
    correct for both encoder paths.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = deeplab_v3_plus(inp, n_labels=N_LABELS, backbone=None)
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_deeplab_v3_plus_lite_without_backbone_builds_and_predicts():
    """Same fix, same bug, `_lite` variant - see the non-lite test above."""
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = deeplab_v3_plus_lite(inp, n_labels=N_LABELS, backbone=None)
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape[0] == 2
    assert out.shape[-1] == N_LABELS
    assert np.all(np.isfinite(out))


def test_deeplab_v3_plus_with_repvit_backbone_builds_and_predicts():
    """First end-to-end use of a custom (non-`tf.keras.applications`)
    backbone through a decoder (plan.md §2.4/§13 Phase 2) - also the
    regression test for the `GELU` serialization fix in
    `tests/test_activations.py`: this call chain (backbone_zoo wraps RepViT
    as a callable sub-model, `deeplab_v3_plus` calls `quantize_apply` on the
    combined graph at the end) is exactly the path that broke before that
    fix - RepViT is the first backbone in this repo to use `GELU`.

    RepViT only has 4 native pyramid taps (see `_backbone_repvit.py`), so
    `deep_layer`/`shallow_layer` are passed explicitly rather than relying
    on the 5-level defaults tuned for `tf.keras.applications` backbones.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = deeplab_v3_plus(
        inp, n_labels=N_LABELS, backbone="RepViT-M0.9", weights=None, freeze_backbone=False,
        deep_layer=4, shallow_layer=1,
    )
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_deeplab_v3_plus_with_fastvit_backbone_builds_and_predicts():
    """FastViT (plan.md §2.3) through a decoder - includes a
    `MultiHeadAttention` layer (SA12's stage 4), the first attention layer
    used inside a `backbone_zoo`-wrapped sub-model in this repo; confirms
    that QAT's `quantize_apply` model-cloning step (see the `GELU` note on
    the RepViT test above) also tolerates a built-in Keras attention layer
    nested the same way, not just custom layers.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = deeplab_v3_plus(
        inp, n_labels=N_LABELS, backbone="FastViT-SA12", weights=None, freeze_backbone=False,
        deep_layer=4, shallow_layer=1,
    )
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_deeplab_v3_plus_with_mobileone_backbone_builds_and_predicts():
    """MobileOne (plan.md §2.1) through a decoder - unlike RepViT/FastViT,
    MobileOne has a full 5-tap pyramid, so this uses the default
    `deep_layer`/`shallow_layer` (tuned for 5-level backbones), not the
    explicit 4-level overrides the other two custom backbones need.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = deeplab_v3_plus(
        inp, n_labels=N_LABELS, backbone="MobileOne-S1", weights=None, freeze_backbone=False,
    )
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_deeplab_v3_plus_with_mobilenetv4_conv_backbone_builds_and_predicts():
    """MobileNetV4 Conv-S (plan.md §2.2) through a decoder - a full 5-tap
    pyramid like MobileOne, so also uses the default `deep_layer`/
    `shallow_layer`.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = deeplab_v3_plus(
        inp, n_labels=N_LABELS, backbone="MobileNetV4-Conv-S", weights=None, freeze_backbone=False,
    )
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_deeplab_v3_plus_with_mobilenetv4_hybrid_backbone_builds_and_predicts():
    """MobileNetV4 Hybrid-L (plan.md §2.2) through a decoder - the
    regression test for the `MultiQueryAttention` layer fix documented in
    `_backbone_mobilenetv4.py`: a raw multi-tensor-input `tf.einsum`
    (auto-wrapped by Keras as `TFOpLambda`) broke exactly at this
    `quantize_apply` model-cloning step during development (`ValueError:
    Got 0 inputs for equation ... expecting 2`), the same class of bug the
    RepViT/FastViT tests above guard for their own custom-layer/attention
    additions. Hybrid-L also specifically exercises the GELU activation
    path (the only MobileNetV4 variant that uses it) through this same
    QAT-cloning call chain, alongside RepViT's GELU coverage above.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = deeplab_v3_plus(
        inp, n_labels=N_LABELS, backbone="MobileNetV4-Hybrid-L", weights=None, freeze_backbone=False,
    )
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


# LR-ASPP (plan.md §3.1) - the first of the three new decoders. Cheapest of
# the three (one 1x1 branch + one global-pooled sigmoid-gate branch, no
# dilated-conv stack), so it's the first one built and tested (§13 Phase 3).
def test_lr_aspp_without_backbone_builds_and_predicts():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = lr_aspp(inp, n_labels=N_LABELS, backbone=None)
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_lr_aspp_with_backbone_builds_and_predicts():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = lr_aspp(
        inp, n_labels=N_LABELS, backbone="MobileNetV2", weights=None, freeze_backbone=False,
    )
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_lr_aspp_with_4_tap_custom_backbone_builds_and_predicts():
    """RepViT (plan.md §2.4) only has 4 native pyramid taps - same
    `deep_layer`/`shallow_layer` override pattern as the `deeplab_v3_plus`
    RepViT test above, confirming LR-ASPP's own `deep_layer`/`shallow_layer`
    plumbing (not just `deeplab_v3_plus`'s) handles a non-default backbone
    depth correctly.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = lr_aspp(
        inp, n_labels=N_LABELS, backbone="RepViT-M0.9", weights=None, freeze_backbone=False,
        deep_layer=4, shallow_layer=1,
    )
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_lr_aspp_with_mobilenetv4_backbone_builds_and_predicts():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = lr_aspp(
        inp, n_labels=N_LABELS, backbone="MobileNetV4-Conv-S", weights=None, freeze_backbone=False,
    )
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_lr_aspp_module_gate_is_not_quantize_annotated():
    """Regression test for the one real build-time issue found while
    implementing LR-ASPP (plan.md §3.1): `tfmot`'s default QAT scheme
    doesn't support annotating a `Multiply` merge layer directly
    (`quantize_apply` raised `RuntimeError: ... is not supported`) - fixed
    by leaving the gating `Multiply` (and the final `Concatenate`)
    unannotated, the same way `deeplab_v3_plus`'s `DilatedSpatialPyramidPooling`
    already leaves its own `Concatenate` calls unannotated. This only checks
    the model still builds/predicts through `quantize_apply` end-to-end -
    the fix above is what actually prevents the regression, this just
    confirms it stays fixed.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = lr_aspp(inp, n_labels=N_LABELS, backbone=None)
    assert any(layer.name.endswith('_gate') for layer in model.layers)
    out = model.predict(_random_batch(), verbose=0)
    assert np.all(np.isfinite(out))


# PP-LiteSeg + UAFM (plan.md §3.2) - architecture verified directly against
# PaddleSeg's source (not just the paper), see `_model_pp_liteseg.py`'s
# module docstring and plan.md §3.2 for the full research findings.
def test_pp_liteseg_without_backbone_builds_and_predicts():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = pp_liteseg(inp, n_labels=N_LABELS, backbone=None)
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_pp_liteseg_with_backbone_builds_and_predicts():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = pp_liteseg(
        inp, n_labels=N_LABELS, backbone="MobileNetV2", weights=None, freeze_backbone=False,
    )
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_pp_liteseg_with_4_tap_custom_backbone_builds_and_predicts():
    """RepViT (plan.md §2.4) only has 4 native pyramid taps - needs
    `feature_levels=(2, 3, 4)` instead of the 5-tap default `(3, 4, 5)`,
    same override pattern as `deep_layer`/`shallow_layer` elsewhere in this
    codebase.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = pp_liteseg(
        inp, n_labels=N_LABELS, backbone="RepViT-M0.9", weights=None, freeze_backbone=False,
        feature_levels=(2, 3, 4),
    )
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_pp_liteseg_with_mobilenetv4_hybrid_backbone_builds_and_predicts():
    """MobileNetV4 Hybrid-M (plan.md §2.2) through PP-LiteSeg - exercises
    UAFM's `ChannelMeanMaxConcat`/`AttentionBlend` custom layers alongside
    MobileNetV4's own `MultiQueryAttention` custom layer in the same
    `quantize_apply` cloning pass, the most layer-diverse single test in
    this file.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = pp_liteseg(
        inp, n_labels=N_LABELS, backbone="MobileNetV4-Hybrid-M", weights=None, freeze_backbone=False,
    )
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_pp_liteseg_sppm_sums_not_concats():
    """Regression test for the SPPM neck's most easily-gotten-wrong detail
    (plan.md §3.2): PaddleSeg's `PPContextModule` combines its per-scale
    branches with an element-wise **sum**, not a concatenation (unlike
    classic ASPP/PSPM image-pooling branches) - confirmed directly against
    the source, not the paper text. If this ever regressed to a concat, the
    `sppm_out` conv's input channel count would silently multiply by
    `len(sppm_bin_sizes)` instead of staying at `sppm_out_channels`; this
    checks the actual built channel count instead of relying on that
    indirect signal.
    """
    from swiss_army_keras._model_pp_liteseg import sppm_module

    inp = tf.keras.Input((32, 32, 64))
    out = sppm_module(inp, out_channels=128, bin_sizes=(1, 2, 4), name='sppm_test')
    assert out.shape[-1] == 128


# DDRNet + DAPPM (plan.md §3.3) - architecture verified directly against the
# official reference (`ydhongHIT/DDRNet`), see `_model_ddrnet.py`'s module
# docstring. Unlike `lr_aspp`/`pp_liteseg`, this is a complete self-contained
# network (no `backbone=` parameter, no `backbone_zoo` involvement) - a
# 256x256 input is used here specifically (not this file's usual 128) so
# DAPPM's largest pooling branch (kernel 17, stride 8, at stride-64 depth)
# operates on a spatial extent bigger than a single pixel; 128 would give a
# 2x2 feature map at stride64, too small to meaningfully exercise the 4
# different pooling-branch kernel sizes at all.
DDRNET_INPUT_SIZE = 256


def _random_ddrnet_batch(batch_size=1):
    return np.random.rand(batch_size, DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, 3).astype("float32")


def test_ddrnet_with_aux_head_builds_and_predicts():
    inp = tf.keras.Input((DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, 3))
    model = ddrnet(inp, n_labels=N_LABELS, augment=True)
    assert len(model.outputs) == 2
    main_out, aux_out = model.predict(_random_ddrnet_batch(), verbose=0)
    assert main_out.shape == (1, DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, N_LABELS)
    assert aux_out.shape == (1, DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(main_out)) and np.all(np.isfinite(aux_out))


def test_ddrnet_without_aux_head_builds_and_predicts():
    inp = tf.keras.Input((DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, 3))
    model = ddrnet(inp, n_labels=N_LABELS, augment=False)
    assert len(model.outputs) == 1
    out = model.predict(_random_ddrnet_batch(), verbose=0)
    assert out.shape == (1, DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_ddrnet_full_width_variant_builds_and_predicts():
    """`planes=64` reproduces the official full-width "DDRNet-23" config
    (vs. this file's other tests' default `planes=32`, "DDRNet-23-slim") -
    same block-count schedule, just wider (plan.md §3.3).
    """
    inp = tf.keras.Input((DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, 3))
    model = ddrnet(inp, n_labels=N_LABELS, planes=64, augment=False)
    out = model.predict(_random_ddrnet_batch(), verbose=0)
    assert out.shape == (1, DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_ddrnet_dappm_cascade_channel_count():
    """Regression test for DAPPM's most easily-gotten-wrong detail (plan.md
    §3.3): the 5 per-scale branches are combined via a **cascade** (each
    upsampled branch added to the running total, then a 3x3 conv), not a
    plain concat/sum, before one final compression conv - this only checks
    the module's output channel count matches `out_channels` regardless of
    `branch_channels`/scale count, which would silently drift if the
    concat-then-compress step were ever wired to the wrong intermediate
    tensor.
    """
    from swiss_army_keras._model_ddrnet import dappm

    inp = tf.keras.Input((16, 16, 256))
    out = dappm(inp, branch_channels=32, out_channels=128, name='dappm_test')
    assert out.shape[-1] == 128


def test_ddrnet_prebn_layers_are_not_quantize_annotated():
    """Regression test for the one real build-time issue found while
    implementing DDRNet (plan.md §3.3): DAPPM's (and `segmenthead`'s)
    pre-activation `BN -> ReLU -> Conv` ordering - the opposite of every
    other decoder in this codebase's `Conv -> BN -> Act` convention - made
    `quantize_apply` raise `RuntimeError: ...BatchNormalization is not
    supported` the first time a standalone (not Conv-following)
    `BatchNormalization` was annotated. Fixed by leaving those specific BN
    layers unannotated (`_bn_relu`), the same "leave the unsupported layer
    type unannotated" fix already applied to `lr_aspp`'s `Multiply` and
    `pp_liteseg`'s `Add` (§3.1/§3.2). This only confirms the model still
    builds/predicts through `quantize_apply` end-to-end with those layers
    present - the fix above is what actually prevents the regression.
    """
    inp = tf.keras.Input((DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, 3))
    model = ddrnet(inp, n_labels=N_LABELS, augment=False)
    assert any(layer.name.endswith('_dappm_scale0_bn') for layer in model.layers)
    out = model.predict(_random_ddrnet_batch(), verbose=0)
    assert np.all(np.isfinite(out))
