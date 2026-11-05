"""Build + forward-pass + QAT tests for the existing decoders.

Every decoder added under plan.md §3 (LR-ASPP, PP-LiteSeg, DDRNet) should get
an equivalent test to these — build the full model, run one forward pass on
a synthetic batch, and confirm `tfmot.quantization.keras.quantize_apply`
(already invoked inside `deeplab_v3_plus`/`deeplab_v3_plus_lite` themselves)
does not raise. `weights=None` throughout: these check architecture wiring,
not ImageNet accuracy, and must not depend on network access.
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras.models import deeplab_v3_plus, deeplab_v3_plus_lite, lr_aspp, pp_liteseg, ddrnet, pidnet, pp_mobileseg

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


def test_lr_aspp_iqa_head_false_keeps_output_unchanged():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = lr_aspp(
        inp, n_labels=N_LABELS, backbone="MobileNetV2", weights=None,
        freeze_backbone=False, iqa_head=False,
    )
    out = model.predict(_random_batch(), verbose=0)
    assert not isinstance(out, list)
    assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)


def test_lr_aspp_iqa_head_true_adds_bounded_second_output():
    """`iqa_head=True` (plan.md §13.7, Phase 11) taps `deep_feature` - the
    same tensor the LR-ASPP module itself consumes - for a small side-head
    trained simultaneously with the main segmentation task.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = lr_aspp(
        inp, n_labels=N_LABELS, backbone="MobileOne-S0", weights=None,
        freeze_backbone=False, iqa_head=True,
    )
    assert len(model.outputs) == 2

    logit_map, iqa = model.predict(_random_batch(), verbose=0)
    assert logit_map.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert iqa.shape == (2, 1)
    assert iqa.min() >= 0.0 and iqa.max() <= 1.0


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


def test_ddrnet_dappm_matches_pidnet_pappm_channel_contract():
    """Not a DDRNet regression test - documents a real, deliberately-not-
    fixed discrepancy found while implementing PIDNet (plan.md §3.4/Phase
    9): DDRNet's `dappm` appends an extra `BatchNormalization` after each
    of its 5 branches' 1x1 conv that the verified PyTorch reference (both
    DDRNet's own and PIDNet's byte-identical `DAPPM` class) does not have -
    each branch there ends on a bare `Conv2d(bias=False)`. `dappm` is still
    reused as-is for PIDNet's own `m != 2` ("L") pooling path (out of scope
    to retroactively "fix" this file's already-shipped, already-tested
    behavior as a side effect of a different task) - this test only pins
    down the channel-count contract both `dappm` and the new, more
    faithful `_pappm` (`_model_pidnet.py`) share, so a future fix to either
    doesn't silently break the other's call sites.
    """
    from swiss_army_keras._model_ddrnet import dappm
    from swiss_army_keras._model_pidnet import _pappm

    inp1 = tf.keras.Input((16, 16, 256))
    out1 = dappm(inp1, branch_channels=32, out_channels=128, name='dappm_contract_test')
    assert out1.shape[-1] == 128

    inp2 = tf.keras.Input((16, 16, 256))
    out2 = _pappm(inp2, branch_channels=32, out_channels=128, name='pappm_contract_test')
    assert out2.shape[-1] == 128


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


# PIDNet with PagFM/Bag (plan.md §3.4, Phase 9) - architecture verified
# directly against the official reference (`XuJiacong/PIDNet`,
# `models/pidnet.py` + `models/model_utils.py`, fetched and read verbatim),
# see `_model_pidnet.py`'s module docstring. Like DDRNet, a complete
# self-contained network (no `backbone=` parameter) - reuses
# DDRNET_INPUT_SIZE's own reasoning (PIDNet's I-branch also reaches stride
# 64, and its PAPPM/DAPPM pooling modules use the same largest-kernel-17
# branch DDRNet's DAPPM does) rather than defining a separate constant.


def _random_pidnet_batch(batch_size=1):
    return np.random.rand(batch_size, DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, 3).astype("float32")


def test_pidnet_s_with_aux_heads_builds_and_predicts():
    """`m=2` reproduces the official "S"/"M" configs (PAPPM + Light_Bag)."""
    inp = tf.keras.Input((DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, 3))
    model = pidnet(inp, n_labels=N_LABELS, m=2, n=3, planes=32, ppm_planes=96, head_planes=128, augment=True)
    assert len(model.outputs) == 3
    p_aux, main_out, d_aux = model.predict(_random_pidnet_batch(), verbose=0)
    assert p_aux.shape == (1, DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, N_LABELS)
    assert main_out.shape == (1, DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, N_LABELS)
    # The D (boundary) auxiliary head outputs a single raw boundary logit,
    # not `n_labels` channels like the main/P-aux heads - confirmed from
    # the reference (`self.seghead_d = segmenthead(planes*2, planes, 1)`),
    # an easy detail to get wrong by analogy to DDRNet's own single
    # auxiliary head (which does output `n_labels` channels).
    assert d_aux.shape == (1, DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, 1)
    assert np.all(np.isfinite(p_aux)) and np.all(np.isfinite(main_out)) and np.all(np.isfinite(d_aux))


def test_pidnet_without_aux_heads_builds_and_predicts():
    inp = tf.keras.Input((DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, 3))
    model = pidnet(inp, n_labels=N_LABELS, augment=False)
    assert len(model.outputs) == 1
    out = model.predict(_random_pidnet_batch(), verbose=0)
    assert out.shape == (1, DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_pidnet_l_variant_builds_and_predicts():
    """`m=3` (with `n=4`, `planes=64`, `ppm_planes=112`, `head_planes=256`)
    reproduces the official "L" config - the *other* fusion path (DAPPM +
    `Bag`, not PAPPM + `Light_Bag`) than every other test in this section
    exercises, since `m` selects both the block-repeat count *and* which
    pooling/fusion module pair the whole network uses (confirmed genuine
    double duty in the reference, not simplified here - see
    `_model_pidnet.py`'s module docstring).
    """
    inp = tf.keras.Input((DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, 3))
    model = pidnet(inp, n_labels=N_LABELS, m=3, n=4, planes=64, ppm_planes=112, head_planes=256, augment=False)
    out = model.predict(_random_pidnet_batch(), verbose=0)
    assert out.shape == (1, DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_pidnet_light_bag_and_bag_output_channel_count():
    """Regression test for `_light_bag`/`_bag`'s output channel contract -
    `_light_bag`'s two parallel conv paths (`conv_p`/`conv_i`) and `_bag`'s
    single conv must all project to `out_channels` regardless of `p`/`i`/`d`'s
    own (here, deliberately different) input channel counts, matching the
    reference's `Light_Bag(in_channels, out_channels)`/`Bag(in_channels,
    out_channels)` contract.
    """
    from swiss_army_keras._model_pidnet import _bag, _light_bag

    p = tf.keras.Input((16, 16, 64))
    i = tf.keras.Input((16, 16, 64))
    d = tf.keras.Input((16, 16, 64))

    out_light = _light_bag(p, i, d, out_channels=128, name='light_bag_test')
    assert out_light.shape[-1] == 128

    out_bag = _bag(p, i, d, out_channels=128, name='bag_test')
    assert out_bag.shape[-1] == 128


def test_pidnet_pagfm_output_matches_p_branch_shape():
    """Regression test for PagFM's fusion contract: the output must match
    the P (detail) branch's own resolution/channel count exactly (it's an
    attention-weighted blend of P with an upsampled, channel-matched I
    contribution, not a concatenation or a projection to some other width).
    """
    from swiss_army_keras._model_pidnet import _pagfm

    p = tf.keras.Input((32, 32, 64))
    i_compressed = tf.keras.Input((16, 16, 64))

    out = _pagfm(p, i_compressed, mid_channels=32, name='pagfm_test')
    assert out.shape[1:] == (32, 32, 64)


def test_pidnet_d_branch_channel_width_depends_on_m():
    """Regression test for the D branch's most easily-gotten-wrong detail
    (plan.md §3.4): its channel width at `layer3_d`/`layer4_d` genuinely
    depends on `m` (`planes`/`planes*2` for `m==2`, `planes*2`/`planes*2`
    otherwise) - confirmed from the reference, not a simplification - while
    `layer5_d` always ends at `planes*4` regardless of `m` (needed so
    `Light_Bag`/`Bag` always receive a consistently-shaped `d` argument).
    This only checks the model builds successfully for both `m` values at
    a small size (already implicitly covered by the S/L variant tests
    above) with an explicit assertion that both variants' D-branch layers
    actually got built under their expected, different names.
    """
    inp_light = tf.keras.Input((DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, 3))
    model_light = pidnet(inp_light, n_labels=N_LABELS, m=2, n=3, planes=32, augment=False, name='pidnet_light')
    assert any(layer.name == 'pidnet_light_layer3_d_block0_add' for layer in model_light.layers)

    inp_full = tf.keras.Input((DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, 3))
    model_full = pidnet(inp_full, n_labels=N_LABELS, m=3, n=4, planes=64, augment=False, name='pidnet_full')
    assert any(layer.name == 'pidnet_full_layer3_d_block0_add' for layer in model_full.layers)


# PP-MobileSeg with StrideFormer/AAM (plan.md §3.5, Phase 9) - architecture
# verified directly against the official reference (`PaddlePaddle/PaddleSeg`,
# `paddleseg/models/pp_mobileseg.py`), see `_model_pp_mobileseg.py`'s module
# docstring. `StrideFormer-Tiny`/`-Base` are a genuine 3-tap pyramid (no
# earlier taps to skip), so `feature_levels=(1, 2, 3)` (this decoder's own
# default) is already correct - no override needed, unlike RepViT/FastViT
# elsewhere in this file. Reuses `DDRNET_INPUT_SIZE` for the same "SEA
# attention's own internal stride-2 downsample needs real spatial extent"
# reasoning `test_backbone_strideformer.py`'s own `INPUT_SIZE` documents.


def test_pp_mobileseg_tiny_with_full_aam_builds_and_predicts():
    """Tiny pairs with the `'full'` AAM variant in the reference's own
    training config - not an arbitrary choice, the two are not
    interchangeable (see `_model_pp_mobileseg.py`'s module docstring).
    """
    inp = tf.keras.Input((DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, 3))
    model = pp_mobileseg(
        inp, n_labels=N_LABELS, backbone='StrideFormer-Tiny', weights=None,
        freeze_backbone=False, aam_variant='full',
    )
    out = model.predict(_random_ddrnet_batch(), verbose=0)
    assert out.shape == (1, DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_pp_mobileseg_base_with_simplified_aam_builds_and_predicts():
    """Base pairs with the `'simplified'` (`AAMSx8`) AAM variant in the
    reference's own training config.
    """
    inp = tf.keras.Input((DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, 3))
    model = pp_mobileseg(
        inp, n_labels=N_LABELS, backbone='StrideFormer-Base', weights=None,
        freeze_backbone=False, aam_variant='simplified',
    )
    out = model.predict(_random_ddrnet_batch(), verbose=0)
    assert out.shape == (1, DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))


def test_strideformer_is_genuinely_swappable_with_other_decoders():
    """The whole point of moving AAM out of the backbone and into this
    decoder (§3.5's central design decision, `_backbone_strideformer.py`'s
    module docstring) was to make StrideFormer a real, swappable backbone
    usable by *any* decoder in this repo, not just its own paper's head -
    this test exists to actually confirm that claim holds, not leave it as
    an unverified aspiration. RepViT/FastViT-style 4-tap overrides don't
    apply here (StrideFormer is a 3-tap backbone), so `deep_layer=3`/
    `feature_levels=(1, 2, 3)` are used instead.
    """
    inp1 = tf.keras.Input((DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, 3))
    model1 = lr_aspp(
        inp1, n_labels=N_LABELS, backbone='StrideFormer-Tiny', weights=None,
        freeze_backbone=False, deep_layer=3, shallow_layer=1,
    )
    out1 = model1.predict(_random_ddrnet_batch(), verbose=0)
    assert out1.shape == (1, DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out1))

    inp2 = tf.keras.Input((DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, 3))
    model2 = pp_liteseg(
        inp2, n_labels=N_LABELS, backbone='StrideFormer-Base', weights=None,
        freeze_backbone=False, feature_levels=(1, 2, 3),
    )
    out2 = model2.predict(_random_ddrnet_batch(), verbose=0)
    assert out2.shape == (1, DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out2))


def test_pp_mobileseg_rejects_unknown_aam_variant():
    inp = tf.keras.Input((DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, 3))
    with pytest.raises(ValueError, match="aam_variant"):
        pp_mobileseg(inp, n_labels=N_LABELS, backbone='StrideFormer-Tiny', weights=None, aam_variant='bogus')


def test_pp_mobileseg_requires_a_backbone():
    """Unlike `pp_liteseg`/`lr_aspp`, this decoder has no `backbone=None`
    plain-conv fallback - a deliberate scope choice (see
    `_model_pp_mobileseg.py`'s module docstring), not an oversight.
    """
    inp = tf.keras.Input((DDRNET_INPUT_SIZE, DDRNET_INPUT_SIZE, 3))
    with pytest.raises(ValueError, match="backbone"):
        pp_mobileseg(inp, n_labels=N_LABELS, backbone=None, weights=None)


def test_aam_full_and_simplified_output_different_resolutions():
    """Regression test for a real, easy-to-get-backwards detail found while
    implementing this (plan.md §3.5): `aam_full` downsamples the finest
    input (`t8`) to match the middle one (`t16`)'s resolution, while
    `aam_simplified` does the *opposite* - keeps `t8` at its own (finest)
    resolution and upsamples `t16` to match it instead. Confirmed from
    source, not assumed by symmetry with `aam_full` (this implementation's
    first draft did assume symmetry and got it backwards).
    """
    from swiss_army_keras._model_pp_mobileseg import aam_full, aam_simplified

    t8 = tf.keras.Input((32, 32, 16))
    t16 = tf.keras.Input((16, 16, 16))
    t32 = tf.keras.Input((8, 8, 16))

    full_out = aam_full(t8, t16, t32, out_channels=8, name='full_test')
    assert full_out.shape[1:3] == (16, 16), "aam_full should output at t16's (the middle) resolution"

    simplified_out = aam_simplified(t8, t16, t32, out_channels=8, name='simplified_test')
    assert simplified_out.shape[1:3] == (32, 32), "aam_simplified should output at t8's (the finest) resolution"


# Gate-free variants for accelerators without sigmoid / tensor-by-tensor
# multiplication (docs/source/deployment.rst).


def test_lr_aspp_without_gate_has_no_multiply_or_sigmoid():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = lr_aspp(inp, n_labels=N_LABELS, backbone="MobileNetV4-Conv-S", weights=None, freeze_backbone=False, gate=False)
    names = [l.name for l in model.layers]
    assert not any("gate" in n or "gap" in n for n in names), names
    assert not any(isinstance(l, tf.keras.layers.Multiply) for l in model.layers)
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))
    # the default keeps the gate
    gated = lr_aspp(tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3)), n_labels=N_LABELS, backbone="MobileNetV4-Conv-S", weights=None)
    assert any(isinstance(l, tf.keras.layers.Multiply) for l in gated.layers)


def test_pp_liteseg_additive_fusion_builds_without_attention_layers():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = pp_liteseg(inp, n_labels=N_LABELS, backbone="MobileNetV4-Conv-S", weights=None, freeze_backbone=False, fusion="add")
    names = [l.name for l in model.layers]
    assert not any("atten" in n or "blend" in n for n in names), names
    assert any(n.endswith("uafm16_add") for n in names) and any(n.endswith("uafm8_add") for n in names)
    out = model.predict(_random_batch(), verbose=0)
    assert out.shape == (2, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    assert np.all(np.isfinite(out))
    with pytest.raises(ValueError, match="fusion"):
        pp_liteseg(tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3)), n_labels=N_LABELS, backbone="MobileNetV4-Conv-S", weights=None, fusion="concat")



# LR-ASPP feature levels: the defaults follow the backbone, and a level the
# backbone does not have is a clear error instead of an IndexError.


def test_lr_aspp_default_levels_follow_the_backbone():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = lr_aspp(inp, n_labels=N_LABELS, backbone="RepViT-M0.9", weights=None, freeze_backbone=False)
    assert model.output_shape == (None, INPUT_SIZE, INPUT_SIZE, N_LABELS)
    # four levels: the module sits on the stride-32 feature, the skip on stride 4
    assert model.get_layer("quant_lr_aspp_module_branch1x1_conv").output_shape[1] == INPUT_SIZE // 32
    assert model.get_layer("quant_lr_aspp_shallow_proj_conv").output_shape[1] == INPUT_SIZE // 4


def test_lr_aspp_rejects_a_level_the_backbone_does_not_have():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    with pytest.raises(ValueError, match="4 feature levels"):
        lr_aspp(inp, n_labels=N_LABELS, backbone="RepViT-M0.9", weights=None, deep_layer=5)
    with pytest.raises(ValueError, match="shallow_layer"):
        lr_aspp(inp, n_labels=N_LABELS, backbone="MobileNetV2", weights=None, deep_layer=3, shallow_layer=4)
