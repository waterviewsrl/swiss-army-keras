"""Shape-contract tests for `swiss_army_keras._backbone_zoo.backbone_zoo`.

Every backbone added under plan.md §2 (MobileOne/MobileNetV4/FastViT/RepViT)
must satisfy the same "N named feature maps, strictly decreasing spatial
size" contract the existing `tf.keras.applications` backbones already
satisfy here — this file is the reference for that contract, parametrized so
new backbones slot in as additional cases once they're registered (plan.md
§2.5's `custom_backbone_constructors`).
"""

import pytest
import tensorflow as tf

from swiss_army_keras._backbone_zoo import backbone_zoo

# weights=None everywhere below: these tests check architecture wiring, not
# ImageNet accuracy, and must not depend on network access to download
# pretrained weights.
#
# MobileNetV3Large was excluded here for a while: `BACKBONE_REGISTRY`'s
# entry for it used to key its 5 feature-pyramid layers by hardcoded,
# auto-generated name (`re_lu_2`, `re_lu_6`, ...), which Keras assigns from a
# *global*, process-wide counter - correct only if MobileNetV3Large happened
# to be built at the exact layer-count offset those names assumed, which
# shifted depending on what else was built earlier in the same process
# (confirmed nondeterministic across suite compositions - see plan.md §10).
# Fixed by keying that entry's layers by *index* instead of name (indices
# are stable per-model regardless of global naming state, verified
# empirically); MobileNetV3Large is back in the parametrized list below.
#
# MobileOne-S1 is included directly here (not given RepViT/FastViT-style
# separate dispatch/depth-clamp tests) since MobileOne genuinely has a full
# 5-tap pyramid (single-stride-2-conv stem, not a 4x-jump patchify stem -
# see `_backbone_mobileone.py`'s module docstring) - it satisfies this
# file's existing 5-tap assumptions exactly, no special-casing needed.
# MobileNetV4-Hybrid-M is included the same way (also a genuine 5-tap
# pyramid, see `_backbone_mobilenetv4.py`) - the Hybrid variant specifically
# so this shared/generic pyramid-contract test also exercises the
# Mobile-MQA attention path, not just the plain-conv one (which the
# dedicated `test_backbone_mobilenetv4.py` already covers per-variant).
BACKBONES = ["MobileNetV2", "MobileNetV3Large", "ResNet50", "MobileOne-S1", "MobileNetV4-Hybrid-M"]


@pytest.mark.parametrize("backbone_name", BACKBONES)
def test_backbone_zoo_full_depth_pyramid(backbone_name):
    inp = tf.keras.Input((96, 96, 3))
    model = backbone_zoo(backbone_name, None, inp, depth=5, freeze_backbone=False, freeze_batch_norm=True)

    assert len(model.outputs) == 5

    sizes = [int(o.shape[1]) for o in model.outputs]
    assert sizes == sorted(sizes, reverse=True), f"feature map sizes not strictly decreasing: {sizes}"
    assert len(set(sizes)) == len(sizes), f"expected strictly decreasing sizes, got ties: {sizes}"

    assert callable(model.preprocessing)


@pytest.mark.parametrize("backbone_name", BACKBONES)
@pytest.mark.parametrize("depth", [1, 3, 5])
def test_backbone_zoo_respects_requested_depth(backbone_name, depth):
    inp = tf.keras.Input((96, 96, 3))
    model = backbone_zoo(backbone_name, None, inp, depth=depth, freeze_backbone=False, freeze_batch_norm=True)
    assert len(model.outputs) == depth


@pytest.mark.parametrize("backbone_name", BACKBONES)
def test_backbone_zoo_forward_pass(backbone_name):
    import numpy as np

    inp = tf.keras.Input((96, 96, 3))
    model = backbone_zoo(backbone_name, None, inp, depth=5, freeze_backbone=False, freeze_batch_norm=True)
    x = np.random.rand(1, 96, 96, 3).astype("float32")
    outputs = model.predict(x, verbose=0)
    assert len(outputs) == 5
    for out, expected_shape in zip(outputs, model.output_shape):
        assert out.shape == (1,) + tuple(expected_shape[1:])


def test_backbone_zoo_freeze_backbone():
    inp = tf.keras.Input((96, 96, 3))
    model = backbone_zoo("MobileNetV2", None, inp, depth=5, freeze_backbone=True, freeze_batch_norm=True)
    assert all(not layer.trainable for layer in model.layers)


def test_mobilenetv3large_layer_lookup_is_stable_across_build_order():
    """Regression test for the bug documented in the `BACKBONES` comment
    above: build another ReLU-using backbone first (shifts Keras's global
    layer-auto-naming counter), then confirm MobileNetV3Large's index-based
    layer lookup still resolves to a valid 5-level pyramid.
    """
    inp_other = tf.keras.Input((96, 96, 3))
    backbone_zoo("MobileNetV2", None, inp_other, depth=5, freeze_backbone=False, freeze_batch_norm=True)

    inp = tf.keras.Input((96, 96, 3))
    model = backbone_zoo("MobileNetV3Large", None, inp, depth=5, freeze_backbone=False, freeze_batch_norm=True)
    assert len(model.outputs) == 5
    sizes = [int(o.shape[1]) for o in model.outputs]
    assert sizes == sorted(sizes, reverse=True)
    assert len(set(sizes)) == len(sizes)


def test_backbone_zoo_unknown_backbone_raises_clear_error():
    inp = tf.keras.Input((96, 96, 3))
    with pytest.raises(ValueError, match="Unknown backbone"):
        backbone_zoo("NotARealBackbone", None, inp, depth=5, freeze_backbone=False, freeze_batch_norm=True)


# RepViT (plan.md §2.4) is the first custom (non-tf.keras.applications)
# backbone registered - kept out of the `BACKBONES` list above since it only
# exposes a 4-level pyramid (see `_backbone_repvit.py`'s module docstring),
# not the 5 those tests hardcode; architecture-specific coverage (per-variant
# channel counts, block counts, reparameterization) lives in
# `tests/test_backbone_repvit.py`. These just cover dispatch through the
# shared `backbone_zoo()` entry point.
def test_backbone_zoo_repvit_dispatch_and_depth_clamp():
    inp = tf.keras.Input((64, 64, 3))
    # RepViT only has 4 native taps - requesting depth=5 (the default other
    # backbones satisfy exactly) must clamp instead of raising.
    model = backbone_zoo("RepViT-M0.9", None, inp, depth=5, freeze_backbone=False, freeze_batch_norm=True)
    assert len(model.outputs) == 4
    sizes = [int(o.shape[1]) for o in model.outputs]
    assert sizes == sorted(sizes, reverse=True)
    assert len(set(sizes)) == 4
    assert callable(model.preprocessing)


def test_backbone_zoo_repvit_respects_requested_depth():
    inp = tf.keras.Input((64, 64, 3))
    model = backbone_zoo("RepViT-M1.1", None, inp, depth=2, freeze_backbone=False, freeze_batch_norm=True)
    assert len(model.outputs) == 2


# FastViT (plan.md §2.3) - also a custom backbone, also a 4-tap pyramid, same
# reasoning as RepViT above; architecture-specific coverage lives in
# tests/test_backbone_fastvit.py.
def test_backbone_zoo_fastvit_dispatch_and_depth_clamp():
    inp = tf.keras.Input((64, 64, 3))
    model = backbone_zoo("FastViT-SA12", None, inp, depth=5, freeze_backbone=False, freeze_batch_norm=True)
    assert len(model.outputs) == 4
    sizes = [int(o.shape[1]) for o in model.outputs]
    assert sizes == sorted(sizes, reverse=True)
    assert len(set(sizes)) == 4
    assert callable(model.preprocessing)


def test_backbone_zoo_fastvit_respects_requested_depth():
    inp = tf.keras.Input((64, 64, 3))
    model = backbone_zoo("FastViT-T8", None, inp, depth=2, freeze_backbone=False, freeze_batch_norm=True)
    assert len(model.outputs) == 2
