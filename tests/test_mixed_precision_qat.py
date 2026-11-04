"""Build + forward-pass tests for every QAT-annotated decoder/head under a
global `mixed_float16` dtype policy (`tf.keras.mixed_precision.set_global_
policy`), the kind a caller sets to speed up backbone training on a
Tensor-Core GPU.

`tfmot`'s fake-quantization op (`FakeQuantWithMinMaxVars`) hard-requires
float32 inputs - confirmed empirically, not documented by `tfmot` itself -
so every decoder/head builder function (all of which unconditionally end
with `quantize_apply`, no opt-out) would otherwise crash under such a policy.
`swiss_army_keras.layer_utils.qat_annotate_layer` and the module-level
`QuantizeWrapper`/`QuantizeLayer` patch it applies force every QAT-annotated
layer to float32 regardless of the ambient policy, while leaving backbones
(never QAT-annotated, see `backbone_zoo`) free to actually use the policy.
These tests exist to catch a regression in that split - a future change
that makes a QAT-annotated layer's output leak the ambient float16 policy
again (as `_model_detection.py`'s regression-branch `ScaleLayer` once did,
and the "old" `deeplab_v3_plus`'s un-annotated final `Concatenate`/`Conv2D`
head did too) - not just that these models build under the *default*
float32 policy, which `test_decoders.py`/`test_classifiers.py`/
`test_detection.py` already cover.
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras.models import (
    deeplab_v3_plus,
    deeplab_v3_plus_lite,
    lr_aspp,
    pp_liteseg,
    ddrnet,
    distiller_classifier,
    fcos_lite_detection,
    gfl_lite_detection,
)

INPUT_SIZE = 128
N_LABELS = 3


@pytest.fixture(autouse=True)
def mixed_float16_policy():
    """Every test in this file runs under a global `mixed_float16` policy;
    always reset back to the default afterward (even on failure) so this
    doesn't leak into other test files' models, which assume float32.
    """
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        yield
    finally:
        tf.keras.mixed_precision.set_global_policy("float32")


def _random_batch(batch_size=2):
    return np.random.rand(batch_size, INPUT_SIZE, INPUT_SIZE, 3).astype("float32")


def _assert_finite_float32(output):
    outputs = output if isinstance(output, (list, tuple)) else [output]
    for out in outputs:
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))


def test_lr_aspp_builds_and_predicts_under_mixed_precision():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = lr_aspp(inp, n_labels=N_LABELS, backbone=None, weights=None)
    _assert_finite_float32(model.predict(_random_batch(), verbose=0))


def test_pp_liteseg_builds_and_predicts_under_mixed_precision():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = pp_liteseg(inp, n_labels=N_LABELS, backbone=None, weights=None)
    _assert_finite_float32(model.predict(_random_batch(), verbose=0))


def test_ddrnet_builds_and_predicts_under_mixed_precision():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = ddrnet(inp, n_labels=N_LABELS)
    _assert_finite_float32(model.predict(_random_batch(), verbose=0))


def test_deeplab_v3_plus_builds_and_predicts_under_mixed_precision():
    """Regression test: the un-annotated final `Concatenate`/`Conv2D` head
    (unlike `deeplab_v3_plus_lite`'s equivalent, already-annotated one) used
    to leave this decoder's output at float16 under a mixed-precision
    policy - Keras' own guidance is that a model's final output should stay
    float32 for numerical stability. Fixed by annotating both, matching
    `deeplab_v3_plus_lite`.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = deeplab_v3_plus(inp, n_labels=N_LABELS, backbone=None, weights=None)
    _assert_finite_float32(model.predict(_random_batch(), verbose=0))


def test_deeplab_v3_plus_lite_builds_and_predicts_under_mixed_precision():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = deeplab_v3_plus_lite(inp, n_labels=N_LABELS, backbone=None, weights=None)
    _assert_finite_float32(model.predict(_random_batch(), verbose=0))


def test_distiller_classifier_with_repvit_backbone_under_mixed_precision():
    """RepViT's patchify stem (4-tap pyramid) is this repo's own
    documented QAT-cloning edge case (GELU serialization); exercising it
    here too, not just a backbone-less classifier head.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = distiller_classifier(
        inp, 5, backbone="RepViT-M1.0", weights=None, freeze_backbone=False, deep_layer=4,
    )
    _assert_finite_float32(model.predict(_random_batch(), verbose=0))


def test_fcos_lite_detection_builds_and_predicts_under_mixed_precision():
    """Regression test, now covering two separate historical bugs in the
    same spot:
    (1) the regression branch's per-level scale (formerly a plain,
    non-QAT-annotated `ScaleLayer` sitting between the QAT-annotated
    `reg_head_conv` and NMS) used to silently pick up the ambient float16
    policy and cast the box predictions back down, crashing NMS
    (`TypeError: ... has type float16 that does not match expected type
    of float32`). Now implemented as `_qat_channelwise_scale` (a
    QAT-annotated `DepthwiseConv2D(1)`, replacing `ScaleLayer` for an
    unrelated `tfmot`/`quantize_apply` incompatibility - see that
    function's own docstring), which stays float32 via
    `qat_annotate_layer`'s own patch with no separate override needed.
    (2) `_nms_dispatch`'s own default (dynamic-batch) path uses
    `_AssertBatchSizeOne` (a plain custom `Layer`, replacing
    `tf.image.combined_non_max_suppression` - see `_nms_dispatch`'s own
    docstring for why) - found, via this exact test, to have the *same
    class* of bug `ScaleLayer` originally had: no explicit `dtype`, so it
    silently cast its output to float16 under this policy. Fixed by
    forcing `dtype='float32'` on `_AssertBatchSizeOne` itself (not just at
    one call site), so `fcos_lite_instance_segmentation`'s own use of the
    same layer is covered too.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = fcos_lite_detection(
        inp, n_classes=N_LABELS, backbone="MobileOne-S1", weights=None,
    )
    nms_boxes, nms_scores, nms_classes, valid_detections = model.predict(
        _random_batch(batch_size=1), verbose=0
    )
    _assert_finite_float32([nms_boxes, nms_scores, valid_detections.astype("float32")])


def test_gfl_lite_detection_builds_and_predicts_under_mixed_precision():
    """Same regression test as `fcos_lite_detection`'s own - `gfl_lite_
    detection` shares the same `_nms_dispatch`/`_AssertBatchSizeOne` path,
    previously untested under mixed precision at all.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = gfl_lite_detection(
        inp, n_classes=N_LABELS, backbone="MobileOne-S1", weights=None,
    )
    nms_boxes, nms_scores, nms_classes, valid_detections = model.predict(
        _random_batch(batch_size=1), verbose=0
    )
    _assert_finite_float32([nms_boxes, nms_scores, valid_detections.astype("float32")])


def test_backbone_still_uses_mixed_precision_policy():
    """The whole point of forcing QAT-annotated layers to float32 is that
    backbones (never QAT-annotated - see `backbone_zoo`) stay free to use
    the ambient policy for a real speedup; confirm that split actually
    holds rather than the fix accidentally forcing float32 everywhere.
    """
    from swiss_army_keras._backbone_zoo import backbone_zoo

    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    backbone = backbone_zoo("MobileOne-S1", None, inp, 5, True, True)
    conv_layers = [
        layer for layer in backbone.layers
        if "conv" in layer.name.lower() and hasattr(layer, "dtype_policy")
    ]
    assert conv_layers
    assert conv_layers[0].dtype_policy.name == "mixed_float16"
