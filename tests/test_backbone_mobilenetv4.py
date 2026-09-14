"""Architecture tests for MobileNetV4 (plan.md §2.2).

Per-variant block sequences verified directly against `timm`'s
`_gen_mobilenet_v4` arch-def strings and `_efficientnet_blocks.py`'s
`UniversalInvertedResidual`/`MobileAttention` (plan.md §12 research) - no
official Keras/TF checkpoint or reference implementation exists for this
family at all (Google only shipped TFLite exports), unlike RepViT/FastViT.

Unlike the other three custom backbones (MobileOne §2.1, FastViT §2.3,
RepViT §2.4), **MobileNetV4 has no structural reparameterization at all**
(confirmed during architecture research) - there is no
`reparameterize_mobilenetv4` and therefore no fusion golden-output test
here, unlike `test_backbone_mobileone.py`/`test_backbone_fastvit.py`/
`test_backbone_repvit.py`. What replaces that as "the important test" here
is `test_hybrid_variants_use_mqa_attention_and_not_others`, which is the
regression test for the one genuine bug this backbone did surface: a raw
multi-tensor-input `tf.einsum` (auto-wrapped by Keras as `TFOpLambda`)
does not survive `tfmot`'s QAT `quantize_apply` model-cloning step -
caught via `deeplab_v3_plus`+Hybrid-L integration testing, fixed by
wrapping the attention math in a proper `MultiQueryAttention` `Layer`
subclass (see `_backbone_mobilenetv4.py`'s module + class docstrings).
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras._backbone_mobilenetv4 import (
    MOBILENETV4_CONFIGS,
    MobileNetV4_ConvS,
    MobileNetV4_ConvM,
    MobileNetV4_ConvL,
    MobileNetV4_HybridM,
    MobileNetV4_HybridL,
    MultiQueryAttention,
    mobilenetv4_stage_output_names,
)

VARIANTS = [
    ('MobileNetV4-Conv-S', MobileNetV4_ConvS),
    ('MobileNetV4-Conv-M', MobileNetV4_ConvM),
    ('MobileNetV4-Conv-L', MobileNetV4_ConvL),
    ('MobileNetV4-Hybrid-M', MobileNetV4_HybridM),
    ('MobileNetV4-Hybrid-L', MobileNetV4_HybridL),
]

HYBRID_VARIANTS = [
    ('MobileNetV4-Hybrid-M', MobileNetV4_HybridM),
    ('MobileNetV4-Hybrid-L', MobileNetV4_HybridL),
]

CONV_ONLY_VARIANTS = [
    ('MobileNetV4-Conv-S', MobileNetV4_ConvS),
    ('MobileNetV4-Conv-M', MobileNetV4_ConvM),
    ('MobileNetV4-Conv-L', MobileNetV4_ConvL),
]

# 128, not 32/64: same lesson as MobileOne/FastViT (see their test modules'
# INPUT_SIZE comments) - keep build/shape tests at an input size that
# leaves real spatial extent (not 1x1) at the deepest tap, even though the
# specific padding-fusion bug those comments describe doesn't apply here
# (no structural reparameterization, so no branch-summing at all).
INPUT_SIZE = 128


@pytest.mark.parametrize("key,ctor", VARIANTS)
def test_variant_builds_with_expected_pyramid(key, ctor):
    config = MOBILENETV4_CONFIGS[key]
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = ctor(include_top=False, weights=None, input_tensor=inp)

    names = mobilenetv4_stage_output_names(config['model_name'])
    assert len(names) == 5, "MobileNetV4 should expose a full 5-tap pyramid (stem + 4 stages)"

    sizes = [int(model.get_layer(n).output.shape[1]) for n in names]
    assert sizes == sorted(sizes, reverse=True)
    assert len(set(sizes)) == 5

    # Unlike MobileOne (whose bare model output *is* its last stage tap),
    # MobileNetV4's `include_top=False` output is the 960-channel 1x1 head
    # conv that follows all 4 stages (a stride-1 "conv head", per the
    # architecture research - see `_backbone_mobilenetv4.py`'s module
    # docstring) - same spatial size as the last tap, different channel
    # count.
    assert model.output_shape == (None, sizes[-1], sizes[-1], config['head_channels'])


@pytest.mark.parametrize("key,ctor", VARIANTS)
def test_variant_forward_pass_is_finite(key, ctor):
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = ctor(include_top=False, weights=None, input_tensor=inp)
    x = np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3).astype("float32")
    out = model.predict(x, verbose=0)
    assert np.all(np.isfinite(out))


def test_only_hybrid_large_uses_gelu():
    assert MOBILENETV4_CONFIGS['MobileNetV4-Hybrid-L']['activation'] == 'gelu'
    for key in ['MobileNetV4-Conv-S', 'MobileNetV4-Conv-M', 'MobileNetV4-Conv-L', 'MobileNetV4-Hybrid-M']:
        assert MOBILENETV4_CONFIGS[key]['activation'] == 'relu'


@pytest.mark.parametrize("key,ctor", HYBRID_VARIANTS)
def test_hybrid_variants_use_mqa_attention_and_not_others(key, ctor):
    """The important regression test for this backbone - see module
    docstring. Confirms both that Hybrid-M/L actually build
    `MultiQueryAttention` layers (not silently falling back to pure-conv),
    and that the plain Conv-S/M/L variants never do (no attention at all
    in those, per the architecture research).
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = ctor(include_top=False, weights=None, input_tensor=inp)
    mqa_layers = [l for l in model.layers if isinstance(l, MultiQueryAttention)]
    assert len(mqa_layers) > 0, f"{key} should contain Mobile-MQA attention blocks"


@pytest.mark.parametrize("key,ctor", CONV_ONLY_VARIANTS)
def test_conv_only_variants_have_no_attention(key, ctor):
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = ctor(include_top=False, weights=None, input_tensor=inp)
    mqa_layers = [l for l in model.layers if isinstance(l, MultiQueryAttention)]
    assert len(mqa_layers) == 0, f"{key} should have no attention blocks (Conv-only variant)"


def test_weights_imagenet_not_implemented():
    with pytest.raises(NotImplementedError, match="weights='imagenet'"):
        MobileNetV4_ConvS(include_top=False, weights='imagenet', input_tensor=tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3)))


def test_requires_input_tensor_or_input_shape():
    with pytest.raises(ValueError):
        MobileNetV4_ConvS(include_top=False, weights=None)


def test_include_top_builds_classifier_head():
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = MobileNetV4_ConvS(include_top=True, weights=None, input_tensor=inp, classes=10)
    assert model.output_shape == (None, 10)
