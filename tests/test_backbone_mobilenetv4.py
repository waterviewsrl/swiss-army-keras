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


def test_weights_local_path_loads(tmp_path):
    """`weights=<path>` (e.g. a checkpoint produced by
    `scripts/port_mobilenetv4_weights.py`, plan.md §12.5) should load real
    weights into a freshly-built model - checked structurally (do saved and
    reloaded weights match?), not numerically against PyTorch, since that
    needs `torch`/`timm` + network access this test suite shouldn't depend
    on (plan.md §12: one-time conversion-only tools, never a runtime
    dependency).
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    source_model = MobileNetV4_ConvS(include_top=False, weights=None, input_tensor=inp)
    weights_path = tmp_path / "mobilenetv4_conv_s.weights.h5"
    source_model.save_weights(str(weights_path))

    inp2 = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    loaded_model = MobileNetV4_ConvS(include_top=False, weights=str(weights_path), input_tensor=inp2)

    for source_layer, loaded_layer in zip(source_model.layers, loaded_model.layers):
        for source_w, loaded_w in zip(source_layer.get_weights(), loaded_layer.get_weights()):
            np.testing.assert_array_equal(source_w, loaded_w)


def test_extradw_stride_is_on_mid_conv_not_start():
    """Regression test for a real architecture bug found while porting real
    ImageNet weights (plan.md §12.5): when a UIB block has both depthwise
    convs (`start_k>0` and `mid_k>0`, "ExtraDW" - the most common
    downsampling block kind in every variant), the reference always puts
    the block's stride on `dw_mid`, never on `dw_start` - this file
    previously had it backwards. Confirmed directly via the reference
    (`UniversalInvertedResidual.__init__`: `dw_mid` always gets
    `stride=stride` when it exists; `dw_start` only gets the stride when
    `dw_mid` doesn't exist at all).
    """
    from swiss_army_keras._backbone_mobilenetv4 import uib_block

    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    uib_block(inp, 32, start_k=3, mid_k=5, stride=2, expand_ratio=4, name='block')
    model = tf.keras.Model(inp, uib_block(inp, 32, start_k=3, mid_k=5, stride=2, expand_ratio=4, name='block'))

    dw_start = model.get_layer('block_dwstart_dw')
    dw_mid = model.get_layer('block_dwmid_dw')
    assert dw_start.strides == (1, 1), "dw_start must NOT carry the block's stride when dw_mid also exists"
    assert dw_mid.strides == (2, 2), "dw_mid must carry the block's full stride"


def test_hybrid_uib_blocks_have_layer_scale():
    """Regression test for a real bug found while porting real ImageNet
    weights (plan.md §12.5): the reference applies a learnable per-channel
    `LayerScale` to every UIB block's output for Hybrid-M/L
    (`layer_scale_init_value=1e-5`) but not for Conv-S/M/L
    (`layer_scale_init_value=None`) - this file had no `LayerScale` in
    `uib_block` at all until this fix.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    hybrid_model = MobileNetV4_HybridM(include_top=False, weights=None, input_tensor=inp)
    assert any(l.name.endswith('_scale') and 'stage2_block0' in l.name for l in hybrid_model.layers), \
        "Hybrid-M's UIB blocks should have a LayerScale layer"

    inp2 = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    conv_model = MobileNetV4_ConvS(include_top=False, weights=None, input_tensor=inp2)
    uib_scale_layers = [l for l in conv_model.layers if l.name.endswith('_scale') and 'stage2_block0' in l.name]
    assert not uib_scale_layers, "Conv-S's UIB blocks should have no LayerScale layer"


def test_mqa_kv_have_independent_downsample_convs():
    """Regression test for a real bug found while porting real ImageNet
    weights (plan.md §12.5): the reference (`MultiQueryAttention2d`) builds
    two *separate*, independently-trained depthwise conv+BN modules for K
    and V's optional spatial downsample (`key.down_conv`, `value.down_conv`)
    - this file previously shared one downsample conv's output between both
    K and V projections.
    """
    from swiss_army_keras._backbone_mobilenetv4 import mobile_mqa_block

    inp = tf.keras.Input((16, 16, 32))
    out = mobile_mqa_block(inp, num_heads=4, key_dim=16, kv_stride=2, name='mqa')
    model = tf.keras.Model(inp, out)

    k_dw = model.get_layer('mqa_k_dw')
    v_dw = model.get_layer('mqa_v_dw')
    assert k_dw is not v_dw
    assert k_dw.get_weights()[0].shape == v_dw.get_weights()[0].shape
    assert not np.array_equal(k_dw.get_weights()[0], v_dw.get_weights()[0]), \
        "K and V downsample convs must be independently initialized/trained, not a shared conv"


def test_mqa_output_proj_has_no_bias():
    """Regression test for a real bug found while porting real ImageNet
    weights (plan.md §12.5): the reference's `MultiQueryAttention2d` never
    adds a bias to Q/K/V/output projections (`use_bias=False` throughout,
    including the output proj - confirmed via the reference's own state
    dict having no `attn.output.proj.bias` key); this file had
    `use_bias=True` on the output proj specifically.
    """
    from swiss_army_keras._backbone_mobilenetv4 import mobile_mqa_block

    inp = tf.keras.Input((16, 16, 32))
    out = mobile_mqa_block(inp, num_heads=4, key_dim=16, kv_stride=1, name='mqa')
    model = tf.keras.Model(inp, out)
    assert model.get_layer('mqa_out_proj').use_bias is False


def test_stem_conv_uses_explicit_symmetric_padding():
    """Regression test for a real bug found while porting real ImageNet
    weights (plan.md §12.5): every stride>1 conv in this file used Keras's
    `padding='same'` (can pad asymmetrically at stride>1) instead of the
    reference's fixed, symmetric `kernel_size // 2` padding (`timm`'s
    `create_conv2d` - confirmed by reading its `.padding` attribute
    directly) - the same bug class already fixed for MobileOne/FastViT,
    just never applied here. Checked structurally: the stem conv must be
    preceded by an explicit `ZeroPadding2D` and itself use `padding='valid'`.
    """
    inp = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = MobileNetV4_ConvS(include_top=False, weights=None, input_tensor=inp)
    stem_conv = model.get_layer('MobileNetV4_ConvS_stem_conv')
    assert stem_conv.padding == 'valid'
    pad_layer = model.get_layer('MobileNetV4_ConvS_stem_conv_pad')
    assert isinstance(pad_layer, tf.keras.layers.ZeroPadding2D)
