"""One-time PyTorch(timm) -> Keras weight conversion for MobileNetV4 (plan.md
§12/§12.1/§12.5).

Not a runtime dependency of `swiss_army_keras` itself - see
`scripts/port_repvit_weights.py`'s docstring for the general "why" and the
install command for `torch`/`timm`.

MobileNetV4 is the highest-risk family in this plan (§12): no Keras
reference exists, AND no official TF/Keras *checkpoint* exists to port from
even in principle (Google only released TFLite exports) - so both the
architecture (§2.2) and this conversion were built directly from `timm`'s
`_gen_mobilenet_v4`/`_efficientnet_blocks.py` source, cross-checked
numerically rather than assumed correct from shape-matching alone (this
plan's running lesson, most recently from FastViT/MobileOne, plan.md
§12.2-§12.4). That diligence paid off - reading the source line-by-line
while writing this script found **six** real bugs, more than any other
family, several of them wrong for structural (not just numerical) reasons:

1. **BN epsilon mismatch** - identical bug, identical fix, to every other
   backbone in this plan: `1e-3` -> `1e-5`.
2. **`uib_block`'s stride was on the wrong depthwise conv.** The reference
   always gives `dw_mid` the block's full stride when it exists,
   unconditionally; `dw_start` only ever carries the stride when `dw_mid`
   doesn't exist at all. This file had it backwards (stride on `dw_start`
   whenever present) - wrong for every "ExtraDW" block (both convs present,
   the single most common block kind, and exactly the kind used at nearly
   every stage's stride-2 transition).
3. **Missing `LayerScale` on Hybrid-M/L's UIB blocks.** The reference
   applies a learnable per-channel `LayerScale` (`layer_scale_init_value=
   1e-5`) to *every* UIB block's output before its residual add, but only
   when `layer_scale_init_value` is not `None` - which the reference sets
   only for `'hybrid' in variant`. This file had no `LayerScale` in
   `uib_block` at all; Conv-S/M/L are correctly unaffected either way
   (`layer_scale_init_value=None` there too), so this only affects Hybrid-M/L.
4. **Mobile-MQA's K and V did not each get their own downsample.** The
   reference (`MultiQueryAttention2d`) builds two *separate*, independently
   -trained depthwise conv+BN modules (`key.down_conv`, `value.down_conv`)
   when `kv_stride > 1` - this file shared a single downsample conv's output
   between both K and V projections.
5. **Mobile-MQA's output projection had a bias it shouldn't.** The
   reference's `MultiQueryAttention2d` never adds a bias to Q/K/V/output
   (`bias=use_bias`, `use_bias` defaulting - and always staying - `False`,
   including on the output proj, per the reference's own source comment
   "why not here if used w/ mhsa?"); this file had `use_bias=True` on the
   output proj specifically.
6. **`MOBILENETV4_CONFIGS`'s hand-transcribed per-stage block sequences had
   real errors in 4 of the 5 variants.** Re-derived programmatically by
   importing `timm` and capturing `_gen_mobilenet_v4`'s actual `arch_def`
   list (via monkeypatching `decode_arch_def` to intercept its input, no
   network access needed) rather than trusting a second by-eye read of the
   compressed block-string notation - see `_backbone_mobilenetv4.py`'s
   `MOBILENETV4_CONFIGS` comment for the itemized diff (wrong expand ratios,
   two wrong kernel sizes, and Conv-L's stage 2 silently missing 3 of 5
   repeats of one block).

The `hidden_channels` formula (`make_divisible(in_chs * exp_ratio, 8)`, not
a plain `round()`) was also ported exactly as `_make_divisible` in the
backbone file itself, though for every real config here it turns out to be
a no-op (channel counts are already multiples of 8 and expand ratios are
integers) - see that function's docstring.

Usage
----------
    python scripts/port_mobilenetv4_weights.py [VARIANT ...]

    VARIANT: one or more of MobileNetV4-Conv-S, MobileNetV4-Conv-M,
    MobileNetV4-Conv-L, MobileNetV4-Hybrid-M, MobileNetV4-Hybrid-L (default:
    all five). Writes
    `<repo_root>/pretrained_weights/mobilenetv4/<model_name>.weights.h5`.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import swiss_army_keras  # noqa: F401 - import before tensorflow
import tensorflow as tf
import timm
import torch

from swiss_army_keras._backbone_mobilenetv4 import MOBILENETV4_CONFIGS, _build_mobilenetv4

TIMM_NAMES = {
    "MobileNetV4-Conv-S": "mobilenetv4_conv_small",
    "MobileNetV4-Conv-M": "mobilenetv4_conv_medium",
    "MobileNetV4-Conv-L": "mobilenetv4_conv_large",
    "MobileNetV4-Hybrid-M": "mobilenetv4_hybrid_medium",
    "MobileNetV4-Hybrid-L": "mobilenetv4_hybrid_large",
}


def _conv2d_weight(w):
    return w.detach().numpy().transpose(2, 3, 1, 0)


def _dwconv2d_weight(w):
    return w.detach().numpy().transpose(2, 3, 0, 1)


def _bn_weights(sd, prefix):
    return [
        sd[f"{prefix}.weight"].numpy(), sd[f"{prefix}.bias"].numpy(),
        sd[f"{prefix}.running_mean"].numpy(), sd[f"{prefix}.running_var"].numpy(),
    ]


def _set_conv_bn(keras_model, conv_name, bn_name, sd, torch_conv_key, torch_bn_prefix, depthwise):
    w = sd[torch_conv_key]
    kw = _dwconv2d_weight(w) if depthwise else _conv2d_weight(w)
    keras_model.get_layer(conv_name).set_weights([kw])
    keras_model.get_layer(bn_name).set_weights(_bn_weights(sd, torch_bn_prefix))


def _set_cn(keras_model, name, sd, torch_prefix):
    _set_conv_bn(keras_model, f"{name}_conv", f"{name}_bn", sd, f"{torch_prefix}.conv.weight", f"{torch_prefix}.bn1", depthwise=False)


def _set_er(keras_model, name, sd, torch_prefix):
    _set_conv_bn(keras_model, f"{name}_expand_conv", f"{name}_expand_bn", sd, f"{torch_prefix}.conv_exp.weight", f"{torch_prefix}.bn1", depthwise=False)
    _set_conv_bn(keras_model, f"{name}_proj_conv", f"{name}_proj_bn", sd, f"{torch_prefix}.conv_pwl.weight", f"{torch_prefix}.bn2", depthwise=False)


def _set_uir(keras_model, name, sd, torch_prefix, start_k, mid_k, has_layer_scale):
    if start_k > 0:
        _set_conv_bn(keras_model, f"{name}_dwstart_dw", f"{name}_dwstart_dw_bn", sd, f"{torch_prefix}.dw_start.conv.weight", f"{torch_prefix}.dw_start.bn", depthwise=True)
    _set_conv_bn(keras_model, f"{name}_pwexp_conv", f"{name}_pwexp_bn", sd, f"{torch_prefix}.pw_exp.conv.weight", f"{torch_prefix}.pw_exp.bn", depthwise=False)
    if mid_k > 0:
        _set_conv_bn(keras_model, f"{name}_dwmid_dw", f"{name}_dwmid_dw_bn", sd, f"{torch_prefix}.dw_mid.conv.weight", f"{torch_prefix}.dw_mid.bn", depthwise=True)
    _set_conv_bn(keras_model, f"{name}_pwproj_conv", f"{name}_pwproj_bn", sd, f"{torch_prefix}.pw_proj.conv.weight", f"{torch_prefix}.pw_proj.bn", depthwise=False)
    if has_layer_scale:
        keras_model.get_layer(f"{name}_scale").set_weights([sd[f"{torch_prefix}.layer_scale.gamma"].numpy()])


def _set_mqa(keras_model, name, sd, torch_prefix, kv_stride):
    keras_model.get_layer(f"{name}_norm").set_weights(_bn_weights(sd, f"{torch_prefix}.norm"))
    keras_model.get_layer(f"{name}_q_proj").set_weights([_conv2d_weight(sd[f"{torch_prefix}.attn.query.proj.weight"])])

    for prefix, torch_kv in (("k", "key"), ("v", "value")):
        if kv_stride > 1:
            _set_conv_bn(
                keras_model, f"{name}_{prefix}_dw", f"{name}_{prefix}_dw_bn",
                sd, f"{torch_prefix}.attn.{torch_kv}.down_conv.weight", f"{torch_prefix}.attn.{torch_kv}.norm", depthwise=True,
            )
        keras_model.get_layer(f"{name}_{prefix}_proj").set_weights([_conv2d_weight(sd[f"{torch_prefix}.attn.{torch_kv}.proj.weight"])])

    keras_model.get_layer(f"{name}_out_proj").set_weights([_conv2d_weight(sd[f"{torch_prefix}.attn.output.proj.weight"])])
    keras_model.get_layer(f"{name}_scale").set_weights([sd[f"{torch_prefix}.layer_scale.gamma"].numpy()])


def convert(variant_key, input_size=224):
    config = MOBILENETV4_CONFIGS[variant_key]
    model_name = config["model_name"]
    stages = config["stages"]
    has_layer_scale = config.get("layer_scale_init") is not None

    torch_model = timm.create_model(TIMM_NAMES[variant_key], pretrained=True)
    torch_model.eval()
    sd = torch_model.state_dict()

    constructor = _build_mobilenetv4(**config)
    keras_model = constructor(include_top=False, weights=None, input_shape=(input_size, input_size, 3))

    _set_conv_bn(keras_model, f"{model_name}_stem_conv", f"{model_name}_stem_bn", sd, "conv_stem.weight", "bn1", depthwise=False)

    for stage_idx, blocks in enumerate(stages):
        for block_idx, spec in enumerate(blocks):
            kind = spec[0]
            name = f"{model_name}_stage{stage_idx}_block{block_idx}"
            torch_prefix = f"blocks.{stage_idx}.{block_idx}"

            if kind == "cn":
                _set_cn(keras_model, name, sd, torch_prefix)
            elif kind == "er":
                _set_er(keras_model, name, sd, torch_prefix)
            elif kind == "uir":
                _, out_channels, start_k, mid_k, expand = spec
                _set_uir(keras_model, name, sd, torch_prefix, start_k, mid_k, has_layer_scale)
            elif kind == "mqa":
                _, num_heads, key_dim, kv_stride = spec
                _set_mqa(keras_model, name, sd, torch_prefix, kv_stride)
            else:
                raise ValueError(f"Unknown block kind: {kind}")

    # timm's own stage 4 (`cn_r1_k1_s1_c960`, part of `self.blocks`, distinct
    # from the *post-pooling* `conv_head`/`norm_head` classifier-only step -
    # see `_backbone_mobilenetv4.py`'s module docstring) is this file's
    # `_head` step - the stage index right after the last entry in `stages`.
    _set_conv_bn(
        keras_model, f"{model_name}_head_conv", f"{model_name}_head_bn",
        sd, f"blocks.{len(stages)}.0.conv.weight", f"blocks.{len(stages)}.0.bn1", depthwise=False,
    )

    return keras_model, torch_model


def verify(keras_model, torch_model, input_size=224, seed=0):
    rng = np.random.default_rng(seed)
    x_np = rng.uniform(0, 1, size=(1, input_size, input_size, 3)).astype("float32")
    x_torch = torch.from_numpy(x_np.transpose(0, 3, 1, 2).copy())

    with torch.no_grad():
        # forward_features() returns the raw output of self.blocks - the
        # backbone feature this repo's `include_top=False` also returns, no
        # post-pooling conv_head/classifier involved (matching precedent:
        # FastViT's final_conv exclusion, plan.md §12.3).
        torch_out = torch_model.forward_features(x_torch).numpy()

    keras_out = keras_model.predict(x_np, verbose=0).transpose(0, 3, 1, 2)
    max_diff = np.abs(torch_out - keras_out).max()
    return max_diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="*", default=list(MOBILENETV4_CONFIGS.keys()))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "pretrained_weights", "mobilenetv4"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for variant in args.variants:
        print(f"=== {variant} ===")
        keras_model, torch_model = convert(variant)
        max_diff = verify(keras_model, torch_model)
        print(f"vs torch max abs diff: {max_diff:.4f}")
        if max_diff > 0.5:
            print(f"WARNING: {variant}'s max diff is unexpectedly large - do not trust this checkpoint without investigating.")

        out_path = os.path.join(args.out_dir, f"{MOBILENETV4_CONFIGS[variant]['model_name']}.weights.h5")
        keras_model.save_weights(out_path)
        print(f"saved: {out_path}")
