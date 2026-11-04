"""One-time PyTorch(timm) -> Keras weight conversion for MobileOne (plan.md
§12/§12.1/§12.4).

Not a runtime dependency of `swiss_army_keras` itself - see
`scripts/port_repvit_weights.py`'s docstring for the general "why" and the
install command for `torch`/`timm`.

MobileOne is the family with *no* existing Keras reference at all (§12's
table) - `_backbone_mobileone.py`'s architecture was built and verified
directly against the official PyTorch source, cross-checked against timm's
independent `byobnet.py`-based port, in an earlier pass of this plan (its
own module docstring references two bugs already found and fixed that way -
a stride/padding asymmetry and a depthwise-conv channel-count bug). This
porting pass found only the one remaining, expected bug shared by every
other backbone family ported so far:

1. **BN epsilon mismatch** (identical bug, identical fix, to RepViT's/
   FastViT's - plan.md §12.2/§12.3): defaulted to Keras' `1e-3`, reference
   uses PyTorch's `1e-5`. Fixed by changing the one default in the file.

Every other structural detail (per-pair depthwise+pointwise block split,
`num_conv_branches` count per variant, the SE placement rule - confirmed
here to apply to exactly the last 5-of-10 pairs of stage 3 plus all of
stage 4 for S4, and *only* S4 - and SE's `1/16` reduction ratio) matched the
reference exactly on the first attempt, a sign the earlier structural
verification pass was thorough.

Usage
----------
    python scripts/port_mobileone_weights.py [VARIANT ...]

    VARIANT: one or more of MobileOne-S0, MobileOne-S1, MobileOne-S2,
    MobileOne-S3, MobileOne-S4 (default: all five). Writes
    `<repo_root>/pretrained_weights/mobileone/<model_name>.weights.h5`.
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

from swiss_army_keras._backbone_mobileone import (
    MOBILEONE_CONFIGS, NUM_BLOCKS_PER_STAGE, _build_mobileone, _stage_channels, reparameterize_mobileone,
)

TIMM_NAMES = {
    "MobileOne-S0": "mobileone_s0",
    "MobileOne-S1": "mobileone_s1",
    "MobileOne-S2": "mobileone_s2",
    "MobileOne-S3": "mobileone_s3",
    "MobileOne-S4": "mobileone_s4",
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


def _set_conv_bn(keras_model, conv_name, bn_name, sd, torch_conv_prefix, torch_bn_prefix, depthwise):
    w = sd[f"{torch_conv_prefix}.weight"]
    kw = _dwconv2d_weight(w) if depthwise else _conv2d_weight(w)
    keras_model.get_layer(conv_name).set_weights([kw])
    keras_model.get_layer(bn_name).set_weights(_bn_weights(sd, torch_bn_prefix))


def _set_se(keras_model, our_name, sd, torch_prefix):
    keras_model.get_layer(f"{our_name}_reduce").set_weights([
        _conv2d_weight(sd[f"{torch_prefix}.fc1.weight"]), sd[f"{torch_prefix}.fc1.bias"].numpy(),
    ])
    keras_model.get_layer(f"{our_name}_expand").set_weights([
        _conv2d_weight(sd[f"{torch_prefix}.fc2.weight"]), sd[f"{torch_prefix}.fc2.bias"].numpy(),
    ])


def _set_mobileone_block(keras_model, our_name, sd, torch_prefix, num_conv_branches, has_scale, has_identity, has_se, depthwise):
    for i in range(num_conv_branches):
        _set_conv_bn(
            keras_model, f"{our_name}_conv{i}", f"{our_name}_conv{i}_bn",
            sd, f"{torch_prefix}.conv_kxk.{i}.conv", f"{torch_prefix}.conv_kxk.{i}.bn", depthwise=depthwise,
        )
    if has_scale:
        _set_conv_bn(
            keras_model, f"{our_name}_scale", f"{our_name}_scale_bn",
            sd, f"{torch_prefix}.conv_scale.conv", f"{torch_prefix}.conv_scale.bn", depthwise=depthwise,
        )
    if has_identity:
        keras_model.get_layer(f"{our_name}_identity_bn").set_weights(_bn_weights(sd, f"{torch_prefix}.identity"))
    if has_se:
        # timm names the SE module `attn` (`attn.fc1`/`attn.fc2`), not `se`.
        _set_se(keras_model, f"{our_name}_se", sd, f"{torch_prefix}.attn")


def convert(variant_key, input_size=224):
    config = MOBILEONE_CONFIGS[variant_key]
    width_multipliers = config["width_multipliers"]
    num_conv_branches = config["num_conv_branches"]
    use_se = config["use_se"]

    torch_model = timm.create_model(TIMM_NAMES[variant_key], pretrained=True)
    torch_model.eval()
    sd = torch_model.state_dict()

    constructor = _build_mobileone(**config)
    keras_model = constructor(include_top=False, weights=None, input_shape=(input_size, input_size, 3))

    model_name = config["model_name"]
    stem_channels, stage_channels = _stage_channels(width_multipliers)

    # Stem: num_conv_branches=1 always, kernel=3 (has_scale=True),
    # stride=2 (no identity), regular (non-depthwise) conv.
    _set_mobileone_block(keras_model, f"{model_name}_stem", sd, "stem", num_conv_branches=1, has_scale=True, has_identity=False, has_se=False, depthwise=False)

    prev_channels = stem_channels
    flat_idx = 0
    for stage_idx, (channels, num_blocks) in enumerate(zip(stage_channels, NUM_BLOCKS_PER_STAGE)):
        stage_name = f"{model_name}_stage{stage_idx + 1}"
        if stage_idx == 2:
            se_start = num_blocks - (num_blocks // 2)
        elif stage_idx == 3:
            se_start = 0
        else:
            se_start = num_blocks

        for pair_id in range(num_blocks):
            pair_se = use_se and pair_id >= se_start
            stride = 2 if pair_id == 0 else 1

            dw_name = f"{stage_name}_pair{pair_id}_dw"
            _set_mobileone_block(
                keras_model, dw_name, sd, f"stages.{stage_idx}.{flat_idx}",
                num_conv_branches=num_conv_branches, has_scale=True, has_identity=(stride == 1), has_se=pair_se, depthwise=True,
            )
            flat_idx += 1

            pw_name = f"{stage_name}_pair{pair_id}_pw"
            _set_mobileone_block(
                keras_model, pw_name, sd, f"stages.{stage_idx}.{flat_idx}",
                num_conv_branches=num_conv_branches, has_scale=False, has_identity=(prev_channels == channels), has_se=pair_se, depthwise=False,
            )
            flat_idx += 1

            prev_channels = channels
        flat_idx = 0

    return keras_model, torch_model


def verify(keras_model, torch_model, input_size=224, seed=0):
    rng = np.random.default_rng(seed)
    x_np = rng.uniform(0, 1, size=(1, input_size, input_size, 3)).astype("float32")
    x_torch = torch.from_numpy(x_np.transpose(0, 3, 1, 2).copy())

    with torch.no_grad():
        torch_out = torch_model.forward_features(x_torch).numpy()

    keras_out = keras_model.predict(x_np, verbose=0).transpose(0, 3, 1, 2)
    max_diff = np.abs(torch_out - keras_out).max()

    fused_model = reparameterize_mobileone(keras_model)
    fused_out = fused_model.predict(x_np, verbose=0).transpose(0, 3, 1, 2)
    fused_max_diff = np.abs(torch_out - fused_out).max()

    return max_diff, fused_max_diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="*", default=list(MOBILEONE_CONFIGS.keys()))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "pretrained_weights", "mobileone"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for variant in args.variants:
        print(f"=== {variant} ===")
        keras_model, torch_model = convert(variant)
        max_diff, fused_max_diff = verify(keras_model, torch_model)
        print(f"unfused vs torch max abs diff: {max_diff:.4f}")
        print(f"fused (reparameterize_mobileone) vs torch max abs diff: {fused_max_diff:.4f}")
        if max_diff > 0.5:
            print(f"WARNING: {variant}'s max diff is unexpectedly large - do not trust this checkpoint without investigating.")

        out_path = os.path.join(args.out_dir, f"{MOBILEONE_CONFIGS[variant]['model_name']}.weights.h5")
        keras_model.save_weights(out_path)
        print(f"saved: {out_path}")
