"""One-time PyTorch(timm) -> Keras weight conversion for RepViT (plan.md
§12/§12.1).

Not a runtime dependency of `swiss_army_keras` itself - `torch`/`timm` are
needed only to *run this script*, once, per checkpoint. Install them into a
throwaway/dev environment (`pip install torch --index-url
https://download.pytorch.org/whl/cpu timm`), run this script, then the
`.weights.h5` files it produces are all `swiss_army_keras` itself ever
needs (loaded via plain Keras `Model.load_weights`, no `torch`/`timm`
import required at load time).

Usage
----------
    python scripts/port_repvit_weights.py [VARIANT ...]

    VARIANT: one or more of RepViT-M0.9, RepViT-M1.0, RepViT-M1.1,
    RepViT-M1.5, RepViT-M2.3 (default: all five). Writes
    `<repo_root>/pretrained_weights/repvit/<model_name>.weights.h5`.

Maps timm's `repvit_m*` checkpoints (default, UNFUSED `.state_dict()` - no
`.fuse()` call) onto `swiss_army_keras`'s `RepViT_*` backbones
(`deploy=False`, the default), which use the identical unfused RepVGGDW
structure (3x3 dw+BN branch, 1x1 dw+bias branch, identity, trailing BN) -
confirmed by reading both `_backbone_repvit.py` and timm's `repvit.py`
source directly, not guessed.

Three real, previously-undiscovered numerical bugs were found and fixed in
`_backbone_repvit.py` while building and verifying this conversion (full
account in plan.md §12.1) - this script's own numerical-verification step
(comparing against the PyTorch reference on identical input) is what caught
all three, not code review:

1. **BN epsilon mismatch.** `_backbone_repvit.py` defaulted to Keras'
   `epsilon=1e-3` throughout; the actual reference model (both timm's port
   and, per its own source, the official THU-MIG/RepViT repo) uses
   PyTorch's default `epsilon=1e-5`. Silent - no shape error, just wrong
   numbers - and the public backbone constructors didn't even expose a way
   to override it. Fixed by changing the default to `1e-5`.
2. **`padding='same'` + `strides=2` does not match PyTorch's explicit
   symmetric padding**, for any even-sized feature map (confirmed with a
   minimal, isolated single-conv repro: identical weights and input, ~0.85
   mean absolute difference in the output purely from this). Keras' SAME
   padding rule for a strided conv can be *asymmetric* (e.g. 0 top/left, 1
   bottom/right for a 3x3/stride-2 conv), while PyTorch's `padding=1` is
   always symmetric. Fixed by replacing `padding='same'` with an explicit
   `ZeroPadding2D(1)` + `padding='valid'` on the three affected stride-2
   convs (the stem's two convs, `repvit_downsample`'s depthwise conv) -
   verified to reduce a whole-backbone max absolute output difference from
   ~11 down to ~0.05 (mean ~0.0033) after fixing both this and the epsilon
   bug together (this alone, i.e. correctly matching the reference
   architecture, is what makes correct weight-loading possible at all -
   with SAME padding, no possible weight values would have matched the
   reference output).
3. **Not a bug, but a real, easy-to-get-wrong finding**: timm's
   `stages[i].downsample.pre_block` is *not* an extra layer relative to the
   official flat `cfgs` block list - it is the official list's last regular
   block of the *preceding* channel-width group, which timm's stage/
   downsample class split relabels into the *next* stage's downsample
   bucket (verified by summing block counts against the official cfgs list
   fetched directly from github.com/THU-MIG/RepViT). `swiss_army_keras`'s
   own `stage_blocks` config already accounts for this correctly (its last
   block per stage *is* that block) - this script's mapping just needs to
   know about the relabeling to find the right timm state-dict key for it
   (see `_torch_block_prefix` below).

The ~0.05 max absolute / ~0.003 mean absolute remaining difference (out of
an output with std ~1.9) is not further reducible without changing
`swiss_army_keras.activations.GELU`, which deliberately uses the tanh
approximation (a legitimate, widely-used GELU variant, not itself a bug) -
PyTorch's default `nn.GELU()` (used by the reference model) is the exact/
erf-based formula. This residual is well within normal cross-framework
floating-point tolerance for a full forward pass and is expected to vanish
under any downstream fine-tuning.
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

from swiss_army_keras._backbone_repvit import REPVIT_CONFIGS, _build_repvit, reparameterize_repvit

TIMM_NAMES = {
    "RepViT-M0.9": "repvit_m0_9",
    "RepViT-M1.0": "repvit_m1_0",
    "RepViT-M1.1": "repvit_m1_1",
    "RepViT-M1.5": "repvit_m1_5",
    "RepViT-M2.3": "repvit_m2_3",
}


def _conv2d_weight(w):
    """PyTorch (out, in, kH, kW) -> Keras Conv2D (kH, kW, in, out)."""
    return w.detach().numpy().transpose(2, 3, 1, 0)


def _dwconv2d_weight(w):
    """PyTorch depthwise (channels, 1, kH, kW) -> Keras DepthwiseConv2D
    (kH, kW, channels, 1)."""
    return w.detach().numpy().transpose(2, 3, 0, 1)


def _bn_weights(sd, prefix):
    return [
        sd[f"{prefix}.weight"].numpy(),
        sd[f"{prefix}.bias"].numpy(),
        sd[f"{prefix}.running_mean"].numpy(),
        sd[f"{prefix}.running_var"].numpy(),
    ]


def _set_conv_bn(keras_model, conv_name, bn_name, sd, torch_conv_prefix, torch_bn_prefix, depthwise=False):
    w = sd[f"{torch_conv_prefix}.weight"]
    kw = _dwconv2d_weight(w) if depthwise else _conv2d_weight(w)
    keras_model.get_layer(conv_name).set_weights([kw])
    keras_model.get_layer(bn_name).set_weights(_bn_weights(sd, torch_bn_prefix))


def _set_token_mixer(keras_model, name, sd, torch_prefix):
    """RepVGGDW: dw3x3+BN branch, dw1x1+bias branch, trailing BN."""
    _set_conv_bn(
        keras_model, f"{name}_token_dw3x3", f"{name}_token_dw3x3_bn",
        sd, f"{torch_prefix}.conv.c", f"{torch_prefix}.conv.bn", depthwise=True,
    )
    keras_model.get_layer(f"{name}_token_dw1x1").set_weights([
        _dwconv2d_weight(sd[f"{torch_prefix}.conv1.weight"]),
        sd[f"{torch_prefix}.conv1.bias"].numpy(),
    ])
    keras_model.get_layer(f"{name}_token_bn").set_weights(_bn_weights(sd, f"{torch_prefix}.bn"))


def _set_se(keras_model, name, sd, torch_prefix):
    keras_model.get_layer(f"{name}_reduce").set_weights([
        _conv2d_weight(sd[f"{torch_prefix}.fc1.weight"]), sd[f"{torch_prefix}.fc1.bias"].numpy(),
    ])
    keras_model.get_layer(f"{name}_expand").set_weights([
        _conv2d_weight(sd[f"{torch_prefix}.fc2.weight"]), sd[f"{torch_prefix}.fc2.bias"].numpy(),
    ])


def _set_channel_mixer(keras_model, name, sd, torch_prefix):
    """FFN: pw1(conv+BN)+GELU, pw2(conv+BN)."""
    _set_conv_bn(keras_model, f"{name}_ffn_pw1", f"{name}_ffn_pw1_bn", sd, f"{torch_prefix}.conv1.c", f"{torch_prefix}.conv1.bn")
    _set_conv_bn(keras_model, f"{name}_ffn_pw2", f"{name}_ffn_pw2_bn", sd, f"{torch_prefix}.conv2.c", f"{torch_prefix}.conv2.bn")


def _set_regular_block(keras_model, our_name, sd, torch_block_prefix, has_se):
    _set_token_mixer(keras_model, our_name, sd, f"{torch_block_prefix}.token_mixer")
    if has_se:
        _set_se(keras_model, f"{our_name}_se", sd, f"{torch_block_prefix}.se")
    _set_channel_mixer(keras_model, our_name, sd, f"{torch_block_prefix}.channel_mixer")


def _set_downsample(keras_model, our_name, sd, torch_downsample_prefix):
    """spatial_downsample (dw stride2 + BN) -> channel_downsample (pw + BN) -> ffn."""
    _set_conv_bn(
        keras_model, f"{our_name}_dw3x3", f"{our_name}_dw3x3_bn",
        sd, f"{torch_downsample_prefix}.spatial_downsample.c", f"{torch_downsample_prefix}.spatial_downsample.bn",
        depthwise=True,
    )
    _set_conv_bn(
        keras_model, f"{our_name}_pw", f"{our_name}_pw_bn",
        sd, f"{torch_downsample_prefix}.channel_downsample.c", f"{torch_downsample_prefix}.channel_downsample.bn",
    )
    _set_channel_mixer(keras_model, our_name, sd, f"{torch_downsample_prefix}.ffn")


def convert(variant_key, input_size=224):
    """Build a `swiss_army_keras` RepViT backbone (`weights=None`) for
    `variant_key` and load timm's pretrained checkpoint into it. Returns
    `(keras_model, torch_model)` for verification.
    """
    config = REPVIT_CONFIGS[variant_key]
    stage_channels = config["stage_channels"]
    stage_blocks = config["stage_blocks"]

    torch_model = timm.create_model(TIMM_NAMES[variant_key], pretrained=True)
    torch_model.eval()
    sd = torch_model.state_dict()

    constructor = _build_repvit(**config)
    keras_model = constructor(include_top=False, weights=None, input_shape=(input_size, input_size, 3))

    model_name = config["model_name"]

    _set_conv_bn(keras_model, f"{model_name}_stem_conv1", f"{model_name}_stem_conv1_bn", sd, "stem.conv1.c", "stem.conv1.bn")
    _set_conv_bn(keras_model, f"{model_name}_stem_conv2", f"{model_name}_stem_conv2_bn", sd, "stem.conv2.c", "stem.conv2.bn")

    for stage_idx, (channels, num_blocks) in enumerate(zip(stage_channels, stage_blocks)):
        stage_name = f"{model_name}_stage{stage_idx + 1}"

        if stage_idx > 0:
            _set_downsample(keras_model, f"{stage_name}_downsample", sd, f"stages.{stage_idx}.downsample")

        for block_id in range(num_blocks):
            our_name = f"{stage_name}_block{block_id}"
            se_ratio = 0.25 if (block_id % 2 == 0 and block_id != num_blocks - 1) else 0.0
            has_se = se_ratio > 0

            is_last_block = block_id == num_blocks - 1
            has_next_stage = stage_idx < len(stage_channels) - 1

            if is_last_block and has_next_stage:
                # Relabeled into the NEXT stage's downsample.pre_block in
                # timm - see this module's docstring, finding 3.
                torch_block_prefix = f"stages.{stage_idx + 1}.downsample.pre_block"
            else:
                torch_block_prefix = f"stages.{stage_idx}.blocks.{block_id}"

            _set_regular_block(keras_model, our_name, sd, torch_block_prefix, has_se)

    return keras_model, torch_model


def verify(keras_model, torch_model, input_size=224, seed=0):
    """Compare the converted Keras backbone against the PyTorch reference on
    identical random input. Returns the max absolute difference (also
    checks the fused/deploy-mode model via `reparameterize_repvit`, since
    that's this library's own code path, not just the raw conversion).
    """
    rng = np.random.default_rng(seed)
    x_np = rng.uniform(0, 1, size=(1, input_size, input_size, 3)).astype("float32")
    x_torch = torch.from_numpy(x_np.transpose(0, 3, 1, 2).copy())

    with torch.no_grad():
        torch_out = torch_model.forward_features(x_torch).numpy()

    keras_out = keras_model.predict(x_np, verbose=0).transpose(0, 3, 1, 2)
    max_diff = np.abs(torch_out - keras_out).max()

    fused_model = reparameterize_repvit(keras_model)
    fused_out = fused_model.predict(x_np, verbose=0).transpose(0, 3, 1, 2)
    fused_max_diff = np.abs(torch_out - fused_out).max()

    return max_diff, fused_max_diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="*", default=list(REPVIT_CONFIGS.keys()))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "pretrained_weights", "repvit"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for variant in args.variants:
        print(f"=== {variant} ===")
        keras_model, torch_model = convert(variant)
        max_diff, fused_max_diff = verify(keras_model, torch_model)
        print(f"unfused vs torch max abs diff: {max_diff:.4f}")
        print(f"fused (reparameterize_repvit) vs torch max abs diff: {fused_max_diff:.4f}")
        if max_diff > 0.5:
            print(f"WARNING: {variant}'s max diff is unexpectedly large - do not trust this checkpoint without investigating.")

        out_path = os.path.join(args.out_dir, f"{REPVIT_CONFIGS[variant]['model_name']}.weights.h5")
        keras_model.save_weights(out_path)
        print(f"saved: {out_path}")
