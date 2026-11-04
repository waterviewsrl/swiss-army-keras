"""One-time PyTorch(timm) -> Keras weight conversion for FastViT (plan.md
§12/§12.2).

Not a runtime dependency of `swiss_army_keras` itself - `torch`/`timm` are
needed only to *run this script*, once, per checkpoint (see
`scripts/port_repvit_weights.py`'s own docstring for the install command and
the general "why" of this separation).

Four real, previously-undiscovered bugs in `_backbone_fastvit.py` were found
and fixed while building this conversion - each caught only by actually
reading the reference source (`timm/models/fastvit.py`) line by line and
verifying numerically, not by inspection of our own code:

1. **BN epsilon mismatch** (same bug, same fix, as RepViT's - plan.md
   §12.2): defaulted to Keras' `1e-3` throughout, reference uses PyTorch's
   `1e-5`. Fixed by changing every default in the file.
2. **`rep_conv_block`'s "1x1 scale" branch was unconditional.** The
   reference (`MobileOneBlock`) only builds it when `kernel_size > 1`
   (`if kernel_size > 1 and use_scale_branch`) - for `kernel_size == 1`
   calls (the stem's 3rd conv, every downsample's pointwise projection),
   the reference has *no* second branch at all (a "1x1 scale branch" next
   to a `kernel_size=1` "kxk" branch would just be a redundant duplicate).
   This was silently adding an extra, nonexistent branch to those calls.
   Fixed by only building the 1x1 branch when `kernel_size > 1`.
3. **`rep_downsample`'s channel expansion was in the wrong sub-module.**
   The reference's downsample (`PatchEmbed`) expands `in_chs -> embed_dim`
   in its *first* step (a `groups=in_chs` grouped 7x7/3x3-fused conv -
   `ReparamLargeKernelConv(in_chs, embed_dim, groups=in_chs, ...)`), then
   keeps channels constant in its *second*, pointwise step. This repo had
   it backwards (channel-preserving depthwise first, channel-projecting
   pointwise second). Since `out_channels` is always an exact multiple of
   `in_channels` in every real config, the fix is
   `DepthwiseConv2D(depth_multiplier=out_channels // in_channels)` for the
   7x7/3x3 branches - confirmed against an isolated PyTorch grouped-conv
   repro (max abs diff ~5e-7) including the specific weight-reshape formula
   needed (`torch_weight.reshape(in_ch, multiplier, kH, kW)` *before*
   transposing - a plain transpose alone silently scrambles which output
   channels belong to which input channel). This also needed extending
   `reparam_utils.fuse_conv_bn`/`fuse_conv_bn_layers` to support BN fusion
   for a depthwise conv with `depth_multiplier > 1` (its per-channel BN
   params correspond to the *flattened* `(cin, multiplier)` pair, not
   either kernel axis alone - pass `channel_axis=(-2, -1)`).
4. **`fastvit_convffn`'s residual pointed at the wrong tensor.** It added
   the FFN's output back onto `mlp` (the intermediate depthwise-conv+BN
   result), not `x` (the function's own input, i.e. the *whole block's*
   input before that depthwise conv). The reference
   (`RepMixerBlock`/`AttentionBlock`.forward: `x = x + layer_scale(self.mlp(x))`,
   where `self.mlp` is the *entire* conv->fc1->act->fc2 chain) confirms the
   residual belongs on the pre-conv input - this bug silently dropped the
   positional depthwise-conv+BN step from every residual in the whole
   network (every block, RepMixer and attention alike). Fixed by changing
   the `Add(...)([y, mlp])` to `Add(...)([y, x])`.

Two more mappings were verified with isolated numerical repros before being
trusted, given past experience in this session with subtle, shape-matching-
but-numerically-wrong axis mistakes:
- `DepthwiseConv2D(depth_multiplier=N)`'s weight layout vs. PyTorch's
  `groups=in_ch, out_ch=N*in_ch` grouped conv (see bug 3 above).
- Keras `MultiHeadAttention`'s 4-kernel internal layout
  (query/key/value/attention_output, each `(input_dim, num_heads, key_dim)`
  or `(num_heads, key_dim, output_dim)`) vs. PyTorch's single combined
  `qkv = nn.Linear(dim, 3*dim, bias=False)` + `proj = nn.Linear(dim, dim)` -
  confirmed max abs diff ~1.4e-6 on an isolated repro with the mapping this
  script uses (`_qkv_kernel`/`_out_kernel` below).

Usage
----------
    python scripts/port_fastvit_weights.py [VARIANT ...]

    VARIANT: one or more of FastViT-T8, FastViT-T12, FastViT-S12,
    FastViT-SA12, FastViT-SA24, FastViT-SA36, FastViT-MA36 (default: all
    seven). Writes
    `<repo_root>/pretrained_weights/fastvit/<model_name>.weights.h5`.
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

from swiss_army_keras._backbone_fastvit import FASTVIT_CONFIGS, _build_fastvit, reparameterize_fastvit

TIMM_NAMES = {
    "FastViT-T8": "fastvit_t8",
    "FastViT-T12": "fastvit_t12",
    "FastViT-S12": "fastvit_s12",
    "FastViT-SA12": "fastvit_sa12",
    "FastViT-SA24": "fastvit_sa24",
    "FastViT-SA36": "fastvit_sa36",
    "FastViT-MA36": "fastvit_ma36",
}


def _conv2d_weight(w):
    return w.detach().numpy().transpose(2, 3, 1, 0)


def _dwconv2d_weight(w, depth_multiplier=1):
    """PyTorch grouped conv `(in_ch*multiplier, 1, kH, kW)` -> Keras
    DepthwiseConv2D `(kH, kW, in_ch, multiplier)`. `multiplier=1` is the
    plain depthwise case (a simple transpose); `multiplier>1` needs the
    reshape-before-transpose from this module's docstring, bug 3.
    """
    w = w.detach().numpy()
    out_ch = w.shape[0]
    in_ch = out_ch // depth_multiplier
    if depth_multiplier == 1:
        return w.transpose(2, 3, 0, 1)
    return w.reshape(in_ch, depth_multiplier, w.shape[2], w.shape[3]).transpose(2, 3, 0, 1)


def _bn_weights(sd, prefix):
    return [
        sd[f"{prefix}.weight"].numpy(), sd[f"{prefix}.bias"].numpy(),
        sd[f"{prefix}.running_mean"].numpy(), sd[f"{prefix}.running_var"].numpy(),
    ]


def _set_conv_bn(keras_model, conv_name, bn_name, sd, torch_conv_prefix, torch_bn_prefix, depthwise=False, depth_multiplier=1):
    w = sd[f"{torch_conv_prefix}.weight"]
    kw = _dwconv2d_weight(w, depth_multiplier) if depthwise else _conv2d_weight(w)
    keras_model.get_layer(conv_name).set_weights([kw])
    keras_model.get_layer(bn_name).set_weights(_bn_weights(sd, torch_bn_prefix))


def _set_mobileone_block(keras_model, our_name, sd, torch_prefix, kernel_size, depthwise, has_identity):
    """A `rep_conv_block` call: kxk always, 1x1 only if kernel_size > 1,
    identity (plain BN) if `has_identity`.
    """
    _set_conv_bn(
        keras_model, f"{our_name}_kxk", f"{our_name}_kxk_bn",
        sd, f"{torch_prefix}.conv_kxk.0.conv", f"{torch_prefix}.conv_kxk.0.bn", depthwise=depthwise,
    )
    if kernel_size > 1:
        _set_conv_bn(
            keras_model, f"{our_name}_1x1", f"{our_name}_1x1_bn",
            sd, f"{torch_prefix}.conv_scale.conv", f"{torch_prefix}.conv_scale.bn", depthwise=depthwise,
        )
    if has_identity:
        keras_model.get_layer(f"{our_name}_identity_bn").set_weights(_bn_weights(sd, f"{torch_prefix}.identity"))


def _set_convmlp(keras_model, our_name, sd, torch_prefix):
    """ConvMlp: pos_dw(7x7 depthwise, groups=in_chs)+BN, fc1(1x1, bias), fc2(1x1, bias)."""
    _set_conv_bn(
        keras_model, f"{our_name}_pos_dw", f"{our_name}_pos_bn",
        sd, f"{torch_prefix}.conv.conv", f"{torch_prefix}.conv.bn", depthwise=True,
    )
    keras_model.get_layer(f"{our_name}_fc1").set_weights([
        _conv2d_weight(sd[f"{torch_prefix}.fc1.weight"]), sd[f"{torch_prefix}.fc1.bias"].numpy(),
    ])
    keras_model.get_layer(f"{our_name}_fc2").set_weights([
        _conv2d_weight(sd[f"{torch_prefix}.fc2.weight"]), sd[f"{torch_prefix}.fc2.bias"].numpy(),
    ])


def _set_layer_scale(keras_model, layer_name, sd, torch_gamma_key):
    gamma = sd[torch_gamma_key].numpy().reshape(-1)  # torch (dim,1,1) -> keras (dim,)
    keras_model.get_layer(layer_name).set_weights([gamma])


def _set_repmixer_block(keras_model, our_name, sd, torch_prefix):
    # norm: a MobileOneBlock with only the identity (BN) branch.
    keras_model.get_layer(f"{our_name}_token_norm").set_weights(_bn_weights(sd, f"{torch_prefix}.token_mixer.norm.identity"))
    # mixer: kernel_size=3, depthwise, has identity (in==out, stride=1).
    _set_mobileone_block(keras_model, f"{our_name}_token_deep", sd, f"{torch_prefix}.token_mixer.mixer", kernel_size=3, depthwise=True, has_identity=True)
    _set_layer_scale(keras_model, f"{our_name}_token_scale", sd, f"{torch_prefix}.token_mixer.layer_scale.gamma")
    _set_convmlp(keras_model, f"{our_name}_convffn", sd, f"{torch_prefix}.mlp")
    _set_layer_scale(keras_model, f"{our_name}_convffn_scale", sd, f"{torch_prefix}.layer_scale.gamma")


def _qkv_kernel(w, dim, num_heads, head_dim):
    """PyTorch (out=dim, in=dim) -> Keras MHA (in=dim, num_heads, head_dim)."""
    return w.detach().numpy().T.reshape(dim, num_heads, head_dim)


def _out_kernel(w, dim, num_heads, head_dim):
    """PyTorch (out=dim, in=dim) -> Keras MHA attention_output (num_heads, head_dim, dim)."""
    return w.detach().numpy().T.reshape(num_heads, head_dim, dim)


def _set_attention_block(keras_model, our_name, sd, torch_prefix, dim, num_heads, head_dim=32):
    keras_model.get_layer(f"{our_name}_norm").set_weights(_bn_weights(sd, f"{torch_prefix}.norm"))

    qkv_w = sd[f"{torch_prefix}.token_mixer.qkv.weight"]
    q_w, k_w, v_w = torch.split(qkv_w, dim, dim=0)
    q_kernel = _qkv_kernel(q_w, dim, num_heads, head_dim)
    k_kernel = _qkv_kernel(k_w, dim, num_heads, head_dim)
    v_kernel = _qkv_kernel(v_w, dim, num_heads, head_dim)
    zero_bias = np.zeros((num_heads, head_dim), dtype="float32")
    out_kernel = _out_kernel(sd[f"{torch_prefix}.token_mixer.proj.weight"], dim, num_heads, head_dim)
    out_bias = sd[f"{torch_prefix}.token_mixer.proj.bias"].numpy()
    keras_model.get_layer(f"{our_name}_mhsa").set_weights([
        q_kernel, zero_bias, k_kernel, zero_bias, v_kernel, zero_bias, out_kernel, out_bias,
    ])

    _set_layer_scale(keras_model, f"{our_name}_scale", sd, f"{torch_prefix}.layer_scale_1.gamma")
    _set_convmlp(keras_model, f"{our_name}_convffn", sd, f"{torch_prefix}.mlp")
    _set_layer_scale(keras_model, f"{our_name}_convffn_scale", sd, f"{torch_prefix}.layer_scale_2.gamma")


def _set_cpe(keras_model, our_name, sd, torch_prefix):
    keras_model.get_layer(f"{our_name}_dw").set_weights([
        _dwconv2d_weight(sd[f"{torch_prefix}.pos_enc.weight"]), sd[f"{torch_prefix}.pos_enc.bias"].numpy(),
    ])


def _set_downsample(keras_model, our_name, sd, torch_prefix, in_channels, out_channels):
    depth_multiplier = out_channels // in_channels
    _set_conv_bn(
        keras_model, f"{our_name}_dw7", f"{our_name}_dw7_bn",
        sd, f"{torch_prefix}.proj.0.large_conv.conv", f"{torch_prefix}.proj.0.large_conv.bn", depthwise=True, depth_multiplier=depth_multiplier,
    )
    _set_conv_bn(
        keras_model, f"{our_name}_dw3", f"{our_name}_dw3_bn",
        sd, f"{torch_prefix}.proj.0.small_conv.conv", f"{torch_prefix}.proj.0.small_conv.bn", depthwise=True, depth_multiplier=depth_multiplier,
    )
    _set_mobileone_block(keras_model, f"{our_name}_proj", sd, f"{torch_prefix}.proj.1", kernel_size=1, depthwise=False, has_identity=True)


def convert(variant_key, input_size=256):
    config = FASTVIT_CONFIGS[variant_key]
    stage_channels = config["stage_channels"]
    stage_blocks = config["stage_blocks"]
    stage_types = config["stage_types"]

    torch_model = timm.create_model(TIMM_NAMES[variant_key], pretrained=True)
    torch_model.eval()
    sd = torch_model.state_dict()

    constructor = _build_fastvit(**config)
    keras_model = constructor(include_top=False, weights=None, input_shape=(input_size, input_size, 3))

    model_name = config["model_name"]

    # Stem: 3 MobileOneBlocks - conv1 (k=3,in=3,out=W,no identity),
    # conv2 (k=3,depthwise,no identity), conv3 (k=1,identity, no 1x1 branch).
    _set_mobileone_block(keras_model, f"{model_name}_stem_1", sd, "stem.0", kernel_size=3, depthwise=False, has_identity=False)
    _set_mobileone_block(keras_model, f"{model_name}_stem_2", sd, "stem.1", kernel_size=3, depthwise=True, has_identity=False)
    _set_mobileone_block(keras_model, f"{model_name}_stem_3", sd, "stem.2", kernel_size=1, depthwise=False, has_identity=True)

    prev_channels = stage_channels[0]
    for stage_idx, (channels, num_blocks, stage_type) in enumerate(zip(stage_channels, stage_blocks, stage_types)):
        stage_name = f"{model_name}_stage{stage_idx + 1}"

        if stage_idx > 0:
            _set_downsample(keras_model, f"{stage_name}_downsample", sd, f"stages.{stage_idx}.downsample", prev_channels, channels)

        if stage_type == "transformer":
            _set_cpe(keras_model, f"{stage_name}_cpe", sd, f"stages.{stage_idx}.pos_emb")
            num_heads = channels // 32
            for block_id in range(num_blocks):
                _set_attention_block(keras_model, f"{stage_name}_block{block_id}", sd, f"stages.{stage_idx}.blocks.{block_id}", channels, num_heads)
        else:
            for block_id in range(num_blocks):
                _set_repmixer_block(keras_model, f"{stage_name}_block{block_id}", sd, f"stages.{stage_idx}.blocks.{block_id}")

        prev_channels = channels

    return keras_model, torch_model


def verify(keras_model, torch_model, input_size=256, seed=0):
    """Compares against the raw *last-stage* feature (stem -> stages, no
    further processing) - NOT `torch_model.forward_features(x)`, which also
    applies a `final_conv` (a classifier-head-specific channel-doubling
    `MobileOneBlock`, `cls_ratio=2.0`, applied only ahead of global pooling
    for classification - confirmed by reading `FastVit.forward_features`
    directly). This repo's backbones expose the *stage* feature pyramid as
    their interface (`include_top=False` returns `stage_outputs[-1]`
    unmodified - see `_fastvit_network`), which is what
    `backbone_zoo`/every decoder in this repo actually consumes, so that's
    what's verified here. `final_conv` itself is a real, separate gap this
    conversion does *not* address (out of scope: it only matters for
    FastViT's own `include_top=True` classifier head being pretrained-
    weight-compatible, not for the backbone-as-feature-pyramid role this
    whole library is built around) - plan.md §12.2 documents this
    explicitly rather than silently ignoring it.
    """
    rng = np.random.default_rng(seed)
    x_np = rng.uniform(0, 1, size=(1, input_size, input_size, 3)).astype("float32")
    x_torch = torch.from_numpy(x_np.transpose(0, 3, 1, 2).copy())

    with torch.no_grad():
        torch_out = torch_model.stem(x_torch)
        for block in torch_model.stages:
            torch_out = block(torch_out)
        torch_out = torch_out.numpy()

    keras_out = keras_model.predict(x_np, verbose=0).transpose(0, 3, 1, 2)
    max_diff = np.abs(torch_out - keras_out).max()

    fused_model = reparameterize_fastvit(keras_model)
    fused_out = fused_model.predict(x_np, verbose=0).transpose(0, 3, 1, 2)
    fused_max_diff = np.abs(torch_out - fused_out).max()

    return max_diff, fused_max_diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="*", default=list(FASTVIT_CONFIGS.keys()))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "pretrained_weights", "fastvit"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for variant in args.variants:
        print(f"=== {variant} ===")
        keras_model, torch_model = convert(variant)
        max_diff, fused_max_diff = verify(keras_model, torch_model)
        print(f"unfused vs torch max abs diff: {max_diff:.4f}")
        print(f"fused (reparameterize_fastvit) vs torch max abs diff: {fused_max_diff:.4f}")
        if max_diff > 0.5:
            print(f"WARNING: {variant}'s max diff is unexpectedly large - do not trust this checkpoint without investigating.")

        out_path = os.path.join(args.out_dir, f"{FASTVIT_CONFIGS[variant]['model_name']}.weights.h5")
        keras_model.save_weights(out_path)
        print(f"saved: {out_path}")
