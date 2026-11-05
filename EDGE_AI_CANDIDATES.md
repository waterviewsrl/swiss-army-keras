# Edge AIoT candidates assessment (2026-11-05)

What else is worth adding to swiss-army-keras: networks, optimizers, training
and deployment strategies. The lens is deployment performance on three
hardware tiers:

- **Tier A, Axis camera DLPUs** (ARTPEC-8, ARTPEC-9; also ARTPEC-7, CV25, CV75). The reference target.
- **Tier B, other lightweight aarch64 stacks**: Cortex-A CPUs with LiteRT/XNNPACK, and small NPUs (Rockchip RKNN, NXP i.MX, Hailo, Edge TPU).
- **Tier C, NVIDIA Jetson Orin class** (TensorRT, FP16 or INT8).

Companion documents: [ASSESSMENT.md](ASSESSMENT.md) (state of the code),
[ROADMAP.md](ROADMAP.md) (ordered work list),
[docs/source/deployment.rst](docs/source/deployment.rst) (what export does today).

## How to read this document

**Evidence labels.** `[axis]` means read from the Axis developer
documentation or the Axis model zoo on 2026-11-05. `[measured]` means
measured in this repository on 2026-11-05 (TensorFlow 2.21, random weights,
128x128 input, per-tensor full-integer TFLite). `[web]` means found by a web
search the same day, usually vendor or secondary material. Everything
unlabelled comes from the published literature as I know it up to mid-2026
and should be checked against the paper before work starts. Accuracy and
latency figures of unlabelled items are deliberately omitted.

**Verdicts.**

| Verdict | Meaning |
|---|---|
| **Adopt now** | High value for Tier A, low risk, fits the current code |
| **Adopt later** | Worth doing after the "now" items, or valuable mainly for Tier B/C |
| **Watch** | Promising but unproven for these targets, or blocked by something else |
| **Skip** | Poor fit for the targets, or better consumed from upstream than re-implemented |

**Effort.** S is up to 2 days, M is up to 2 weeks, L is more.

---

## 1. Summary

1. **The biggest gains are in deployment tooling, not in new networks.** The
   library exports MobileOne with its training-time branches (416 TFLite
   operators instead of 50 `[measured]`), always quantizes per channel while
   ARTPEC-8 wants per tensor `[axis]`, and has no check that an exported
   graph only uses operators the target accelerates.
2. **Axis DLPUs reward plain networks.** Axis recommends regular 3x3
   convolutions over depthwise ones, ReLU or ReLU6, and channel counts in
   multiples of 6 (ARTPEC-8) or 16 (ARTPEC-9) `[axis]`. The published
   ARTPEC-9 operator list has no GELU, hard-swish or softmax, and multiplies
   a tensor only by a constant `[axis]`. Squeeze-excite gates, attention
   fusion modules and transformer blocks therefore risk CPU fallback.
3. **Most of what the library added in 2026 is tuned for phone CPUs, not for
   DLPUs.** RepViT, FastViT and StrideFormer rely on GELU, squeeze-excite or
   attention; PP-LiteSeg, PIDNet and LR-ASPP fuse features with sigmoid
   gates. MobileNetV4-Conv, reparameterized MobileOne (S0 to S3), MobileNetV2
   and EfficientNet-Lite are the families that match the Axis guidance.
4. **The highest-value new networks are old and plain**: ResNet-18/34, STDC,
   MobileDet-style fused blocks, a gate-free decoder, and an NMS-free
   detection head.
5. **For Tier C, do not re-implement transformer detectors in Keras.** Use
   the upstream PyTorch implementations on Orin, and use them as teachers
   and auto-labellers for the small models this library is good at.

The recommended order is in section 6.

> **Status update, later on 2026-11-05.** Step 1 of section 6 is implemented
> except for the raw-output detection export: deployment profiles,
> per-tensor export, the operator check and fusion of nested backbones are in
> `swiss_army_keras/deploy_utils.py`. The work also showed that QAT-wrapped
> heads cannot be exported cleanly for integer-only targets (a float
> operator and per-channel weights remain), so accelerator profiles now
> strip the wrappers and quantize post-training. That makes the "PTQ-only
> pipeline evaluation" of section 4.1 more pressing: on Axis targets the
> library already is a PTQ pipeline. Sections 3 and 4 describe the state
> before this change.
>
> **Second update, same day.** Cross-layer equalization, an error
> measurement and gate-free variants of LR-ASPP and PP-LiteSeg are
> implemented. Measurements on ImageNet-pretrained backbones (table in
> `docs/source/deployment.rst`) refine section 3: MobileNetV4-Conv-S is the
> only custom backbone that keeps good feature quality after post-training
> quantization, and with equalization it is the best per-tensor choice.
> Fused MobileOne and RepViT have the right operators but quantize poorly
> (noise as large as signal at the deepest tap, even per channel), and
> MobileNetV2 collapses per tensor because ReLU6 blocks equalization. This
> raises the priority of QARepVGG-style blocks and of better activation
> calibration, and it means the "Good" ratings of fused MobileOne and
> MobileNetV2 in the section 3 table hold for operator support only.

---

## 2. What the targets reward

### 2.1 Axis DLPU family

| Chip | Model format | Quantization | Guidance from Axis `[axis]` |
|---|---|---|---|
| ARTPEC-7 | `.tflite` compiled for the Edge TPU | int8 | Extra Edge TPU compile step |
| ARTPEC-8 | `.tflite` | int8 or uint8, **per tensor** ("performs much better than per-channel") | Prefer regular Conv2D over DepthwiseConv2D; 3x3 kernels; stride 2 where possible (native), otherwise pooling; filter counts in multiples of 6; ReLU over leaky ReLU because the DLPU skips exact zeros; fixed tensor sizes |
| ARTPEC-9 | `.tflite` | int8 or uint8; per-channel only for weights and biases of a limited set of operators | Prefer ReLU6; avoid explicit padding; channels are processed in groups of 16; batch size 1; at most 4 dimensions; NHWC |
| CV25, CV75 | `.tflite` or `.onnx` converted to a native binary | toolchain-specific | Conversion toolchain is not public; Axis rates ease of use "hard" |

Operator support, as published `[axis]`:

- **ARTPEC-8** lists, among others, Conv2d, DepthwiseConv2d, TransposeConv,
  pooling, Concatenation, ResizeBilinear, ResizeNearestNeighbor, Add, Mul,
  Relu, Relu6, HardSwish, Gelu, Softmax, FullyConnected. The page as read
  does not list LOGISTIC (sigmoid), MEAN, PAD, LEAKY_RELU, PRELU or
  BATCH_MATMUL.
- **ARTPEC-9** lists a shorter, heavily constrained set. Conv2D and
  DepthwiseConv2D with kernels 1, 2, 3, 5, 7, 9 and stride 1 or 2 (depth
  multiplier 1 only). TransposeConv with stride 2 only. AveragePool only
  3x3 stride 1, or global on a 7x7 or 8x8 map. Mean only as that same global
  reduction. **Mul only by a per-channel constant.** Resize with output size
  2n or 2n-1. Concat and Split with channel counts in multiples of 16. Pad up
  to 7 per side. Sigmoid, Tanh, LeakyReLU, ReLU present. No HardSwish, Gelu,
  Softmax, PReLU or BatchMatMul.

These two lists were extracted from the web pages by an automated reader and
the two chips' lists differ in surprising ways. Treat them as a checklist to
verify with `larod-client` on a device, not as ground truth. Anything a DLPU
does not support runs on the camera CPU.

**Reference latencies from the Axis model zoo** (AXIS OS 12.11.72 for
detection) `[axis]`:

| Model | ARTPEC-8 (Q1656-LE) | ARTPEC-9 (Q1728) | mAP |
|---|---|---|---|
| MobileNetV2 classifier | 5.4 ms | 2.0 ms | n/a |
| SSD MobileNetV2 | 17.8 ms | 10.5 ms | 25.6 |
| SSDLite MobileDet | 28.2 ms | 21.5 ms | 32.9 |
| YOLOv5n (Axis variant) | 55.6 ms | 31.9 ms | 23.5 / 23.3 |
| YOLOv5s (Axis variant) | 70.0 ms | 35.0 ms | 32.3 / 32.2 |
| YOLOv5m (Axis variant) | 95.7 ms | 42.7 ms | 37.9 / 38.1 |

Three lessons follow from this table.

- **Width and depth are cheap, overhead is not.** On ARTPEC-9, YOLOv5m is only
  34% slower than YOLOv5n for 14.8 mAP more. On ARTPEC-8 the step costs 72%.
  A slightly larger plain network usually beats a clever small one.
- **Architecture match dominates.** On ARTPEC-8, SSDLite MobileDet reaches
  the accuracy of YOLOv5s in 40% of the time. MobileDet was searched for
  accelerators and uses regular-convolution fused blocks.
- **Budget anchor.** A 30 fps pipeline leaves about 33 ms per frame. On
  ARTPEC-8 that excludes every YOLOv5 variant at the benchmarked resolution.

### 2.2 Other aarch64 stacks (Tier B)

| Stack | What it rewards | Notes |
|---|---|---|
| Cortex-A CPU, LiteRT with XNNPACK | Depthwise-separable networks, per-channel int8, small inputs | The inverse of the DLPU preference. MobileNetV4, MobileOne, RepViT and FastViT are designed for this |
| Rockchip RK3588 NPU (RKNN) | int8, operators on its support list | Needs conversion to `.rknn` on an x86 host; unsupported operators fall back to the CPU silently `[web]` |
| NXP i.MX 8M Plus, Hailo-8, Edge TPU | int8, mostly plain CNN operators | Vendor compilers consume TFLite or ONNX. Not verified in this pass |

`tf.lite` inside TensorFlow is in maintenance mode. The interpreter moved to
the `ai_edge_litert` package, and post-training quantization for existing
`.tflite` files is offered by a separate tool, LiteRT Quantizer, with
selective and int16-activation recipes `[web]`.

### 2.3 Jetson Orin (Tier C)

TensorRT consumes ONNX. FP16 is the safe default. INT8 needs calibration or
Q/DQ nodes, does not always beat FP16, and the DLA has extra restrictions on
quantization-aware models `[web]`. Attention, GELU, layer normalization and
large kernels are all fine here. The constraint on Orin is engineering time,
not operator support.

---

## 3. Where the library stands against these targets

Operators in per-tensor full-integer exports `[measured]`:

| Model | Operators beyond convolutions, ADD and CONCAT | Risk on a DLPU |
|---|---|---|
| MobileOne-S0 backbone, training-time | 183 ADD, 35 MUL, 5 PAD (416 operators in total) | Exported unfused: several times slower than necessary |
| MobileOne-S0 backbone, reparameterized | 5 PAD (50 operators in total) | Clean except explicit padding |
| `classifier` + MobileNetV4-Conv-S | MEAN, FULLY_CONNECTED, LOGISTIC, MUL, SOFTMAX, 5 PAD | Head uses swish (sigmoid times tensor) and softmax |
| `classifier` + RepViT-M0.9 | 27 GELU, 11 LOGISTIC, 11 MEAN, 34 MUL | GELU and squeeze-excite throughout |
| `lr_aspp` + MobileOne-S0 | RESIZE_BILINEAR, AVERAGE_POOL, LOGISTIC, MUL, two float islands | Sigmoid gate; float islands |
| `pp_liteseg` + MobileOne-S0 | 4 MEAN, 4 REDUCE_MAX, 2 LOGISTIC, SUB, MUL, 6 RESIZE_BILINEAR | Attention fusion (UAFM) |
| `deeplab_v3_plus` + MobileNetV2 | 3 SPACE_TO_BATCH_ND and 3 BATCH_TO_SPACE_ND, one float island | Dilated convolutions |

Fit of the backbone families already in the registry:

| Family | Tier A | Tier B CPU | Tier C | Why |
|---|---|---|---|---|
| MobileNetV2, EfficientNet-Lite | Good | Good | Fine | ReLU6, no squeeze-excite. Both are in the Axis model zoo `[axis]` |
| MobileNetV3Large (minimalistic) | Good | Good | Fine | The minimalistic variant drops squeeze-excite and hard-swish |
| MobileNetV4-Conv | Good | Very good | Good | ReLU family, fused blocks in the early stages, no attention |
| MobileOne S0 to S3, reparameterized | Good, if fused before export | Very good | Good | Plain convolutions after fusion. Fused kernels need care under per-tensor quantization |
| ResNet50 and larger | Fair (heavy but plain) | Poor | Good | Regular convolutions and ReLU are what Axis asks for |
| MobileOne-S4, MobileNetV4-Hybrid | Poor | Good | Good | Squeeze-excite or attention |
| RepViT, FastViT, StrideFormer | Poor | Very good | Good | GELU or hard-swish, squeeze-excite, attention |
| EfficientNet B0 to B7, DenseNet | Poor | Fair | Fair | Swish and squeeze-excite; dense concatenations |

---

## 4. Candidates

### 4.1 Deployment and quantization tooling

| Candidate | What it is | A | B | C | Effort | Verdict |
|---|---|---|---|---|---|---|
| **Deployment profiles** | `Quantizer(profile=...)` presets: per-tensor or per-channel, integer I/O type, channel-multiple check, reparameterization step | High | High | Med | M | **Adopt now** |
| **Operator linter** | List the operators of an exported `.tflite` and compare with a per-target allow-list kept as a dated data file; fail the export or the test when something would fall back | High | High | Low | S | **Adopt now** |
| **Reparameterize inside full models** | Fuse a MobileOne, RepViT or FastViT backbone nested in a decoder or classifier before export | High | High | High | M | **Adopt now** |
| **Cross-layer equalization and bias correction** | Rescale consecutive layers so per-channel weight ranges even out (Nagel et al., 2019). The standard remedy for per-tensor quantization of depthwise and fused networks | High | Low | Low | M | **Adopt now** |
| **Quantization debugger report** | Wrap `tf.lite.experimental.QuantizationDebugger` to rank layers by quantization error, per profile | High | Med | Low | S | **Adopt now** |
| **Edge-clean builder switches** | `edge_activation='relu6'` (replace GELU, hard-swish, swish), `use_se=False`, channel rounding to a multiple, SAME padding instead of explicit PAD, gate-free decoder variants. Needs fine-tuning after the swap | High | Low | Low | M | **Adopt now** |
| **Backbone QAT after fusion** | Today only heads are QAT-wrapped; backbone activations are calibrated post-training. Optionally wrap plain-convolution backbones and fine-tune briefly after fusion | High | Med | Low | L | Adopt later |
| **Raw-output detection export** | Export `*_base` outputs and ship a reference NumPy or C decoder; keep decoding and NMS off the accelerator | High | High | Med | S | **Adopt now** |
| **On-device benchmark harness** | Run `larod-client` over SSH the way the Axis model zoo does `[axis]`; LiteRT benchmark on aarch64; `trtexec` on Orin. Publish a backbone by decoder by resolution table | High | High | High | M | **Adopt now** |
| **ONNX export path** | `tf2onnx` from the Keras model or from the `.tflite`, for TensorRT, RKNN and Hailo | None | High | High | M | Adopt later |
| **LiteRT migration** | Use `ai_edge_litert` for the interpreter; keep `tf.lite.TFLiteConverter` for conversion `[web]` | Low | Med | Low | S | Adopt later |
| **PTQ-only pipeline evaluation** | Measure plain post-training quantization plus equalization (and LiteRT Quantizer recipes) against today's QAT on two real tasks. If the gap is small, QAT can become optional, which removes the dependency that pins the library to legacy Keras 2 | High (strategic) | High | Med | M | **Adopt now** (as an experiment) |
| AdaRound, BRECQ, QDrop | Learned rounding for post-training quantization | Med | Med | Low | L | Watch: not available in the TFLite converter |
| LSQ, PACT (learned quantizer ranges) | QAT with trainable step sizes | Med | Low | Low | L | Watch: fragile to implement on `tfmot` |
| 16-bit activations | TFLite 16x8 mode | None | Low | Low | S | Skip for Tier A: DLPUs take int8 or uint8 only `[axis]` |
| Unstructured weight pruning | `tfmot` magnitude pruning | Unknown | None | None | S | Skip, with one exception below |

One cheap experiment is specific to ARTPEC-8. Axis states that the DLPU skips
exact zeros, which is why ReLU is preferred `[axis]`. An activation-sparsity
penalty during fine-tuning (an L1 term on post-ReLU feature maps) might
therefore reduce latency without changing the graph. This is speculative and
needs one afternoon on a device to confirm or discard.

### 4.2 Backbones

| Candidate | Why it matters | A | B | C | Effort | Verdict |
|---|---|---|---|---|---|---|
| **ResNet-18 and ResNet-34** | Regular 3x3 convolutions and ReLU only: the closest match to Axis's guidance, and the default backbone of most NPU vendors. Not available in `tf.keras.applications`. ImageNet weights can be ported with the existing script pattern | High | Med | High | S | **Adopt now** |
| **STDC1 and STDC2** (Fan et al., 2021) | Convolution, batch norm, ReLU and concatenation only. The native backbone of PP-LiteSeg, which the library already has without it | High | Med | High | S to M | **Adopt now** |
| **MobileDet-style backbone** (Xiong et al., 2021) | Searched for accelerators; fused regular-convolution blocks. The best accuracy per millisecond in the Axis zoo on ARTPEC-8 `[axis]` | High | Med | Med | M | **Adopt now** |
| **QARepVGG-style blocks** (Chu et al., 2022) | Plain 3x3 convolution and ReLU after fusion, designed so the fused network survives 8-bit quantization, which the original RepVGG does not. Also a diagnostic for MobileOne, RepViT and FastViT, which share the fusion mechanism | High | Med | High | M | **Adopt now** |
| **Vendored EfficientNet-Lite** | Already supported in principle, but the external package fails to import on Python 3.12. Axis benchmarks EfficientNet-Lite0 and Lite2 `[axis]`. Move the few hundred lines into the library | High | High | Low | S | **Adopt now** |
| **MobileNetV3Small (minimalistic)** | One registry entry; a 2 to 3 ms classifier on a DLPU | High | High | Low | S | **Adopt now** |
| CSP backbone with ReLU6 | The Axis YOLOv5 recipe generalized: cross-stage-partial blocks with regular convolutions | High | Med | High | M | Adopt later, together with the detection head work |
| FasterNet (Chen et al., 2023) | Partial convolution: a regular 3x3 on a fraction of the channels. Built for latency on CPU and GPU | Med | High | High | M | Adopt later |
| EfficientViT (Cai et al., 2023) | ReLU linear attention with strong TensorRT results; has segmentation variants | None | Med | High | L | Adopt later, Tier C only |
| PP-HGNetV2 | Regular-convolution backbone used by RT-DETR and D-FINE | Med | Low | High | M | Watch: weights exist in Paddle format only |
| SHViT, EfficientFormerV2, MobileViT, EdgeNeXt | Mobile hybrid transformers | None | Med | Med | M each | Skip: RepViT, FastViT and MobileNetV4-Hybrid already cover this niche |
| GhostNetV2 and V3, StarNet | Depthwise-heavy; StarNet multiplies two feature maps | Poor | Med | Low | M | Skip |
| ConvNeXtV2 atto to pico | Layer norm, GELU, 7x7 depthwise | None | Low | Med | M | Skip |
| LSNet, RepNeXt and other 2025 designs | Recent CPU-latency designs | Unknown | Unknown | Unknown | n/a | Watch: no deployment evidence found for these targets |

### 4.3 Semantic segmentation

| Candidate | Why it matters | A | B | C | Effort | Verdict |
|---|---|---|---|---|---|---|
| **Gate-free convolutional decoder** | A feature-pyramid head with only 1x1 and 3x3 convolutions, 2x resizes, and add or concat. Four of the six decoder families (LR-ASPP, PP-LiteSeg, PIDNet, PP-MobileSeg) gate features with sigmoids, DeepLabV3+ uses dilation, and DDRNet's context module pools with large windows. This would be the first decoder designed to be DLPU-clean | High | High | Med | S | **Adopt now** |
| **Gate-free options on existing decoders** | `lr_aspp(gate=False)`; PP-LiteSeg with plain addition instead of UAFM; pooling-free context module | High | Med | Low | S | **Adopt now** |
| **Training-only auxiliary branches** | Keep the deployed graph plain and add supervision that is discarded at export: deep-supervision heads (already present for DDRNet and PIDNet), boundary heads, or a transformer branch aligned to the CNN features as in SCTNet (Xu et al., 2024) | High | High | Med | M | Adopt later |
| BiSeNetV2, STDC-Seg | Classic real-time designs; their fusion modules use sigmoid gates like PP-LiteSeg | Med | High | High | M | Skip: STDC plus the gate-free decoder covers the need |
| SegNeXt, SeaFormer, TopFormer, AFFormer | Convolutional or mobile attention | Poor | Med | High | M each | Skip: PP-MobileSeg covers the niche |
| SegFormer-B0, EfficientViT-Seg | Transformer segmenters | None | Low | High | L | Adopt later with EfficientViT, Tier C only |
| Fast-SCNN | Very light, but depends on multi-size pooling | Med | High | Low | S | Skip |
| Matting networks for privacy masking (MODNet, RVM, PP-Matting) | Soft person masks | Low | Med | High | L | Watch |

### 4.4 Object detection

| Candidate | Why it matters | A | B | C | Effort | Verdict |
|---|---|---|---|---|---|---|
| **NMS-free one-to-one head** | A second head branch trained with one-to-one assignment, as in YOLOv10 and YOLO26, so inference needs only a top-k. Removes NMS from the camera CPU. The library already has ATSS, SimOTA and DSLA assigners to build on | High | High | High | M | **Adopt now** |
| **FCOS-lite as the documented DLPU default** | It regresses distances directly. GFL-lite decodes a distribution with a softmax, which the ARTPEC-9 list does not include `[axis]`. YOLO26 dropped that distribution loss for the same export reason `[web]` | High | Med | Low | S | **Adopt now** (documentation and a test) |
| **Detection data loader and training loop** | The heads exist but cannot be trained end to end with the library. COCO-format reader, in-pipeline target assignment, mosaic and copy-paste augmentation | High | High | High | L | **Adopt now** (already item 4 of ASSESSMENT.md) |
| Heatmap head (CenterNet, TTFNet style) | NMS-free by 3x3 max-pooling, fully convolutional; also the basis for keypoints and counting | High | High | Med | M | Adopt later |
| Decoupled head and CSP neck with ReLU6 (YOLOX, RTMDet style) | A stronger neck than PAFPN-lite, in regular convolutions | High | Med | High | M | Adopt later |
| SSD-style anchor head | Proven in the Axis zoo, but relies on a TFLite custom post-processing operator | Med | Med | Low | M | Skip: the anchor-free heads cover it |
| RT-DETR, D-FINE, DEIM, RF-DETR | NMS-free transformer detectors; the accuracy leaders at real-time speeds on GPU `[web]` | None | Low | High | L | **Skip as implementations.** Run upstream PyTorch on Orin; use them as teachers and auto-labellers |
| YOLO-World, YOLOE (open vocabulary) | Text-prompted detection | None | None | Med | L | Skip; use for auto-labelling |

### 4.5 Other heads and tasks

| Candidate | Why it matters | A | B | C | Effort | Verdict |
|---|---|---|---|---|---|---|
| **Multi-task model builder** | One backbone, several heads (segmentation, classification, image quality) in one pass. A camera has one accelerator and usually one model slot per stream. Add a loss-balancing helper (uncertainty weighting) | High | High | Med | M | **Adopt now** |
| **Temporal input** | Stack two or three frames, or add a frame-difference channel, at the input. Smoke and water are dynamic textures, so motion is a strong cue. The graph stays plain. Needs stem inflation for pretrained weights and loader support for frame tuples | High | High | High | M | **Adopt now** |
| Coordinate-classification keypoint head (SimCC, RTMPose style) | Keypoints as two 1-D classifications; quantizes better than heatmaps and is more accurate than direct regression | High | High | High | M | Adopt later |
| Dense anomaly detection (PaDiM style) | Per-location Gaussian statistics on backbone features. `ood_utils.dense_mahalanobis_distance` already provides most of it; needs a recipe and a test. Use cases: tampering, lens obstruction, unusual scene state | Med | High | High | S | Adopt later |
| EfficientAD (Batzner et al., 2024) | Student-teacher anomaly detection with small convolutional networks | Med | High | High | M | Watch |
| Tiled inference (SAHI style) | Slice a high-resolution frame, run the model per tile, merge. Multiplies accelerator passes | Med | Med | High | S | Adopt later |
| Online temporal shift (TSM) | Feature sharing across frames; needs cached features as extra model inputs and outputs | Low | Med | High | L | Watch |
| Tracking (ByteTrack, OC-SORT) | Association logic on the CPU, no network | n/a | n/a | n/a | n/a | Out of scope for the library; the re-identification head already supports it |

### 4.6 Optimizers and schedules

The library has more exotic optimizers than basic training infrastructure.
The gaps are in the basics.

| Candidate | Why it matters | Effort | Verdict |
|---|---|---|---|
| **Learning-rate schedules** | Warm-up plus cosine and one-cycle helpers, wired into `TrainingDriver`. Today the second phase reuses the first phase's constant rate | S | **Adopt now** |
| **Differential learning rate** | Lower rate for the pretrained backbone than for the head in the unfrozen phase. Recommended by `use_case_recipes.md`, not implemented | S | **Adopt now** |
| **Weight averaging: EMA and SWA** | Stabilizes the final checkpoint on small, noisy datasets and helps under domain shift. The Keras optimizer base class already has an EMA option; what is missing is using the averaged weights for validation, checkpoints and export, and re-estimating batch-norm statistics | S to M | **Adopt now** |
| **AdamW and SGD presets with correct weight decay** | Decoupled decay, none on batch-norm and bias parameters | S | **Adopt now** |
| **Gradient clipping in the recipes** | Built into Keras optimizers; QAT fine-tuning is the phase that needs it | S | **Adopt now** (documentation) |
| ASAM (adaptive SAM) | Scale-invariant variant of the SAM wrapper already present. Flat minima also tolerate quantization noise better, which matters under per-tensor quantization | S | Adopt later: first measure the existing `SAM` during QAT fine-tuning |
| Schedule-Free AdamW and SGD (Defazio et al., 2024) | No schedule to tune; needs batch-norm recalibration | M | Adopt later |
| MuSGD | Hybrid of SGD and Muon introduced with YOLO26; evidence is vendor material only `[web]` | M | Watch |
| Prodigy, D-Adaptation | Learning-rate-free | M | Watch |
| Adan, Lookahead, LARS, LAMB, Sophia | No clear benefit for small CNN fine-tuning | n/a | Skip |

### 4.7 Training strategies

| Candidate | Why it matters | Effort | Verdict |
|---|---|---|---|
| **Knowledge distillation wrapper** | A `Distiller` model with logit, feature and channel-wise (dense prediction) distillation. Two uses: a larger float teacher for the small student, and the float model as teacher of its own quantized copy during QAT | M | **Adopt now** |
| **Foundation models as off-device teachers** | Pseudo-labels from large segmentation, detection and depth models, following the pattern `iqa_utils` already uses for image quality. Turns unlabelled camera footage into training data | S per task | **Adopt now** (recipes, not library code) |
| **Rare-class and domain-balanced sampling** | Guarantees positives in each batch for small targets. Recommended by `use_case_recipes.md`, not implemented | M | **Adopt now** |
| **Non-square inputs** | Train and deploy at the sensor's aspect ratio (for example 640x384) instead of a square. About 40% fewer pixels at the same horizontal resolution. The data loaders mix up width and height today | S | **Adopt now** (a bug fix) |
| **Camera-realistic augmentation presets** | Day and night (monochrome infrared mode), noise at high gain, wide-dynamic-range tone curves, rain and fog, lens dirt. albumentations already provides the transforms; the library should ship presets | S | **Adopt now** |
| **Loss-weight ramping callback** | Needed by the boundary-loss recipes; not implemented | S | **Adopt now** |
| **Active-learning loop** | Use the energy and Mahalanobis scores the library already computes to select frames worth labelling | M | Adopt later |
| Structured channel pruning with fine-tuning | Shrinks width where the task allows. Width is cheap on ARTPEC-9, so the gain there is small | L | Adopt later |
| Mosaic, MixUp, copy-paste | Standard detection augmentation; comes with the detection loader | n/a | With 4.4 |
| Semi-supervised training (FixMatch, mean teacher) | Largely superseded by foundation-model pseudo-labels for these tasks | L | Skip |
| Neural architecture search | A width and resolution sweep on the benchmark harness gives most of the benefit | L | Skip |
| Test-time adaptation | A frozen int8 graph on a DLPU cannot adapt | n/a | Skip |

### 4.8 System-level strategies

These live in the camera application more than in the library. They often
matter more than the choice of network, so the documentation should mention
them.

- **Do not run the network when nothing changed.** Gate inference on the camera's motion detection, or run every n-th frame and interpolate.
- **Crop before inferring.** A region of interest at native resolution beats the full frame downscaled.
- **Train on what the accelerator sees.** The DLPU receives frames from the image pipeline, not decoded video. If only recordings are available for training, add compression augmentation.
- **Calibrate thresholds on the quantized model.** A sigmoid output on ARTPEC-9 has a fixed scale of 1/256 `[axis]`; thresholds tuned on the float model shift.
- **Keep post-processing cheap.** Top-k instead of NMS, argmax on the host at reduced resolution, integer arithmetic where possible.

---

## 5. What not to build

- **Transformer detectors and open-vocabulary models in Keras.** High effort, legacy Keras 2, and the upstream implementations run well on the only tier that can use them.
- **More mobile hybrid transformers.** The library has three families for that niche and none of them suits a DLPU.
- **More attention-fusion decoders.** Four of six decoders already gate with sigmoids.
- **More exotic optimizers** before schedules, weight averaging and differential learning rates exist.

---

## 6. Recommended sequence

| Step | Content | Tier | Effort |
|---|---|---|---|
| 1 | Deployment profiles, per-tensor export, operator linter with dated allow-lists, reparameterization of nested backbones, raw-output detection export | A, B | M |
| 2 | On-device benchmark harness; publish the first backbone by decoder by resolution table for ARTPEC-8 and ARTPEC-9 | A | M |
| 3 | Cross-layer equalization, bias correction, quantization debugger report; per-tensor accuracy tests | A | M |
| 4 | Edge-clean switches (activation swap, no squeeze-excite, channel rounding, SAME padding); gate-free decoder and gate-free options | A | M |
| 5 | Plain backbones: ResNet-18/34, STDC1/2, vendored EfficientNet-Lite, MobileNetV3Small, then MobileDet-style and QARepVGG-style | A, C | M |
| 6 | Training basics: schedules, differential learning rate, EMA and SWA, sampling, non-square inputs, augmentation presets, ramping callback | all | M |
| 7 | Detection end to end: data loader, training loop, NMS-free head | all | L |
| 8 | Distillation wrapper and teacher recipes; multi-task builder; temporal input | all | M |
| 9 | PTQ-only evaluation feeding the Keras 3 decision; LiteRT migration | strategic | M |
| 10 | Tier C: ONNX and TensorRT export, EfficientViT | C | L |

Steps 1 to 4 change what the existing models cost on a camera and need no new
architecture. They should come before any item of step 5 or later, with one
exception: ResNet-18 and STDC are small enough to add alongside step 4, and
they give the benchmark harness something DLPU-shaped to measure.

---

## 7. Sources

Axis (read 2026-11-05):

- [Axis DLPU overview](https://developer.axis.com/computer-vision/computer-vision-on-device/axis-dlpu)
- [ARTPEC-8 model conversion](https://developer.axis.com/computer-vision/on-device/system-on-chip/artpec-8/model-conversion/), [optimization tips](https://developer.axis.com/computer-vision/on-device/system-on-chip/artpec-8/optimization-tips/), [supported operators](https://developer.axis.com/computer-vision/on-device/system-on-chip/artpec-8/supported-ops/)
- [ARTPEC-9 model conversion](https://developer.axis.com/computer-vision/on-device/system-on-chip/artpec-9/model-conversion/), [optimization tips](https://developer.axis.com/computer-vision/on-device/system-on-chip/artpec-9/optimization-tips/), [supported operators](https://developer.axis.com/computer-vision/on-device/system-on-chip/artpec-9/supported-ops/)
- [Axis model zoo](https://github.com/AxisCommunications/axis-model-zoo)

Other web material (read 2026-11-05):

- [LiteRT migration guide](https://developers.google.com/edge/litert/migration), [LiteRT repository](https://github.com/google-ai-edge/LiteRT), [LiteRT Quantizer](https://github.com/google-ai-edge/ai-edge-quantizer)
- [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26), [YOLO26 overview paper](https://arxiv.org/html/2509.25164v1)
- [NVIDIA: YOLOv5 on Jetson Orin with cuDLA, QAT to inference](https://developer.nvidia.com/blog/deploying-yolov5-on-nvidia-jetson-orin-with-cudla-quantization-aware-training-to-inference/), [NVIDIA forum: INT8 without gain over FP16](https://forums.developer.nvidia.com/t/jetson-thor-int8-quantization-show-no-performance-gain-over-fp16/353308)
- [RF-DETR architecture search paper](https://arxiv.org/pdf/2511.09554), [Roboflow comparison (vendor)](https://blog.roboflow.com/rf-detr-vs-alternatives/)
- [Rockchip RKNPU2 repository](https://github.com/rockchip-linux/rknpu2/blob/master/README.md)
- [TensorFlow Model Optimization repository](https://github.com/tensorflow/model-optimization)

Measurements in this repository: the per-tensor operator inventories and the
MobileOne reparameterization comparison were produced on 2026-11-05 with the
project environment; the method is described in
[docs/source/deployment.rst](docs/source/deployment.rst).

Papers named in the tables (not re-read for this document): Nagel et al.
2019 (data-free quantization); Fan et al. 2021 (STDC); Xiong et al. 2021
(MobileDet); Chu et al. 2022 (QARepVGG); Chen et al. 2023 (FasterNet); Cai et
al. 2023 (EfficientViT); Xu et al. 2024 (SCTNet); Batzner et al. 2024
(EfficientAD); Defazio et al. 2024 (schedule-free optimization).
