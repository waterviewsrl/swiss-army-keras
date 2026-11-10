# Backbone and head candidates by task (2026-11-10)

A second research pass, organized by task. The first pass,
[EDGE_AI_CANDIDATES.md](EDGE_AI_CANDIDATES.md) of 2026-11-05, is organized by
kind of work (tooling, backbones, heads, optimizers, strategies) and stays
valid for the hardware tiers, the Axis constraints, optimizers and training
strategies. This document replaces its sections 4.2 to 4.5 where the two
disagree, and adds the tasks the first pass covered thinly: depth,
keypoints, anomaly detection, re-identification, counting and video.

Companion documents: [ASSESSMENT.md](ASSESSMENT.md) section 13 (state of the
code on the same day, including the PIDNet review), [ROADMAP.md](ROADMAP.md)
(ordered work list).

> **Status, later on 2026-11-10.** Implemented since this document was
> written: the seven export repairs of section 2 (and a fixed batch at
> conversion, which also removes the DeepLab operators), the DDRNet and
> PIDNet repairs, the ResNet-18/34, STDC, QARepVGG, cross-stage-partial and
> MobileNetV3Small backbones, the plain pyramid head, the
> coordinate-classification keypoint head, the density head, the
> re-identification training wrapper, the affine-invariant depth losses and
> learned multi-task loss balancing. The tables below describe the state
> before that work. Still open from this document: GCNet, the detection
> loader and NMS-free head, ZipDepth, EfficientAD, the out-of-distribution
> additions, and pretrained weights for everything new.

## How to read this document

**Hardware tiers** (as in the first pass). A: Axis camera DLPUs (ARTPEC-8,
ARTPEC-9). B: other aarch64 devices, on the CPU with LiteRT or on a small
NPU. C: NVIDIA Jetson Orin class.

**Evidence labels.**

| Label | Meaning |
|---|---|
| `[measured]` | Measured in this repository on 2026-11-10: each model built with random weights at 256x256, QAT wrappers stripped, converted to per-tensor full-integer TFLite, checked against both Axis operator lists, and run once. The backbone is MobileNetV4-Conv-S unless the network brings its own |
| `[axis]` | Read from the Axis developer pages or the Axis model zoo on 2026-11-10 |
| `[web]` | Found by web search on 2026-11-10. Figures come from the cited paper or vendor page and were not reproduced here |
| unlabelled | From the published literature as I know it up to mid-2026. Check the paper before starting work |

The Axis operator lists in `deploy_utils.py` are still unverified on a
device. Every "off the list" statement below inherits that caveat.

**Verdicts.** Adopt now, Adopt later, Watch, Skip, with the meaning of the
first pass. Effort: S is up to 2 days, M up to 2 weeks, L more.

---

## 1. Summary

1. **The cheapest wins are in heads the library already has.** The export
   sweep found seven heads that leave the accelerator for one small,
   avoidable reason each (an exponential, a slice, a global mean, a dynamic
   reshape, a pooling power). Section 2 lists them. Each is about a day of
   work and none needs a new architecture.
2. **For segmentation on a DLPU, DDRNet is the network to stand on, not
   PIDNet.** DDRNet-23-slim is the only segmentation network in the library
   whose export has no operator off the ARTPEC-9 list `[measured]`, and
   Hailo ships it with a loss of 0.4 mIoU after quantization `[web]`.
   PIDNet-S needs 11 off-list operators on ARTPEC-8 and multiplies two
   tensors eight times `[measured]`.
3. **The best new segmentation candidate is GCNet** (CVPR 2025). It is a
   DDRNet-style network whose blocks fuse into single 3x3 convolutions at
   inference, with no attention, and it reports higher accuracy and speed
   than PIDNet and DDRNet without ImageNet pre-training `[web]`. The library
   already has the fusion machinery. The open question is quantization: the
   library's own measurements show that fused networks quantize poorly
   post-training.
4. **Depth has a clear recipe now.** Distill a foundation model into a small
   convolutional network. ZipDepth (July 2026) does exactly this with 6.1M
   parameters and has already been added to the Hailo model zoo `[web]`.
   The Small model of Depth Anything V2 is Apache-2.0 and can serve as the
   teacher `[web]`. The library has the heads, loader and metrics; it lacks
   the affine-invariant loss and the recipe.
5. **Detection guidance is unchanged, with better numbers.** On ARTPEC-9 the
   Axis YOLOv5 models now run in 32 to 43 ms under AXIS OS 12.11.72, 25%
   faster than the figures published for AXIS OS 12.6.85, and the medium
   model costs only a third more than the nano `[axis]`. Width is cheap on that chip. A ReLU6
   cross-stage-partial backbone with an NMS-free head remains the target.
6. **New backbones matter less than plain ones.** Nothing published in 2025
   or 2026 that I found changes the first pass: ResNet-18/34, STDC and a
   ReLU6 cross-stage-partial backbone are what the Axis and Hailo zoos run.
   The one upgrade is PP-HGNetV2, now the base of the smallest DEIMv2
   detectors `[web]`.
7. **Use the large models as teachers.** SAM 3 for masks, Depth Anything for
   depth, DEIMv2 or RF-DETR for boxes, DINOv3 for features. Their licenses
   differ and two of them are custom; section 5 has the table.

---

## 2. Where the library's own heads leave the accelerator

Post-training export of every head family `[measured]`. "Off list" counts
operators that the published list of the chip does not contain. Explicit
padding (`PAD`, 5 per model) comes from the MobileNetV4 backbone and is
left out of the ARTPEC-8 column to show what the head adds; it is off the
ARTPEC-8 list in every row that uses that backbone.

| Model | Operators | Off list, ARTPEC-8 (without `PAD`) | Off list, ARTPEC-9 | Other ARTPEC-9 findings |
|---|---|---|---|---|
| `ddrnet` (23-slim) | 113 | none (3 `PAD` of its own) | none | 6 resizes not by 2 |
| `lr_aspp(gate=False)` | 67 | none | none | 2 resizes not by 2 |
| `pp_liteseg(fusion='add')` | 84 | none | none | 3 resizes not by 2 |
| `fastdepth_head`, sigmoid dropped | 83 | none | none | none |
| `classifier(activation='relu')`, softmax dropped | 64 | `MEAN` 1 | none | none |
| `classifier` (default swish) | 66 | `MEAN` 1, `LOGISTIC` 1 | none | 1 tensor-by-tensor `MUL` |
| `multilabel_classifier`, `ordinal_classifier` | 66, 67 | `MEAN` 1, `LOGISTIC` 1 | none | 1 tensor-by-tensor `MUL` |
| `lr_aspp` | 72 | `LOGISTIC` 1 | none | 1 tensor-by-tensor `MUL` |
| `gfl_lite_detection_base` | 139 | none | `STRIDED_SLICE` 6 | none |
| `fcos_lite_detection_base` | 126 | `EXP` 3 | `EXP` 3 | none |
| `fcos_lite_instance_segmentation_base` | 140 | `EXP` 3 | `EXP` 3 | none |
| `pose_regression_head` | 71 | `MEAN` 1, `LOGISTIC` 2, `SHAPE` 1, `PACK` 1 | `SHAPE`, `STRIDED_SLICE`, `PACK` | 1 tensor-by-tensor `MUL` |
| `reid_embedding_head` | 75 | `MEAN`, `LOGISTIC`, `L2_NORMALIZATION`; 2 `POW` left in floating point | `POW` 2, `MINIMUM`, `MAXIMUM`, `L2_NORMALIZATION` | export has float operators |
| `deeplab_v3_plus`, `deeplab_v3_plus_lite` | 91, 90 | `SPACE_TO_BATCH_ND` 3, `BATCH_TO_SPACE_ND` 3 | the same 6 | resizes not by 2 |
| `pp_liteseg` | 106 | `MEAN` 4, `LOGISTIC` 2, `SUB` 2 | `REDUCE_MAX` 4, `SUB` 2 | 4 tensor-by-tensor `MUL` |
| `pidnet` (S) | 154 | `SUM` 2, `LOGISTIC` 3, `SUB` 3 (3 `PAD` of its own) | `SUM` 2, `SUB` 3 | 8 tensor-by-tensor `MUL`, 8 resizes not by 2 |
| `pp_mobileseg` on StrideFormer-Tiny | 334 | 54 (`MEAN`, `LOGISTIC`, `BATCH_MATMUL`) | 52 (`HARD_SWISH`, `BATCH_MATMUL`, `SOFTMAX`, ...) | 14 tensor-by-tensor `MUL` |
| `fcos_lite_detection` (decoding and NMS in the graph) | n/a | does not export this way | | `strip_qat` fails on the decoding operations |

All nineteen models that convert also run on the LiteRT CPU runtime.

What the table suggests, each about a day of work:

| Fix | Affects |
|---|---|
| Replace the `exp` on FCOS distances by a ReLU on a scaled linear output, or export the value before the `exp` | both FCOS models |
| Produce the four GFL sides with four convolutions or a `SPLIT` instead of slices | GFL on ARTPEC-9 |
| Offer `AveragePooling2D` with a fixed window instead of `GlobalAveragePooling2D` | every classifier on ARTPEC-8 |
| Build the pose output with a static reshape; export logits | pose head |
| For integer targets, pool with an average and leave the L2 normalization to the application | re-identification head, image-quality head |
| `dilation=False` option for the DeepLab pyramid (Hailo ships exactly this variant and reports 262 against 121 frames per second `[web]`) | both DeepLab builders |
| Make `strip_qat` handle the full detectors, or document that only the `*_base` models export for accelerators | detection |

---

## 3. Backbones

Fresh evidence first. The Axis model zoo, updated for AXIS OS 12.11.72
`[axis]`:

| Chip (device) | MobileNetV2 224 | SSD MobileNetV2 | SSDLite MobileDet | YOLOv5 n / s / m at 640 |
|---|---|---|---|---|
| ARTPEC-9 (Q1728) | 2.0 ms | 10.5 ms | 21.5 ms | 31.9 / 35.0 / 42.7 ms |
| ARTPEC-8 (Q1656-LE) | 5.4 ms | 17.8 ms | 28.2 ms | 55.6 / 70.0 / 95.7 ms |
| ARTPEC-8 (P1465-LE) | 9.9 ms | 28.1 ms | 38.5 ms | 100.1 ms (n only) |

Reported accuracies: MobileNetV2 69% top-1; SSD MobileNetV2 25.6, MobileDet
32.9, YOLOv5 n / s / m 23.3 / 32.2 / 38.1 mAP. The zoo also lists
EfficientNet-Lite0 and Lite2 on the Ambarella chips and a YOLOv8n at 11.5 ms
on CV75. The same ARTPEC-9 YOLOv5 rows read 42.6 / 45.5 / 53.2 ms under AXIS
OS 12.6.85: a quarter of the latency went away with firmware.

Two things follow. On ARTPEC-9 a model three times larger costs a third
more time, so fixed overhead dominates and wider networks are nearly free.
On ARTPEC-8 latency follows model size, and the device variant matters as
much as the model.

| Candidate | Tier A | Tier B | Tier C | Effort | Verdict |
|---|---|---|---|---|---|
| **MobileNetV4-Conv** (in the library) | Good; best measured per-tensor quality with equalization | Very good | Good | none | Keep as the default |
| **ResNet-18 and ResNet-34** | High. Hailo's FCN on ResNet-18 loses 0.1 mIoU after quantization `[web]` | Med | High | S | **Adopt now** (unchanged) |
| **STDC1 and STDC2** | High. Hailo ships STDC1 at 74.6 mIoU float, 73.8 quantized `[web]` | Med | High | S to M | **Adopt now** (unchanged) |
| **Cross-stage-partial backbone with ReLU6** (the YOLOv5 pattern) | High: it is what Axis trains and publishes for both chips `[axis]` | Med | High | M | **Adopt now**, with the detection work (raised from "later") |
| **MobileDet-style backbone** | High: 32.9 mAP at 21.5 ms on ARTPEC-9 `[axis]` | Med | Med | M | Adopt now (unchanged) |
| **Vendored EfficientNet-Lite**, MobileNetV3Small | High | High | Low | S | Adopt now (unchanged) |
| **GCNet block** (Yang et al., CVPR 2025) | High in principle: one 3x3 convolution and one activation per block after fusion, no residual addition `[web]` | Med | High | M | **Adopt as an experiment**: build it with the existing fusion utilities, then measure its post-training quantization before relying on it |
| **QARepVGG-style blocks** | High | Med | High | M | Adopt now (unchanged). GCNet makes it more relevant: both are plain after fusion and both need to survive 8 bits |
| **PP-HGNetV2** | Med: regular convolutions | Low | High | M | **Adopt later** (raised from "watch"): DEIMv2 builds its four smallest detectors, 0.5M to 3.6M parameters, on a pruned HGNetV2-B0 `[web]` |
| FasterNet | Med | High | High | M | Adopt later (unchanged) |
| EfficientViT | None | Med | High | L | Adopt later, Tier C only (unchanged) |
| DINOv3 ConvNeXt-Tiny and Small (distilled, 29M and 50M parameters) | None | Low | High | M | **Watch**, as teacher or frozen Tier C backbone. Custom DINOv3 License with gated access `[web]`; read it first |
| LSNet (CVPR 2025) | Unknown; built on large-kernel depthwise convolutions, throughput reported on a desktop GPU only `[web]` | Unknown | Unknown | M | Watch (unchanged) |
| MobileNetV5 | None | None | Low | n/a | Skip: it is the 300M-parameter vision encoder of Gemma 3n, ten times the largest MobileNetV4 `[web]`, not an edge classifier backbone |
| RepNeXt, StarNet, SHViT, GhostNetV3, more mobile hybrids | Poor | Med | Low | M each | Skip (unchanged) |

---

## 4. Heads by task

### 4.1 Classification and attributes

The library is well covered. Candidates are refinements.

| Candidate | Why | Effort | Verdict |
|---|---|---|---|
| **Accelerator-clean classifier preset** | ReLU hidden layer, fixed-window average pooling, logits out. Today the default uses swish and a global mean (section 2) | S | **Adopt now** |
| **Multi-attribute head** | Several softmax groups on one feature (vehicle type, colour, direction). `multi_task` already allows several classifier heads; a single head with grouped outputs is cheaper | S | Adopt later |
| Logit-adjusted loss for long-tailed classes | One line on top of cross-entropy; helps rare classes without resampling | S | Adopt later |
| Margin losses (ArcFace, CosFace) | See re-identification below; also useful for fine-grained classification | S | With 4.7 |

### 4.2 Semantic segmentation

Vendor evidence from the Hailo model zoo on Cityscapes at 1024x2048
`[web]`, float and quantized mIoU: DDRNet-23-slim 76.7 and 76.3;
PP-LiteSeg-B 78.4 and 78.4; PP-LiteSeg-T 76.4 and 76.0; STDC1 74.6 and 73.8;
BiSeNetV2 73.0 and 72.6; FCN on ResNet-18 69.4 and 69.3; SegFormer-B0 69.8
and 67.8; YOLO26 semantic models 77.4 and 75.9 (n, 1.6M parameters), 80.0
and 79.5 (s). Convolutional networks lose at most about one point; the
transformer loses two.

Published comparison from the GCNet paper, trained without ImageNet
weights, validation mIoU and frames per second on an A100 at 1024x2048
`[web]`: GCNet-S 77.3 at 193; DDRNet-23-slim 76.3 at 166; PIDNet-S 76.4 at
129; STDC1 71.8 at 184; BiSeNetV2 73.6 at 132; GCNet-M 79.0 at 105;
PIDNet-M 78.2 at 78; DDRNet-23 78.0 at 106.

| Candidate | Why | A | B | C | Effort | Verdict |
|---|---|---|---|---|---|---|
| **DDRNet as the documented accelerator default** | Cleanest export in the library (section 2), small quantization loss at a vendor. Needs the same repairs as PIDNet: free input sizes, the reference's batch-norm layout in the pooling module, a weight port (ASSESSMENT.md section 13) | High | High | High | S | **Adopt now** |
| **GCNet** | DDRNet-like, fused to plain 3x3 convolutions, keeps DDRNet's pooling module, no attention, no pre-training needed | High | High | High | M | **Adopt next**, after the DDRNet repairs, with a quantization measurement as the gate |
| **DeepLab without dilation** | Removes the six off-list operators; one option | High | Med | Low | S | **Adopt now** |
| **Plain pyramid semantic head** (the "gate-free decoder" of the first pass) | The YOLO26 semantic models show that a simple head on a detection backbone reaches DDRNet-level accuracy with 1.6M parameters and quantizes well `[web]`. Their head is not documented in what I read, and the code is AGPL: take the idea, not the code. Doubles as the segmentation head of `multi_task` next to a detector | High | High | High | S | **Adopt now** (confirmed) |
| Boundary auxiliary head for DDRNet | What PIDNet adds over DDRNet, as a training-only branch: the deployed graph stays DDRNet. The boundary losses exist | High | High | High | S | Adopt later |
| SCTNet | Training-only transformer branch; 79.0 mIoU at 117 frames per second in the GCNet table `[web]` | High | High | Med | M | Adopt later (unchanged) |
| PIDNet (in the library) | Accuracy between DDRNet and GCNet, but gated fusion in three places | Low | Med | High | none | Keep for Tiers B and C; do not build a gate-free variant |
| RDRNet (2024) | Same authors as GCNet, superseded by it | n/a | n/a | n/a | n/a | Skip |
| BiSeNetV2 | Quantizes well but ran at 9.7 frames per second on Hailo-8 against 38 for DDRNet-23-slim `[web]` | Low | Med | Med | M | Skip (confirmed) |
| SegFormer-B0 and other transformer segmenters | Largest quantization loss in the vendor table | None | Low | High | L | Skip for A and B (confirmed) |

### 4.3 Object detection

| Candidate | Why | A | B | C | Effort | Verdict |
|---|---|---|---|---|---|---|
| **Repair the two existing heads for export** | FCOS-lite: no `exp`. GFL-lite: no slices (section 2) | High | Med | Low | S | **Adopt now** |
| **NMS-free one-to-one head** | The direction of the field: YOLO26 ships it and drops the distribution loss to simplify export; the announced YOLO27 keeps a convolutional design for its small models and a query-based one for the large `[web]`, vendor material | High | High | High | M | **Adopt now** (unchanged) |
| **ReLU6 cross-stage-partial backbone and neck** | The Axis recipe; see section 3 | High | Med | High | M | **Adopt now** |
| **Detection loader and training loop** | Still the blocker: the heads cannot be trained with the library | High | High | High | L | **Adopt now** (unchanged) |
| Heatmap head (CenterNet style) | NMS-free by max-pooling; shared with keypoints and counting | High | High | Med | M | Adopt later (unchanged) |
| DEIMv2 small variants (Atto to Nano: 0.5M to 3.6M parameters, 23.8 to 43.0 AP `[web]`) | Best accuracy per parameter found, Apache-2.0, but a transformer decoder | None | Low | High | L | Skip as an implementation; teacher and Tier C upstream |
| RF-DETR, D-FINE, RT-DETR | As above | None | Low | High | L | Skip as implementations (unchanged) |
| PP-PicoDet, NanoDet-Plus | GFL-lite with its Ghost neck and SimOTA already is this design | n/a | n/a | n/a | n/a | No action |

License note: the Ultralytics models are AGPL-3.0 `[web]`. Ideas are free
to use; code and weights are not, unless the project accepts that license.

### 4.4 Instance segmentation

| Candidate | Why | A | B | C | Effort | Verdict |
|---|---|---|---|---|---|---|
| **Keep the prototype design, export it raw** | The library's YOLACT-style head exports with only the FCOS `exp` off the lists (section 2). Mask assembly is a matrix product and a sigmoid, done by the application. YOLO26 still uses prototypes `[web]` | High | High | High | S | **Adopt now** (with the FCOS repair and a reference assembler) |
| One-to-one assignment for instances | Follows the NMS-free detection head | High | High | High | M | With 4.3 |
| SparseInst | NMS-free instance activation maps; needs bipartite matching in training | Med | Med | High | L | Adopt later |
| RTMDet-Ins, CondInst (dynamic kernels per instance) | Weights generated at run time cannot run on these accelerators | None | Low | High | L | Skip |

### 4.5 Keypoints and pose

| Candidate | Why | A | B | C | Effort | Verdict |
|---|---|---|---|---|---|---|
| **Repair `pose_regression_head` for export** | Dynamic reshape operators (section 2) | High | High | Low | S | **Adopt now** |
| **Coordinate-classification head (SimCC, as in RTMPose)** | Each keypoint as two 1-D classifications; the outputs are logits and the argmax runs on the host. No high-resolution heatmap to quantize. Top-down: it needs a person crop from a detector | High | High | High | M | **Adopt later** (unchanged), first among the new heads |
| Keypoint branch on the detector (YOLO-pose style) | One stage, many people, shares the detection work. An int8 YOLOv8n-pose runs on a microcontroller NPU, where its vendor recommends per-tensor quantization `[web]` | High | High | High | M | With 4.3 |
| RTMO | One-stage with coordinate classification, 74.8 AP for its large model `[web]`; heavier head | Low | Med | High | L | Watch |
| Heatmap head | Shared with the CenterNet-style detection head | High | High | Med | M | Adopt later |

### 4.6 Monocular depth

| Candidate | Why | A | B | C | Effort | Verdict |
|---|---|---|---|---|---|---|
| **Distillation recipe with a foundation teacher** | Pseudo-labels on the user's own footage, trained with the existing heads, loader and metrics. Depth Anything V2 Small is Apache-2.0 (24.8M parameters); the Base and Large models are CC-BY-NC `[web]` | High | High | High | S | **Adopt now** (recipe and notebook) |
| **Affine-invariant and gradient-matching losses** | Teacher depth is relative: it is defined up to scale and shift. The library only has a scale-invariant loss | High | High | High | S | **Adopt now** |
| **ZipDepth** (Tosi et al., July 2026) | 6.1M parameters, reparameterizable encoder and decoder, distilled from a foundation model, with a release variant meant for NPU export, already in the Hailo zoo `[web]`. Check the license of code and weights, then either port it or reproduce the recipe on `fastdepth_head` | High | High | High | M | **Adopt later**, after the recipe exists to compare against |
| Self-supervised depth from video (SC-Depth, Lite-Mono) | No labels, but needs camera motion; fixed cameras do not provide it | Low | Low | Med | L | Skip for fixed cameras |
| Metric depth heads (binning, camera-aware) | A fixed camera can calibrate scale once from the scene instead | Low | Low | Med | L | Skip |

### 4.7 Re-identification and embeddings

| Candidate | Why | A | B | C | Effort | Verdict |
|---|---|---|---|---|---|---|
| **Accelerator-clean embedding export** | Average pooling, no in-graph normalization (section 2) | High | High | Low | S | **Adopt now** |
| **Batch-norm neck and a margin classifier for training** | The standard strong baseline: an identity classifier with a margin loss next to the triplet loss the library has; the classifier is dropped at export | High | High | High | S | **Adopt now** |
| OSNet | Fewest parameters, but in a Coral study it had the most operations and the slowest quantized inference among compact backbones (49 ms) `[web]` | Low | Med | Med | M | Skip for Tier A; a ResNet-18 or MobileNetV4 embedding is the better fit |
| LightMBN (multi-branch on OSNet) | Higher accuracy, several pooled branches | Low | Med | High | M | Adopt later, Tier C |

### 4.8 Anomaly detection and out-of-distribution scoring

| Candidate | Why | A | B | C | Effort | Verdict |
|---|---|---|---|---|---|---|
| **Diagonal covariance option** | PaDiM-Lite: the distance becomes element-wise, memory drops by three quarters. On MobileNetV2 features it costs accuracy at image level (0.94 to 0.85 AUROC on MVTec AD) and little at pixel level (0.97 to 0.94) `[web]`. A switch on `fit_class_statistics` | High | High | High | S | **Adopt now** |
| **Per-location statistics** (PaDiM) | For fixed cameras; the first pass already listed it | Med | High | High | S | Adopt now (raised: the embedding output now exists) |
| Memory-bank scoring (PatchCore, with coreset or product quantization) | Strongest accuracy of the feature-based methods (0.95 image AUROC on MVTec AD with MobileNetV2 features `[web]`); nearest-neighbour search on the host | Med | High | High | M | Adopt later |
| EfficientAD | Student, teacher and autoencoder are small plain convolutional networks, about 98.8 image AUROC on MVTec AD `[web]`. Three networks in one export, the difference computed by the application. No edge or int8 results found | Med | High | High | M | **Adopt later** (raised from "watch") |
| Reconstruction with large vision transformers (Dinomaly and similar) | Too heavy for A and B | None | None | Med | L | Skip |

### 4.9 Counting

| Candidate | Why | A | B | C | Effort | Verdict |
|---|---|---|---|---|---|---|
| **Density-map head** | The depth heads with a non-negative output already are one; the count is the sum, taken by the application. Needs a point-annotation loader and a density loss | High | High | High | S to M | Adopt later |
| Point-based counting (P2PNet and successors; RepSFNet reports 34% lower latency than P2PNet `[web]`) | Localizes each head; bipartite matching in training | Med | Med | High | L | Watch |
| Counting by detection | For sparse scenes the detector is enough | High | High | High | none | Use 4.3 |

### 4.10 Video and temporal cues

| Candidate | Why | A | B | C | Effort | Verdict |
|---|---|---|---|---|---|---|
| **Frame stacking or a difference channel at the input** | The only temporal option whose graph stays plain | High | High | High | M | **Adopt now** (unchanged) |
| Smoothing of outputs over time in the application | No network change; removes flicker in segmentation and classification | High | High | High | S | Adopt now (documentation) |
| MoViNet streaming models | Causal 3-D convolutions with state carried between frames; still the reference for streaming video `[web]`. 3-D convolutions are not on the accelerator lists | None | Med | High | L | Adopt later, Tiers B and C |
| Temporal shift module | Needs cached features as model inputs and outputs | Low | Med | High | L | Watch (unchanged) |

### 4.11 Several tasks in one model

`multi_task` exists since 2026-11-05. What is missing:

| Candidate | Why | Effort | Verdict |
|---|---|---|---|
| **Detection head types** in `multi_task` | Detection plus segmentation on one backbone is the common camera combination; A-YOLOM does it with convolution-only segmentation heads `[web]` | M | Adopt now, with 4.3 |
| **Loss balancing by learned uncertainty** | One trainable scalar per task; removes hand-tuned weights | S | Adopt now |
| **A loader that yields one target per head** | Today the caller assembles it | M | Adopt now |
| Gradient-surgery methods (PCGrad, GradNorm) | Custom training step; small gains on two or three tasks | M | Skip |

---

## 5. Teachers and automatic labelling

These do not run on the device. They turn unlabelled footage into training
data for the small models.

| Model | Produces | License as found `[web]` | Note |
|---|---|---|---|
| SAM 3 (November 2025) | Masks for every instance of a concept named by a short phrase or an example | Custom SAM License; a vendor summary says it allows commercial use with conditions; gated download | Read the license itself before use |
| Depth Anything V2 Small | Relative depth | Apache-2.0 | Base and Large are CC-BY-NC |
| DEIMv2, RF-DETR, D-FINE, RT-DETR | Boxes | Apache-2.0 | Closed set; fine-tune on a few labelled frames first |
| DINOv3 | Dense features for distillation targets or similarity | Custom DINOv3 License; gated download | Not an open-source license |
| YOLOE-26, YOLO-World | Open-vocabulary boxes and masks | AGPL-3.0 for the Ultralytics models | Labels produced by a model are a separate question from its code; get advice |

---

## 6. Recommended sequence

This refines steps 4 to 8 of the first pass. Steps 1 to 3 of that list
(profiles, benchmark harness, quantization accuracy) keep their priority.

| Step | Content | Effort |
|---|---|---|
| 1 | The seven export repairs of section 2, with an export test per head family | M in total |
| 2 | DDRNet and PIDNet repairs: free input sizes, reference batch-norm layout and constants, weight port script | M |
| 3 | Depth recipe: affine-invariant loss, teacher pseudo-label notebook | S |
| 4 | Plain pyramid semantic head; DeepLab without dilation; DDRNet boundary auxiliary head | S each |
| 5 | Out-of-distribution: diagonal covariance, per-location statistics | S |
| 6 | Re-identification: batch-norm neck, margin classifier | S |
| 7 | Detection end to end: loader, ReLU6 cross-stage-partial backbone, one-to-one head; then detection heads in `multi_task` and the keypoint branch | L |
| 8 | GCNet, gated on a post-training quantization measurement; ResNet-18/34 and STDC alongside | M |
| 9 | Coordinate-classification keypoint head; density head; EfficientAD; ZipDepth | M each |

---

## 7. Sources

Axis (read 2026-11-10):

- [Axis model zoo](https://github.com/AxisCommunications/axis-model-zoo)
- [ARTPEC-8 optimization tips](https://developer.axis.com/computer-vision/on-device/system-on-chip/artpec-8/optimization-tips/), [ARTPEC-9 optimization tips](https://developer.axis.com/computer-vision/on-device/system-on-chip/artpec-9/optimization-tips/). Both unchanged since 2026-11-05: per-tensor for ARTPEC-8; ReLU6, channels in multiples of 16 and no explicit padding for ARTPEC-9
- [Axis DLPU overview](https://developer.axis.com/computer-vision/on-device/axis-dlpu/)

Vendor model zoos:

- [Hailo model zoo, semantic segmentation on Hailo-8](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_semantic_segmentation.rst), [public models index](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/PUBLIC_MODELS.rst)
- [STMicroelectronics YOLOv8n-pose model card](https://huggingface.co/STMicroelectronics/yolov8n_pose)

Papers and project pages:

- Segmentation: [GCNet, CVPR 2025](https://arxiv.org/html/2503.03325v1), [RDRNet](https://arxiv.org/abs/2406.12496v1), [PIDNet reference code](https://github.com/XuJiacong/PIDNet)
- Detection: [DEIMv2](https://arxiv.org/html/2509.20787v3), [YOLO26 paper](https://arxiv.org/abs/2509.25164v2), [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/), [Ultralytics YOLO27 page](https://docs.ultralytics.com/models/yolo27), [YOLO26 semantic segmentation task](https://docs.ultralytics.com/tasks/semantic)
- Depth: [ZipDepth](https://arxiv.org/abs/2607.08771), [ZipDepth code](https://github.com/fabiotosi92/ZipDepth), [Depth Anything V2 Small model card (third-party mirror citing the official license table)](https://huggingface.co/apple/coreml-depth-anything-v2-small/blob/main/README.md?code=true)
- Keypoints: [RTMO](https://arxiv.org/pdf/2312.07526)
- Anomaly detection: [PatchCore-Lite and PaDiM-Lite](https://arxiv.org/html/2603.20288v1), [EfficientAD](https://arxiv.org/pdf/2303.14535)
- Re-identification: [Coral-based person re-identification study](https://arxiv.org/pdf/2209.11024), [LightMBN](https://arxiv.org/abs/2101.10774v1)
- Counting: [RepSFNet](https://arxiv.org/html/2601.20369)
- Video: [MoViNets](https://arxiv.org/pdf/2103.11511)
- Multi-task: [A-YOLOM](https://arxiv.org/html/2310.01641v4)
- Backbones and teachers: [LSNet](https://arxiv.org/abs/2503.23135), [Gemma 3n developer guide (MobileNetV5)](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/), [DINOv3 model card](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/blob/main/README.md), [SAM 3 paper](https://arxiv.org/pdf/2511.16719), [SAM 3 license summary (vendor)](https://roboflow.com/model-licenses/segment-anything-3)

Measurements in this repository: the export sweep of section 2 was run on
2026-11-10 with the project environment (TensorFlow 2.21). Its method is
the pipeline described in
[docs/source/deployment.rst](docs/source/deployment.rst).

Limits of this pass: the searches were single queries per topic, several
results came from secondary pages, and no paper was re-read in full. Treat
every `[web]` figure as a lead to verify.
