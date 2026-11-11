# Roadmap

Open work in priority order, as of 2026-11-10. The reasoning is in
[ASSESSMENT.md](ASSESSMENT.md) (state of the code; sections 13 and 14 are the latest), [EDGE_AI_CANDIDATES.md](EDGE_AI_CANDIDATES.md) (what to add, by
target hardware) and [CANDIDATES_BY_TASK.md](CANDIDATES_BY_TASK.md)
(backbones and heads by task, with the export sweep of 2026-11-10). Completed work is recorded in [CHANGELOG.md](CHANGELOG.md);
[plan.md](plan.md) is the engineering log of the 0.11 cycle and is no longer
updated.

Effort: S is up to 2 days, M is up to 2 weeks, L is more.

## 1. Make exported models right for the target

Done on 2026-11-05 (`swiss_army_keras/deploy_utils.py`, `Quantizer`,
`TrainingDriver(export_profile=...)`): fusion of reparameterizable backbones
nested in a full model, deployment profiles with per-tensor weights, the
export check against the target's operator list, and stripping of QAT
wrappers for integer-only targets, which also removed the float operators
found in LR-ASPP and DeepLabV3+ exports.

Done on 2026-11-10: the export repairs found by the sweep (FCOS without `EXP`, GFL without slices, windowed
pooling, pose and embedding heads, fixed batch conversion, which also removes the DeepLab `SPACE_TO_BATCH_ND`
operators), `strip_qat` on graphs Keras cannot clone, the zero-scale and near-zero-scale lint checks, and
`select_outputs`.

Still open:

| Item | Effort |
|---|---|
| Verify the Axis operator lists in `deploy_utils.py` on real ARTPEC-8 and ARTPEC-9 devices and correct them | S |
| Cover the remaining published per-operator constraints in the lint (pooling windows, kernel sizes, padding amounts) | S |
| Export detectors without in-graph decoding and NMS, with a reference decoder | S |
| Extend `tests/test_tflite_export.py` to every decoder, head and backbone family | M |
| The image-quality head still pools with GeM by default, which stays in floating point; the full detectors still decode and run NMS in the graph, so only their `*_base` models suit an accelerator | S |
| On-device benchmark harness and a published latency table | M |
| Decide the future of QAT wrapping: it no longer contributes activation ranges for accelerator exports. Options: keep it as a training regularizer, make it optional per builder, or drop it in favour of post-training quantization (see section 7) | M |

## 2. Make quantized models accurate

Done on 2026-11-05: cross-layer equalization, an error measurement, gate-free options for LR-ASPP and PP-LiteSeg.
Done on 2026-11-10: batch-norm folding and bias correction (default for per-tensor profiles), a per-layer error
report (`quantization_debug_report`), QAT made optional (`set_qat`), and the **first task-accuracy measurement**:
two models trained in September, re-exported in every variant (table in `docs/source/deployment.rst`).
Per-tensor PP-LiteSeg on MobileNetV4-Conv-S went from 0.741 to 0.808 mIoU, against 0.832 in float.
Done on 2026-11-11: whole-model quantization-aware training (`prepare_qat`), per channel and per tensor, with a
measured gain per tensor on fused MobileOne-S1 (ASSESSMENT.md section 15).

Still open, in the order the measurements suggest:

| Item | Effort |
|---|---|
| Repeat the task-accuracy measurement on a model trained with a recorded split, and on a classifier; the September runs did not record their split | M |
| Fused MobileOne and RepViT: whole-model QAT (`prepare_qat`, 2026-11-11) brought LR-ASPP on MobileOne-S1 to 0.713 mIoU per tensor after a five-minute fine-tune (float 0.731 after the same fine-tune). Open: confirm on a second model and with a recorded split, measure RepViT, train QARepVGG and compare | M |
| Equalization and bias correction interact badly on the fused MobileOne model (bias correction alone was better); understand why before trusting the default on reparameterized backbones | M |
| Bias correction with the activations of the partly quantized model, as in the original method | M |
| Equalization through ReLU6 (per-channel clipping), or a ReLU variant of MobileNetV2 / EfficientNet-Lite with fine-tuning | M |
| Edge-clean builder switches for the phone-oriented backbones: ReLU6 or ReLU instead of GELU, hard-swish and swish; no squeeze-excite; `'same'` padding | M |
| Decide whether the head QAT wrappers stay the default: they added 0.001 mIoU over post-training quantization; whole-model QAT (`prepare_qat`) is what helps | S |
| `TrainingDriver` option for a whole-model QAT phase after training (today: `prepare_qat`, then `compile` / `fit` / `Quantizer` by hand) | S |
| Whole-model QAT for PIDNet: give its custom gate layers a quantization config that matches the fixed sigmoid output scale, removing the two float requantization steps | S |

## 2b. Depth and out-of-distribution analysis

Done on 2026-11-05: depth heads on every backbone family, a depth data
loader and depth metrics, export of depth models for ARTPEC-8 without the
final sigmoid, an embedding output for Mahalanobis statistics, a NumPy-only
distance for devices, and out-of-distribution evaluation metrics. None of
it has been trained or evaluated on a real dataset.

Still open:

| Item | Effort |
|---|---|
| Train one depth model (teacher pseudo-labels or NYU) and one model with out-of-distribution scoring; report the depth metrics and AUROC for float and exported files | M |
| Example notebooks for depth and for out-of-distribution scoring | S |
| Depth: scale-and-shift invariant loss and gradient-matching loss for teacher pseudo-labels; affine alignment in the metrics | S |
| Out-of-distribution: per-location statistics (PaDiM style) for fixed cameras; embedding choice helper for decoders with several branches (DDRNet, PIDNet) | S to M |
| An image-quality head that exports to integer (GeM pooling stays in floating point) | S |

## 2c. PIDNet and DDRNet

Done on 2026-11-10: free input sizes (any multiple of 8), the reference layer layout and batch-norm constants,
loss weights and output selection in `TrainingDriver`, export tests. PIDNet-S, M and L match the parameter count
of the official architecture exactly.

Still open:

| Item | Effort |
|---|---|
| Port script and numeric check against the official PyTorch models (running the official code was not possible in the session that did the repairs); then ImageNet and Cityscapes weights, after a license review | M |
| The boundary-attentive term of the PIDNet loss takes two outputs and still needs a custom training step | S |
| Make DDRNet the documented segmentation default for Axis accelerators once it has weights; boundary auxiliary head for DDRNet | S |

## 3. Ship pretrained weights

| Item | Effort |
|---|---|
| License review of the converted weights (MobileOne first) | S |
| Host the converted files; `weights='imagenet'` download with checksums; `WEIGHTS.md` | M |

## 4. Training infrastructure

| Item | Effort |
|---|---|
| Done on 2026-11-10 as separate tools: schedules (`WarmupCosine`, `OneCycle`), per-group learning rates, decay presets, weight averaging (`WeightAveraging`), SAM inside `fit`, distillation (`Distiller`), learned loss balancing. They work with `model.fit`; wiring them into `TrainingDriver` (its second phase rebuilds the optimizer) is open | S |
| Decide and fix batch-norm handling in the unfrozen phase (backbone batch-norm layers become trainable today) | S |
| `TrainingDriver` output directory, optional export, resume | S |
| Loss-weight ramping callback; rare-class and domain-balanced sampling | M |
| Data loaders: non-square inputs, persisted split manifest, `close()`, unit tests; benchmark the worker pool against plain `tf.data` | M |
| Distillation of intermediate features (the wrapper distills one output) | S |

## 5. New components for Axis-class accelerators

| Item | Effort |
|---|---|
| Done on 2026-11-10: ResNet-18/34, STDC1/2, QARepVGG, CSP with ReLU6, MobileNetV3Small; the plain pyramid head; the coordinate-classification keypoint head; the density head; the re-identification training wrapper. All without pretrained weights and without a numeric comparison with their references | |
| Weights for the plain backbones: port torchvision ResNet-18/34 and the STDC checkpoints (`torch_padding=True` reproduces their alignment) | S to M |
| GCNet (CVPR 2025): DDRNet-style segmentation network fused to plain 3x3 convolutions; gate on a post-training quantization measurement | M |
| Depth: teacher pseudo-label recipe and notebook (the affine-invariant losses exist; Depth Anything V2 Small is Apache-2.0); evaluate ZipDepth | S, then M |
| Out-of-distribution: diagonal covariance and per-location statistics | S each |
| Later: EfficientAD; detection heads as `multi_task` head types | M each |
| Vendored EfficientNet-Lite; MobileNetV3Small | S |
| MobileDet-style and QARepVGG-style backbones | M |
| Detection end to end: data loader, training loop, NMS-free head | L |
| Multi-task: the builder exists (`multi_task`, 2026-11-05). Open: more head types (PP-LiteSeg, detection), a loss-balancing helper (uncertainty weighting), a data loader that yields one target per head, `TrainingDriver` support for several outputs | M |
| Temporal (multi-frame) input | M |

## 6. Project hygiene

| Item | Effort |
|---|---|
| Continuous integration: fast tests on pull requests, export tests nightly, docs build | S |
| `pyproject.toml`; linting | S |
| Validate the recipes of `use_case_recipes.md` on real datasets | L |
| Remove debugging `print()` calls and `eval()`-based activation lookup from the inherited model files | S |
| Split `_model_detection.py`, `losses.py` and `dataset_utils.py` | M |
| Trim development history out of docstrings (see CONTRIBUTING.md) | M |

## 7. Strategic

| Item | Effort |
|---|---|
| Measure plain post-training quantization (with equalization) against today's QAT on two real tasks; if the gap is small, make QAT optional and write the Keras 3 migration decision | M |
| Migrate the interpreter to LiteRT (`ai_edge_litert`) | S |
| ONNX and TensorRT export path for Jetson-class devices | M |
