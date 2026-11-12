# Roadmap

Open work in priority order, as of 2026-11-12. The reasoning is in
[ASSESSMENT.md](ASSESSMENT.md) (state of the code; sections 15 and 16 are the latest), [EDGE_AI_CANDIDATES.md](EDGE_AI_CANDIDATES.md) (what to add, by
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

Done on 2026-11-12: detectors export without decoding and NMS (`.base_model`), with a description of the file's
outputs and a NumPy reference decoder for the host (`detection_utils`).

Still open:

| Item | Effort |
|---|---|
| Verify the Axis operator lists in `deploy_utils.py` on real ARTPEC-8 and ARTPEC-9 devices and correct them | S |
| Cover the remaining published per-operator constraints in the lint (pooling windows, kernel sizes, padding amounts) | S |
| Reference decoder: cover instance segmentation (prototype masks); time it on a device | S |
| Extend `tests/test_tflite_export.py` to every decoder, head and backbone family | M |
| On-device benchmark harness and a published latency table | M |

## 2. Make quantized models accurate

Done on 2026-11-05: cross-layer equalization, an error measurement, gate-free options for LR-ASPP and PP-LiteSeg.
Done on 2026-11-10: batch-norm folding and bias correction (default for per-tensor profiles), a per-layer error
report (`quantization_debug_report`), QAT made optional (`set_qat`), and the **first task-accuracy measurement**:
two models trained in September, re-exported in every variant (table in `docs/source/deployment.rst`).
Per-tensor PP-LiteSeg on MobileNetV4-Conv-S went from 0.741 to 0.808 mIoU, against 0.832 in float.
Done on 2026-11-11: whole-model quantization-aware training (`prepare_qat`), per channel and per tensor, with a
measured gain per tensor on fused MobileOne-S1 (ASSESSMENT.md section 15).
Done on 2026-11-12: a whole-model QAT phase in `TrainingDriver` (`qat_epochs`), and a fixed output range for
PIDNet's gate layers, which removed the last floating-point steps of its QAT export.

Still open, in the order the measurements suggest:

| Item | Effort |
|---|---|
| Repeat the task-accuracy measurement on a model trained with a recorded split (the loaders record it since 2026-11-12), and on a classifier; the September runs did not record their split. Needs a training run | M |
| Fused MobileOne and RepViT: whole-model QAT (`prepare_qat`, 2026-11-11) brought LR-ASPP on MobileOne-S1 to 0.713 mIoU per tensor after a five-minute fine-tune (float 0.731 after the same fine-tune). Open: confirm on a second model and with a recorded split, measure RepViT, train QARepVGG and compare | M |
| Equalization and bias correction interact badly on the fused MobileOne model (bias correction alone was better); understand why before trusting the default on reparameterized backbones | M |
| Bias correction with the activations of the partly quantized model, as in the original method | M |
| Equalization through ReLU6 (per-channel clipping), or a ReLU variant of MobileNetV2 / EfficientNet-Lite with fine-tuning | M |
| Edge-clean builder switches for the phone-oriented backbones: ReLU6 or ReLU instead of GELU, hard-swish and swish; no squeeze-excite; `'same'` padding | M |
| Decide whether the head QAT wrappers stay the default: they added 0.001 mIoU over post-training quantization and are stripped for accelerators; whole-model QAT (`prepare_qat`) is what helps. Options: keep, make them opt-in, or remove | S |
| Whole-model QAT simulates custom layers (PIDNet's gate and blend) as one step; simulated and exported outputs differ by 7% to 12% there, against 0.1% to 5% elsewhere. Split those layers into standard ones, or measure that the gap does not matter after fine-tuning | M |

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
| Depth: scale-and-shift alignment in the metrics (the losses exist; the metrics align by the median only) | S |
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
| Data loaders: non-square inputs, `close()`, more unit tests; benchmark the worker pool against plain `tf.data`. The split is recorded since 2026-11-12 (`split_manifest`) | M |
| Distillation of intermediate features (the wrapper distills one output) | S |

## 5. New components for Axis-class accelerators

| Item | Effort |
|---|---|
| Done on 2026-11-10: ResNet-18/34, STDC1/2, QARepVGG, CSP with ReLU6, MobileNetV3Small; the plain pyramid head; the coordinate-classification keypoint head; the density head; the re-identification training wrapper. All without pretrained weights and without a numeric comparison with their references | |
| Weights for the plain backbones: port torchvision ResNet-18/34 and the STDC checkpoints (`torch_padding=True` reproduces their alignment) | S to M |
| GCNet (CVPR 2025): DDRNet-style segmentation network fused to plain 3x3 convolutions; gate on a post-training quantization measurement | M |
| Depth: teacher pseudo-label recipe and notebook (the affine-invariant losses exist; Depth Anything V2 Small is Apache-2.0); evaluate ZipDepth | S, then M |
| Out-of-distribution: diagonal covariance (per-location statistics are listed in section 2b) | S |
| Later: EfficientAD | M |
| Vendored EfficientNet-Lite (today an optional external package) | S |
| MobileDet-style backbone | M |
| Detection end to end: data loader, training loop, NMS-free head | L |
| Multi-task: the builder, learned loss balancing (`uncertainty_weighted`) and `TrainingDriver` with losses by output name exist. Open: more head types (PP-LiteSeg, detection), a data loader that yields one target per head | M |
| Temporal (multi-frame) input | M |

## 6. Project hygiene

| Item | Effort |
|---|---|
| Continuous integration: the workflow exists since 2026-11-12 (`.github/workflows/tests.yml`: fast tests and docs on push and pull request, slow tests nightly). It has not run yet: check the first run on GitHub and fix what the runner lacks | S |
| `pyproject.toml`; linting | S |
| Validate the recipes of `use_case_recipes.md` on real datasets | L |
| Remove debugging `print()` calls and `eval()`-based activation lookup from the inherited model files | S |
| Split `_model_detection.py`, `losses.py` and `dataset_utils.py` | M |
| Trim development history out of docstrings (see CONTRIBUTING.md) | M |

## 7. Strategic

| Item | Effort |
|---|---|
| Write the Keras 3 migration decision. Measured on two models, the head QAT wrappers add 0.001 mIoU over post-training quantization and are optional (`set_qat`), so they no longer tie the library to Keras 2. Whole-model QAT (`prepare_qat`) does: it needs `tensorflow_model_optimization`, which only supports Keras 2 | M |
| Migrate the interpreter to LiteRT (`ai_edge_litert`) | S |
| ONNX and TensorRT export path for Jetson-class devices | M |
