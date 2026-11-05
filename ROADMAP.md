# Roadmap

Open work in priority order, as of 2026-11-05. The reasoning is in
[ASSESSMENT.md](ASSESSMENT.md) (state of the code) and
[EDGE_AI_CANDIDATES.md](EDGE_AI_CANDIDATES.md) (what to add, by target
hardware). Completed work is recorded in [CHANGELOG.md](CHANGELOG.md);
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

Still open:

| Item | Effort |
|---|---|
| Verify the Axis operator lists in `deploy_utils.py` on real ARTPEC-8 and ARTPEC-9 devices and correct them | S |
| Cover the remaining published per-operator constraints in the lint (pooling windows, kernel sizes, padding amounts) | S |
| Export detectors without in-graph decoding and NMS, with a reference decoder | S |
| Extend `tests/test_tflite_export.py` to every decoder, head and backbone family | M |
| Intermittent failure of the PP-LiteSeg + fused MobileOne-S0 export test for `artpec9`: the LiteRT CPU delegate (XNNPACK) could not prepare the file in 1 of 7 runs on 2026-11-05. The test uses unseeded random weights and calibration data; find the degenerate case and seed the test | S |
| On-device benchmark harness and a published latency table | M |
| Decide the future of QAT wrapping: it no longer contributes activation ranges for accelerator exports. Options: keep it as a training regularizer, make it optional per builder, or drop it in favour of post-training quantization (see section 7) | M |

## 2. Make quantized models accurate

Done on 2026-11-05: cross-layer equalization (`equalize_cross_layer`, used by
`Quantizer` for per-tensor profiles), an error measurement
(`quantization_error`), gate-free options for LR-ASPP and PP-LiteSeg, and the
root cause of the all-zero QAT output (an untrained output quantizer rounding
tiny values; not a defect). Measurements on pretrained backbones are in
`docs/source/deployment.rst`.

Still open, in the order the measurements suggest:

| Item | Effort |
|---|---|
| Validate on task accuracy: train one segmentation and one classification model, export per channel and per tensor, compare with the float model (the SQNR figures are only a proxy) | M |
| Fix post-training quantization of fused MobileOne and RepViT (noise as large as signal at the deepest tap): percentile or MSE activation calibration, bias correction, QARepVGG-style blocks, or QAT of the fused backbone | L |
| Equalization through ReLU6 (per-channel clipping), or a ReLU variant of MobileNetV2 / EfficientNet-Lite with fine-tuning, so those backbones work per tensor | M |
| Per-layer quantization error report (wrap `tf.lite.experimental.QuantizationDebugger`) | S |
| Edge-clean builder switches: ReLU6 or ReLU instead of GELU, hard-swish and swish; no squeeze-excite; channel rounding; SAME padding | M |
| A gate-free decoder designed for accelerators (feature pyramid with convolutions, 2x resizes and additions only); gate-free variants of PIDNet and PP-MobileSeg | S to M |

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

## 3. Ship pretrained weights

| Item | Effort |
|---|---|
| License review of the converted weights (MobileOne first) | S |
| Host the converted files; `weights='imagenet'` download with checksums; `WEIGHTS.md` | M |

## 4. Training infrastructure

| Item | Effort |
|---|---|
| Learning-rate schedules and a differential backbone/head rate in `TrainingDriver` | S |
| EMA and SWA weight averaging used for validation, checkpoints and export | M |
| Decide and fix batch-norm handling in the unfrozen phase (backbone batch-norm layers become trainable today) | S |
| `TrainingDriver` output directory, optional export, resume | S |
| Loss-weight ramping callback; rare-class and domain-balanced sampling | M |
| Data loaders: non-square inputs, persisted split manifest, `close()`, unit tests; benchmark the worker pool against plain `tf.data` | M |
| Knowledge-distillation wrapper | M |

## 5. New components for Axis-class accelerators

| Item | Effort |
|---|---|
| ResNet-18/34 and STDC1/2 backbones | S to M |
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
