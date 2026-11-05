# Roadmap

Open work in priority order, as of 2026-11-05. The reasoning is in
[ASSESSMENT.md](ASSESSMENT.md) (state of the code) and
[EDGE_AI_CANDIDATES.md](EDGE_AI_CANDIDATES.md) (what to add, by target
hardware). Completed work is recorded in [CHANGELOG.md](CHANGELOG.md);
[plan.md](plan.md) is the engineering log of the 0.11 cycle and is no longer
updated.

Effort: S is up to 2 days, M is up to 2 weeks, L is more.

## 1. Make exported models right for the target

| Item | Effort |
|---|---|
| Fuse reparameterizable backbones nested in a full model before export (today MobileOne exports with 416 operators instead of 50) | M |
| `Quantizer` deployment profiles: per-tensor weights, int8 or uint8 I/O, target presets (ARTPEC-8, ARTPEC-9, LiteRT CPU) | M |
| Operator linter: fail export or test when a graph uses operators the target does not accelerate | S |
| Track down the float islands (DEQUANTIZE / QUANTIZE) in LR-ASPP and DeepLabV3+ exports | S |
| Export detectors without in-graph decoding and NMS, with a reference decoder | S |
| Extend `tests/test_tflite_export.py` to every decoder, head and backbone family, per profile | M |
| On-device benchmark harness and a published latency table | M |

## 2. Make per-tensor quantization accurate

| Item | Effort |
|---|---|
| Cross-layer equalization and bias correction | M |
| Quantization debugger report (per-layer error) | S |
| Edge-clean builder switches: ReLU6 instead of GELU, hard-swish and swish; no squeeze-excite; channel rounding; SAME padding | M |
| Gate-free decoder, and gate-free options for LR-ASPP and PP-LiteSeg | S |
| Root-cause the all-zero output of QAT-wrapped linear output layers on randomly initialized MobileNetV2 | S |

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
| Multi-task model builder; temporal (multi-frame) input | M |

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
