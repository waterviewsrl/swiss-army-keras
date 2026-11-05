# swiss-army-keras — project reassessment (2026-11-04)

Scope: a fresh read of the `v2` branch as it stands today (committed +
uncommitted working tree), `plan.md`, `use_case_recipes.md`, the docs tree,
packaging metadata, and a full run of the test suite in the project venv
(Python 3.12.3, TensorFlow 2.21.0, tf-keras 2.21.0). It deliberately does not
re-review what `plan.md` already tracks as done; it asks what is still wrong,
what is missing, and what to do next.

Section 7 (evidence) carries the `file:line` pointers for every claim above it.

---

## 1. Where the project is

| Item | State |
|---|---|
| Package version | 0.11.0 (`setup.py`); no `__version__` attribute in the package |
| Branch | `v2`, one commit ahead of `main` ("V2 progress") at the time of the read, plus a **large uncommitted diff** (35 modified files, +8.4k lines) **and six source modules that had never been added to git at all** (`_model_pidnet.py`, `_model_pp_mobileseg.py`, `_backbone_strideformer.py`, `_model_depth.py`, `ood_utils.py`, `iqa_utils.py`, with their four test files). Everything was committed as `d7d6d21` on 2026-11-04 (see §8). |
| Source | 18.4k lines in `swiss_army_keras/`; 3 files exceed 950 lines (`_model_detection.py` 2259, `losses.py` 1592, `layer_utils.py` 955) |
| Tests | 21 files, 292 test functions, 5.5k lines; pytest + `conftest.py` forcing `TF_USE_LEGACY_KERAS=1` |
| Test run | see §1.1 |
| CI / lint / typing | none (no `.github/`, no ruff/flake8/mypy config, no pre-commit) |
| Plan | `plan.md` (295 KB) Phases 0–11 all marked DONE; only the "opportunistic" items of §8 and the real-dataset validation of the recipes remain open |
| Backbones | 13 `tf.keras.applications` entries + EfficientNet-Lite (import currently **fails** on Python 3.12) + 24 custom variants across 5 families (MobileOne, MobileNetV4, FastViT, RepViT, StrideFormer) |
| Pretrained weights | 22 ported `.weights.h5` files (1.2 GB) exist **only on this machine** (`pretrained_weights/`, gitignored); `weights='imagenet'` raises `NotImplementedError` for every custom family |
| Heads | 7 segmentation decoders (DeepLabV3+ ×2, LR-ASPP, PP-LiteSeg, DDRNet, PIDNet, PP-MobileSeg) + 8 legacy U-Net family models; FCOS-lite / GFL-lite detection; YOLACT-style instance seg; 8 classifier/embedding heads; 2 depth heads; OOD, IQA, TTA, uint8-input utilities |

### 1.1 Test suite result

**All green: 411 passed, 0 failed** (non-slow tests, about 18 minutes of CPU
time), plus the 2 new `slow` TFLite export tests. Details and two caveats
(one flaky test fixed, one tooling trap) in §8.2.

---

## 2. Critical points (ranked)

These are the things that either block the library from doing what its README
promises, or that will silently bite users.

### C1. The library cannot be installed from PyPI as declared

- `install_requires` contains a direct VCS reference
  (`efficientnet-lite-keras @ git+https://...`). PyPI rejects uploads whose
  metadata contains direct URL dependencies, so the README's
  `pip3 install swiss-army-keras` cannot deliver 0.11.0.
- That fork also caps Python at ≤3.11; under this venv (3.12) its import fails
  and the zoo degrades with a warning, so **EfficientNet-Lite backbones are
  dead on current Python** even though the README still lists them and the
  original example notebooks use them.
- `tensorflow_hub` is imported unconditionally by `_backbone_zoo.py` but is
  **not declared** in `install_requires`. A clean install of the package fails
  on `import swiss_army_keras.models`.
- `typeguard` is declared but never imported. `requirements.txt` depends on
  the package itself via git and mixes Sphinx build deps with runtime deps.
- Classifiers say Python 3.6; the code needs ≥3.9 (TF ≥2.16, dataclasses,
  f-strings). There is no `python_requires`. `setup.py`'s URL points at
  `desmoteo/...` while the remote is `waterviewsrl/...`.
- `.gitignore` ignores `setup.py`, `pyproject.toml`, `setup.cfg` (inherited
  from upstream). `setup.py` is tracked only because it was force-added; a
  future `pyproject.toml` will be silently dropped from commits.

### C2. The headline feature (pretrained edge backbones) is not shippable

All five custom families refuse `weights='imagenet'`. The 1.2 GB of
numerically-verified ported weights live only in a gitignored local folder;
there is no hosting, no download helper, no checksum, and the licensing check
(Apple Sample Code License for MobileOne, timm terms for the rest) flagged in
`plan.md §12` is still open. StrideFormer has no weights at all (Paddle only).
Until this is solved the README claim "pretrained backbones for Edge AI" is
only true for the legacy `tf.keras.applications` set.

### C3. The whole stack is pinned to legacy Keras 2 through an env-var side effect

`swiss_army_keras/__init__.py` sets `TF_USE_LEGACY_KERAS=1` on import. That
only works if the package is imported **before** TensorFlow; otherwise Keras 3
is active and every QAT-wrapped decoder breaks with an opaque tfmot error.
There is no runtime check (`tf.keras.__version__` / `isinstance` probe) that
fails fast with a clear message. Strategically: `tf-keras` is a compatibility
package and `tensorflow_model_optimization` has not moved to Keras 3. The
project has no written decision on what happens when either stops tracking TF
releases. This is the largest long-term risk in the codebase and it is not
mentioned anywhere outside a comment.

### C4. "Edge deployable" is asserted, not tested

No test converts any model to TFLite. The only `TFLiteConverter` call in the
repo is inside `Quantizer.quantize()`, which runs at the end of
`TrainingDriver.run()` and is never exercised by the suite. So the property
the library exists for (every backbone × decoder survives QAT → int8 TFLite
with `TFLITE_BUILTINS_INT8` only) is unverified for 7 decoders × 37 backbones.
Related gaps:

- `Quantizer` hardcodes uint8 in/out and per-channel weights (TFLite default).
  The ARTPEC-8 target that motivated `plan.md §13` needs **per-tensor**
  quantization (`converter._experimental_disable_per_channel`), which is not
  exposed.
- The structural-reparameterization functions (`reparameterize_mobileone`,
  `_fastvit`, `_repvit`) are never called by `Quantizer` or `TrainingDriver`.
  The correct flow (train multi-branch → fuse → quantize) exists only as
  separate pieces the user must know to assemble, and there is no test that
  a fused model quantizes to the same outputs as the unfused one.
- No benchmark harness (params / FLOPs / TFLite CPU latency). The selling
  point of MobileOne/FastViT/RepViT/MobileNetV4 over ResNet/EfficientNet is
  therefore unmeasured.

### C5. Known, unresolved correctness bug in QAT output layers

`plan.md §10` records a reproducible case where a QAT-annotated final
`Dense`/`Conv2D` **without activation** produces exactly-zero output with
`backbone='MobileNetV2', weights=None` (confirmed on `lr_aspp` even after 5
epochs of training). Root cause is not isolated. Every existing
`weights=None` test only asserts `np.isfinite(out)`, which an all-zero tensor
passes, so the suite can't tell whether other heads are silently degenerate.
This should be tracked as an issue, not a paragraph in a plan file.

### C6. The data loaders are the most fragile part of the user-facing API

`dataset_utils.py` (1083 lines, two near-duplicate loader classes) spawns a
ZeroMQ broker plus `cpu_count()/2` daemon worker processes **in `__init__`**,
talks to them through one locked socket from inside `tf.numpy_function` with
`num_parallel_calls=1`, and has no `close()`/context-manager. Concrete defects:

- **Non-reproducible split.** `build_datasets()` shuffles the file list with
  no seed before `take/skip`, so train/val/test membership changes every
  process. Nothing records which files went where.
- `__getstate__` does `del self_dict['pool']` but no `pool` attribute exists
  → pickling raises `KeyError`. Dead or broken code either way.
- Images are decoded with `decode_jpeg` only; PNG inputs fail. Mask paths are
  derived by a string-split on the extension and must be `.png`.
- IPC sockets are created under `/tmp` by name, CURVE keys are hardcoded
  constants (security theatre on a local socket), and augmentations are
  shipped to workers with `dill`.
- The architecture exists to dodge the GIL for albumentations, but cv2/numpy
  release the GIL for most work. Nobody has benchmarked it against the plain
  `tf.py_function` + `num_parallel_calls=AUTOTUNE` baseline.
- Three real data-loader bugs were already found and fixed in Phase 8 running
  notebooks for the first time; the class has zero unit tests.

### C7. Repository hygiene is blocking collaboration

- Working directory is 3.7 GB. Root contains ~600 MB of training artifacts
  (`*.h5`, `*.tflite`, `*.weights.*`, `checkpoint`); `docs/source/notebooks/`
  holds 1.8 GB (datasets, more checkpoints, `tblogs/`). None of these
  patterns are in `.gitignore`.
- Executed notebooks with outputs are **tracked**: one is 12 MB, and the same
  notebooks are duplicated between `examples/` and `docs/source/notebooks/`.
- An unrelated PDF ("Detecting Out-of-Distribution Images.pdf") sits at root.
- `plan.md` is a 295 KB running log. It's an excellent record but no longer a
  plan; new contributors cannot find the current state in it.
- Several model files still contain debugging `print()` calls
  (`_model_deeplab_v3_plus.py`, `_model_u2net_2d.py`, `_model_unet_3plus_2d.py`, …).

---

## 3. Missing features

Ordered by how much they unlock, not by effort.

1. **Weight distribution.** Host the 22 `.weights.h5` (GitHub Release assets
   or Hugging Face Hub), add `tf.keras.utils.get_file` with sha256 to every
   custom constructor, ship a `WEIGHTS.md` with per-family license/attribution.
   Resolve the Apple license question before publishing MobileOne.
2. **Export validation in tests.** One parametrised test per decoder family
   (and one per custom backbone) that builds at a tiny resolution, runs
   `quantize_apply`, converts to int8 TFLite with `TFLITE_BUILTINS_INT8` only,
   and runs one inference. Mark `slow`; run in CI nightly.
3. **CI.** GitHub Actions on CPU: lint (ruff), fast unit tests on PR, slow
   TFLite matrix nightly, docs build. The suite already exists; wiring it is
   cheap.
4. **End-to-end paths for the new heads.** Detection, instance segmentation,
   pose, ReID, depth, IQA and OOD all have model code and unit tests but **no
   data loader, no target-encoder for `tf.data`, no training example**. The
   ATSS/SimOTA/DSLA assigners are NumPy functions with no documented way to
   plug them into `model.fit`. At minimum: a `DetectionAlbumentationsDataLoader`
   (COCO JSON in) and one executed notebook.
5. **Train → reparameterize → quantize pipeline.** A `TrainingDriver` option
   (or a separate `export_edge_model()`) that fuses branches, optionally runs a
   short QAT fine-tune, then converts, with per-tensor/per-channel and
   int8/uint8 I/O switches on `Quantizer`.
6. **Benchmark script.** `scripts/benchmark.py` reporting params, FLOPs
   (via `tf.profiler` or `keras-flops`), and TFLite CPU latency per
   backbone × decoder at 1–2 canonical resolutions; publish the table in docs.
7. **Generated compatibility table.** Emit backbone × decoder × tap-count ×
   QAT × TFLite status from the test run above and render it into the docs.
8. **Config-driven experiments.** A YAML/dict schema for backbone + decoder +
   loss recipe + optimizer + schedule, so `use_case_recipes.md` recipes become
   runnable artefacts. The recipes doc references EMA, SWA, rare-class batch
   sampling and domain-balanced sampling as if they existed; none do.
9. **Deterministic dataset splits** (seed + persisted split manifest) and
   generic format readers (image/mask dirs today; add COCO/VOC/CSV).
10. **Keras 3 migration decision record.** Even if the answer is "stay on
    tf-keras until tfmot moves", write it down with the trigger conditions and
    a fail-fast check at import.
11. **Package metadata modernisation.** `pyproject.toml` (and un-ignore it),
    `python_requires>=3.9`, `__version__`, optional extras
    (`[hub]`, `[efficientnet-lite]`, `[docs]`, `[porting]` for torch/timm).
12. **Documentation of the newer modules.** README, `intro.rst`, `CHANGELOG`
    and `api.rst` stop at Phase 5: PIDNet, PP-MobileSeg, StrideFormer, GFL,
    instance seg, depth, OOD, IQA, pose, ReID, uint8 wrapper and energy score
    are all undocumented outside docstrings and `plan.md`.

---

## 4. Things to improve (design and code quality)

### 4.1 API consistency

- Decoders disagree on how they address the feature pyramid:
  `lr_aspp`/`deeplab_v3_plus` take `deep_layer`/`shallow_layer`, `pp_liteseg`
  takes `feature_levels`; 3-tap (StrideFormer) and 4-tap (RepViT, FastViT)
  backbones need hand-written overrides. Have `backbone_zoo` return taps keyed
  by **output stride** (`{4: ..., 8: ..., 16: ..., 32: ...}`) and let decoders
  ask for strides. That removes a whole class of silent off-by-one choices.
- `batch_norm=False` is the default on every decoder while every backbone but
  VGG uses BN; `bach_norm_checker` warns about it. Default to `True`.
- Mutable default arguments (`filter_num_down=[...]`, `callbacks=[]`,
  `freeze_layers(model, tlist=[])` — the last one **accumulates state across
  calls**, a real bug).
- Activation strings are resolved with `eval()` in ~20 places
  (`layer_utils.py`, the U-Net family). Use `tf.keras.activations.get` or a
  dict. This matters more once configs become user-supplied (item 8 above).
- QAT is always-on: every decoder ends in `quantize_apply` with no opt-out, and
  `layer_utils.py` monkeypatches tfmot's `QuantizeWrapper`/`QuantizeLayer`
  `__init__` at import time. Make `quantize=True` a constructor argument and
  keep the patch, but apply it lazily from that path.
- PP-LiteSeg's SPPM silently assumed the stride-32 map is divisible by its
  largest bin (4): below a 128×128 input it crashed inside Keras with
  "`pool_size` ... Received: (0, 0)". Found by the new TFLite export test;
  `sppm_module` now raises a `ValueError` that states the minimum input size.
  No decoder documents its minimum input size or stride-divisibility
  requirement; a shared `check_input_size()` helper would make this uniform.
- `TrainingDriver` writes to the current directory (`tblogs/`, `.h5`, `.weights`,
  `.pdf`) with no `output_dir`; saves both a full `.h5` model and a weights
  file on every improvement; always quantizes at the end; uses deprecated
  `datetime.utcnow()`; has no LR schedule, resume, or distribution-strategy
  hooks. It is also entirely untested.

### 4.2 Structure

- Split `_model_detection.py` into `neck`, `heads`, `assigners`, `nms`,
  `instance`. Split `losses.py` by task (segmentation / detection / depth /
  ranking). Collapse the two data-loader classes onto one base.
- Retire or clearly deprecate `backbones.py` (already warns), the `hub://`
  path (hardcoded 513×513 input, `trainable=False`, tfhub.dev URLs that now
  redirect to Kaggle), and the CRPS/triplet losses inherited from upstream if
  nobody uses them.
- Move `plan.md` into `docs/` as a changelog-style history and start a short
  `ROADMAP.md` with the open items only (sections 2–3 of this file are a
  starting point). Convert the per-decision prose in `plan.md §10/§12/§13`
  into a handful of ADRs.
- Keep notebooks output-free in git (`nbstripout`), publish executed versions
  as docs build artefacts instead; delete the `examples/` duplicates.

### 4.3 Testing

- Add tests for `dataset_utils` (tiny synthetic image/mask dirs, determinism,
  PNG inputs, pickling) and `training_utils`/`quantization_utils` (one epoch
  on a 2-class toy problem, then convert).
- Replace `isfinite`-only assertions with "output is not constant" checks and,
  where a pretrained reference exists, a numeric tolerance against it (the
  porting scripts already do this against timm; make a trimmed version
  runnable in CI with cached expected outputs so torch is not required).
- The suite takes long enough that it needs `slow` markers and a fast subset.

### 4.4 Documentation

- ~270 Napoleon warnings from the `Input/Output` docstring convention; a
  mechanical migration to NumPy-style would make the autodoc clean.
- `CHANGELOG.md` jumps from 0.1.11 (2021) to 0.11.0 and omits the 0.9.x
  releases that exist in git.

---

## 5. Suggested order of work

1. **Stop the bleeding (half a day):** commit the working tree; extend
   `.gitignore` (`*.h5`, `*.tflite`, `*.weights.*`, `checkpoint`, `tblogs/`,
   `docs/source/notebooks/oxford_iiit/`, un-ignore `pyproject.toml`); move
   stray artifacts out of the repo; strip notebook outputs; declare
   `tensorflow_hub`; drop `typeguard`; add `python_requires`.
2. **Make installation true (1 day):** `pyproject.toml`; EfficientNet-Lite as
   an optional extra or vendored; add a fail-fast Keras-2 check at import.
3. **Make "edge deployable" true (2–3 days):** TFLite export tests, per-tensor
   switch, reparam step in `Quantizer`, benchmark script, compatibility table.
4. **Ship weights (2 days + license review):** hosting, `get_file`, checksums,
   `WEIGHTS.md`.
5. **CI (half a day)** once 1–3 exist.
6. **Close the loop on the new heads (1–2 weeks):** detection loader +
   target encoder + notebook; same pattern for pose/depth as demand dictates.
7. **Data loader rewrite (1 week, behind a benchmark):** deterministic split,
   `decode_image`, no broker/workers unless the benchmark justifies it.
8. **Docs and plan consolidation** as each item lands.

---

## 6. What is in good shape

Worth saying explicitly so the list above is read in proportion:

- The five custom backbone families are implemented from primary sources,
  cross-checked against timm, and have **numerically verified** weight ports
  with the conversion scripts committed. That is rare and valuable.
- The `BACKBONE_REGISTRY` refactor removed `eval()` dispatch from the zoo and
  is the right extension point.
- Loss coverage for hard/imbalanced segmentation (Lovász, boundary, structure,
  OHEM, focal variants) and for anchor-free detection (QFL/DFL/VFL, CIoU/SIoU)
  is broad and tested.
- Mixed-precision + QAT interaction was investigated empirically and fixed
  with tests, not papered over.
- The plan file is an unusually honest engineering log; the problem is only
  that it is now the sole source of truth.

---

## 7. Evidence

| Claim | Where |
|---|---|
| Direct-URL dependency, missing `tensorflow_hub`, unused `typeguard`, Python 3.6 classifier | `setup.py:16-31, 36-67`; `swiss_army_keras/_backbone_zoo.py:61` |
| EfficientNet-Lite import failure on 3.12 | `swiss_army_keras/_backbone_zoo.py:65-82` (warning observed on `import swiss_army_keras.models`) |
| `.gitignore` ignores `setup.py`/`pyproject.toml` | `.gitignore:3-6` |
| `weights='imagenet'` raises for custom families | `_backbone_mobileone.py:239-249`, `_backbone_mobilenetv4.py:337-347`, `_backbone_repvit.py:245-256`, `_backbone_fastvit.py:399`, `_backbone_strideformer.py:527-530` |
| Weights hosted nowhere; license check open | `.gitignore:140-145`; `plan.md §12.1 (l.2507)`, `§12.2-12.5` "Not done" bullets |
| Legacy-Keras env var, no fail-fast | `swiss_army_keras/__init__.py:12`; `tests/conftest.py:16` |
| Only TFLite conversion is in `Quantizer`, untested | `swiss_army_keras/quantization_utils.py:56-117`; `grep TFLiteConverter tests/` → none |
| uint8 I/O hardcoded, no per-tensor switch | `quantization_utils.py:79-87` |
| Reparam never wired into export | `reparameterize_*` defined in `_backbone_mobileone.py:346`, `_backbone_fastvit.py:527`, `_backbone_repvit.py:307`; no reference in `quantization_utils.py`/`training_utils.py` |
| Open all-zero QAT output bug | `plan.md §10 (l.2218-2266)` |
| Unseeded split | `dataset_utils.py:535-540` |
| `__getstate__` deletes non-existent `pool` | `dataset_utils.py:336-339`, `795-798` |
| `decode_jpeg` only, mask path by string split | `dataset_utils.py:178-179, 450, 904` |
| Broker/workers spawned in `__init__`, hardcoded CURVE keys, `/tmp` IPC | `dataset_utils.py:72-78, 255-274` |
| `freeze_layers` mutable default | `training_utils.py:27` |
| `TrainingDriver` cwd outputs, utcnow, always quantizes | `training_utils.py:100-104, 232-242` |
| `eval(activation)` sites | `layer_utils.py:151,216,247-248,307,318,350,378,425,491`; `_model_att_unet_2d.py:119,255`; `_model_r2_unet_2d.py:33,194,264`; `_model_resunet_a_2d.py:142,244`; `_model_unet_plus_2d.py:72`; `_model_transunet_2d.py:38`; `_model_u2net_2d.py:418` |
| tfmot monkeypatch at import | `layer_utils.py:56-88` |
| Inconsistent pyramid addressing | `_model_lr_aspp.py:87-90` vs `_model_pp_liteseg.py:184-187`; `_backbone_zoo.py:312-335` (StrideFormer 3-tap note) |
| Debug `print()` in model files | `_model_deeplab_v3_plus.py:205,228-230`; `_model_u2net_2d.py` (15), `_model_unet_3plus_2d.py` (10), `_model_unet_plus_2d.py` (7) |
| Docs/README stop at Phase 5 | `README.md` features list; `docs/source/intro.rst:20-27`; `docs/source/api.rst` (no pidnet/pp_mobileseg/strideformer/depth/ood/iqa/training/dataset modules) |
| Recipes reference non-existent EMA/SWA/sampling | `use_case_recipes.md:57-72,127`; no match in `swiss_army_keras/` |
| Repo bloat | `du`: 3.7 GB tree, 1.2 GB `pretrained_weights/`, 1.8 GB `docs/source/notebooks/`; 12 `*.h5`/`*.tflite` at root; `git ls-files` shows a 12 MB tracked notebook |
| No CI/lint | no `.github/`; `ruff`/`pyflakes`/`flake8` absent from venv and PATH |

### 7.1 Test run

Recorded in §8.2.

---

## 8. Actions taken on 2026-11-04 (after the assessment above was written)

### 8.1 Code changes

| Change | Files |
|---|---|
| Committed all pending work, including the six never-tracked modules | commit `d7d6d21` |
| `.gitignore`: training/export artifacts, datasets, `tblogs/`, PDFs; un-ignored `pyproject.toml`/`setup.cfg`/`setup.py` | `.gitignore` |
| Packaging: version read from `swiss_army_keras.__version__`; `python_requires>=3.9`; corrected classifiers and URL; declared `tensorflow_hub`; dropped unused `typeguard`; EfficientNet-Lite moved to an `[efficientnet-lite]` extra so the metadata has no direct VCS URL; `[porting]`, `[docs]`, `[test]` extras | `setup.py`, `swiss_army_keras/__init__.py`, `requirements.txt`, `README.md` |
| Fail-fast guard: importing the package in a process where `tf.keras` is Keras 3 now raises an explicit `ImportError` with the fix, instead of an opaque tfmot error later | `swiss_army_keras/_compat.py` (new), `layer_utils.py` |
| Data loaders: optional `seed=` for a reproducible train/val/test split; `decode_image` instead of `decode_jpeg`; `__getstate__` no longer raises `KeyError` on a non-existent `pool`; `logging.warn` → `logging.warning` | `dataset_utils.py` |
| Training driver: `freeze_layers` mutable-default bug; `callbacks=()`; timezone-aware timestamp | `training_utils.py` |
| PP-LiteSeg: explicit minimum-input-size `ValueError` in `sppm_module` | `_model_pp_liteseg.py` |
| New `slow`-marked TFLite export tests: QAT decoder → full-int8 TFLite (`TFLITE_BUILTINS_INT8`, uint8 I/O) → interpreter run, non-constant output. LR-ASPP and PP-LiteSeg on MobileOne-S0 both pass. | `tests/test_tflite_export.py`, `pytest.ini` |

Not done here (still open from §3): weight hosting, CI, benchmark harness,
per-tensor quantization switch, reparameterize-before-export, detection data
path, docs for the post-Phase-5 modules, Keras 3 decision record.

### 8.2 Test run

Environment: project `.venv`, Python 3.12.3, TensorFlow 2.21.0, tf-keras
2.21.0, CPU only. No training jobs were run; every test builds
randomly-initialised models.

| Chunk | Files | Result | Time |
|---|---|---|---|
| A | `test_activations`, `test_backbone_*` (6 files) | 131 passed | 4m08s |
| B | `test_classifiers`, `test_decoders` | 66 passed | 6m33s |
| C | `test_detection`, `test_inference_utils`, `test_iqa_utils`, `test_layer_utils`, `test_losses` | 118 passed after the flaky-test fix below (117 + 1 on the first pass) | 3m43s |
| D | `test_metrics`, `test_mixed_precision_qat`, `test_model_depth`, `test_ood_utils`, `test_optimizers`, `test_reparam_utils`, `test_utils` | 96 passed | 3m19s |
| slow | `test_tflite_export` (new) | 2 passed | 2m27s |

The suite was run in four foreground chunks because a single run takes about
16 minutes. One earlier uninterrupted single-process run reached 84% with no
failures before it was cut off, so nothing suggests cross-file order
dependence, but a clean single-process run of the whole suite is still worth
doing once in CI.

Two things worth knowing:

- **Flaky test, fixed.**
  `test_coordatt_lite_gives_different_rows_different_attention_unlike_eca`
  failed once in chunk C and passed 6/6 when rerun alone. It built an unseeded
  random-init `CoordAttLite` and asserted two gates differ by more than 0.1%.
  It now pins the seed. Other tests that compare values from random-init
  layers without a seed are candidates for the same treatment.
- **Do not edit package sources while the suite is running.** A run that
  overlapped with an edit to `layer_utils.py` reported 18 failures in the
  GeM/ECA/CoordAtt tests (`KeyError: '__class__'`, `math.log` on a symbolic
  tensor). Cause: Keras autograph-converts a custom layer's `call` lazily by
  re-reading its source from disk, so lines shifted by the edit made it
  convert the wrong function. All 18 pass on an undisturbed tree. Not a code
  defect, but it will confuse anyone using a file-watcher test loop.

Runtime note for CI planning: the ten slowest tests take 12 to 25 seconds
each (depth heads, static-batch detection, PIDNet). A `-m "not slow"` PR job
of about 15 to 20 minutes on a 4-core runner is realistic; splitting by file
across 3 or 4 jobs brings it under 7 minutes.

---

## 9. Follow-up on 2026-11-05: documentation pass and candidates assessment

### 9.1 Documentation

| Change | Where |
|---|---|
| Docs build went from 265 docutils warnings to 0. A Sphinx hook renders the package's `Input` / `Output` docstring convention instead of rewriting several hundred docstrings | `docs/source/conf.py` |
| New guides: getting started, model zoo (the single detailed inventory), deployment | `docs/source/getting_started.rst`, `model_zoo.rst`, `deployment.rst` |
| API reference now covers every module (it stopped at the Phase 5 modules) | `docs/source/api.rst` |
| README rewritten around the actual feature set, with a verified quick start; intro page reduced to a summary that points at the model zoo | `README.md`, `docs/source/intro.rst` |
| Docstrings written for the user-facing classes that had none: `TrainingDriver`, `ModelBuilder`, `Quantizer`, both data loaders, `MeanIOU`, `loss_adapter`, `distiller_classifier`, the learnable-resizer classifiers; module docstrings for the public modules | `swiss_army_keras/` |
| Wrong docstrings corrected: `deeplab_v3_plus` and `deeplab_v3_plus_lite` carried the UNet 3+ text; `unfreeze_model` carried the `freeze_model` text | `_model_deeplab_v3_plus.py`, `utils.py` |
| `CONTRIBUTING.md` (setup, tests, documentation rules), `ROADMAP.md` (ordered open work), changelog entry for the unreleased work, implementation-status note in `use_case_recipes.md` | repository root |
| Docs version read from `swiss_army_keras.__version__`; notebooks never re-executed at build time; the MobileNetV4 notebook added to the examples page | `docs/source/conf.py`, `examples.rst` |

### 9.2 New findings from this pass

- **Reparameterization is never applied on export.** A MobileOne-S0 feature
  extractor exports to 416 TFLite operators and 4.2 MB with its training-time
  branches, against 50 operators and 1.0 MB after `reparameterize_mobileone`
  (outputs agree to 4e-4). The fusion functions only accept a backbone built
  directly from its constructor, not one nested in a decoder, so no model
  trained through `TrainingDriver` can be fused today.
- **Exports contain operators that Axis does not list for its DLPUs**:
  SPACE_TO_BATCH_ND from DeepLabV3+'s dilated convolutions, GELU and
  squeeze-excite gates in RepViT, MEAN and REDUCE_MAX in PP-LiteSeg, and
  DEQUANTIZE / QUANTIZE float islands in LR-ASPP and DeepLabV3+.
  Details in `EDGE_AI_CANDIDATES.md` section 3.
- **`unfreeze_model` unfreezes the backbone's batch normalization.** It only
  inspects top-level layers; the nested backbone is switched as a whole, so
  all 226 batch-norm layers of a MobileOne-S0 backbone become trainable in
  the unfrozen phase although `unfreeze_batch_norm=False`.
- **`TrainingDriver` never saved its metric plots** with the default name:
  `os.makedirs('')` raised and the error was only logged. Fixed.
- **Non-square inputs do not work in the data loaders**: images are resized
  to `(height, width)` but shapes are set as `(width, height)`.
- **`use_case_recipes.md` recommends features that do not exist** (EMA, SWA,
  class- and domain-balanced sampling, loss ramping, differential learning
  rate). Now stated at the top of that file.
- Many docstrings in the newer modules narrate development history and
  refer to `plan.md` sections. They render correctly now, but they are long
  and hard to use as reference. Trimming them is listed in `ROADMAP.md`.

### 9.3 Candidates assessment

`EDGE_AI_CANDIDATES.md` assesses further networks, optimizers and strategies
against three hardware tiers (Axis DLPUs, other aarch64 stacks, Jetson Orin).
Its main conclusion is that deployment tooling (per-tensor profiles, operator
linting, reparameterization at export, cross-layer equalization) and plain
convolutional components (ResNet-18/34, STDC, a gate-free decoder, an
NMS-free detection head) are worth more on Axis hardware than further
mobile-transformer families.

---

## 10. Follow-up on 2026-11-05: export pipeline

Implemented the first block of `ROADMAP.md`.

| Change | Where |
|---|---|
| `reparameterize_model`: fuses a MobileOne, RepViT or FastViT backbone nested in any model (or standalone), reusing the existing per-family functions | `swiss_army_keras/deploy_utils.py` |
| Deployment profiles (`default`, `litert-cpu`, `artpec7`, `artpec8`, `artpec9`): weight granularity, I/O type, operator list, channel multiple | same |
| `convert_to_tflite` and `lint_tflite`: conversion per profile and a check of the result (types, float operators, weight granularity, operators outside the target's list, per-operator constraints, channel counts) | same |
| `strip_qat`: plain float copy of a QAT-wrapped model with the trained weights | same |
| `Quantizer(profile=, reparameterize=, strip_qat=, strict=)` runs the pipeline and keeps the report; `TrainingDriver(export_profile=)` | `quantization_utils.py`, `training_utils.py` |
| Tests: 16 fast tests on small models, 8 new slow end-to-end export tests | `tests/test_deploy_utils.py`, `tests/test_tflite_export.py` |

Findings made while building it:

- **QAT-wrapped heads do not export cleanly for integer-only targets.** In
  LR-ASPP and DeepLabV3+ the converter leaves the backbone's last
  convolution in floating point. Cause: `tfmot` attaches an output
  quantizer to the head's pooling layer, pooling must use one scale for
  input and output, and that scale disagrees with the calibrated scale the
  sibling convolution needs on the same backbone tap. Un-annotating the
  pooling layer does not help (`tfmot` adds the output quantizer anyway),
  and neither does an explicit quantizer on the tap.
- **QAT wrappers keep convolution weights per channel** even when the
  converter is told to quantize per tensor.
- Both are solved by stripping the wrappers at export and quantizing the
  whole graph post-training. All five models probed (LR-ASPP, PP-LiteSeg,
  DeepLabV3+ and a classifier, on MobileOne-S0 and MobileNetV4-Conv-S)
  export without errors that way; four of them had errors with the wrappers.
- Consequence: **on Axis targets the QAT wrapping no longer provides
  activation ranges.** It still exposes the head to quantization noise
  during training, but its value is now an open question (roadmap, sections
  1 and 7). `tfmot` does offer a per-tensor QAT scheme
  (`Default8BitQuantizeScheme(disable_per_axis=True)`), which fixes the
  weight granularity but not the float operator.
- Measured effect on `lr_aspp` + MobileOne-S0 at 128x128: 432 operators and
  5.5 MB as built, 61 operators and 1.4 MB after the pipeline with the
  `artpec8` profile.
- Not verified: accuracy of the per-tensor post-training export on a trained
  model. No training was run. This needs a real dataset and, for ARTPEC
  targets, a device.

---

## 11. Follow-up on 2026-11-05: quantization accuracy

| Change | Where |
|---|---|
| `equalize_cross_layer`: cross-layer equalization on plain models, through batch norm, ReLU and padding, inside nested backbones; in place, with `restore()` | `swiss_army_keras/deploy_utils.py` |
| `quantization_error`: SQNR, RMSE and maximum error of a TFLite export against the float model, per output | same |
| `Quantizer(equalize=)`: on by default for per-tensor profiles after stripping; original weights restored after conversion | `quantization_utils.py`, `training_utils.py` |
| `lr_aspp(gate=False)`, `pp_liteseg(fusion='add')`: variants without sigmoid gating | `_model_lr_aspp.py`, `_model_pp_liteseg.py` |
| Tests: 5 new fast tests for equalization and error measurement, 2 decoder tests, 2 slow export tests | `tests/` |

Findings:

- **Critical point C5 is explained, and is not a defect.** The all-zero
  output of QAT-wrapped heads on a randomly initialized MobileNetV2 comes
  from the untrained output quantizer: its initial range of -6 to 6 rounds
  to steps of 12/255, and that model's float logits are around 0.001. The
  stripped float model produces the expected small, non-constant values.
- **Measured on ImageNet-pretrained weights** (Pets images, feature-level
  SQNR; full table in `docs/source/deployment.rst`):
  - Equalization preserves the float function (relative change about 1e-5)
    and lifts per-tensor MobileNetV4-Conv-S from 6.7 to 14.6 dB at the
    deepest tap, close to its per-channel 19.2 dB.
  - Fusing MobileOne improves its quantization quality, so fusing by default
    is right. But fused MobileOne and RepViT stay near 3 dB at the deepest
    tap even per channel: post-training quantization of the reparameterized
    families is poor.
  - MobileNetV2 falls to about -3 dB per tensor and cannot be equalized
    because of ReLU6. Replacing ReLU6 by ReLU is not an option for the Keras
    weights: it changes the features by 50 to 110%.
- **Practical consequence for Axis ARTPEC-8:** MobileNetV4-Conv with
  equalization is the best-measured backbone in the library today.
- **Not verified:** task accuracy. These are feature-level proxies on 48
  images with no trained head. A trained model per profile is the next
  measurement to make (roadmap section 2).

## 12. Follow-up on 2026-11-05: depth, out-of-distribution analysis, shared backbones

The question was whether out-of-distribution analysis and monocular depth
are pluggable. Both were at the model level only. Probes with random
weights found these gaps, all closed in this pass:

| Gap found | What was added | Where |
|---|---|---|
| `fastdepth_head` failed on RepViT, FastViT and StrideFormer; `lr_aspp` and `lr_aspp_depth` failed on them with default arguments | Heads adapt to the number of feature levels; `lr_aspp` defaults follow the backbone and a missing level is a clear error | `_model_depth.py`, `_model_lr_aspp.py`, `_backbone_zoo.py` |
| Depth export failed the ARTPEC-8 check: `LOGISTIC` is not on the list as read, although the docstring recommended the sigmoid output for that chip | `drop_output_activations` exports without the final sigmoid or softmax; docstring corrected | `deploy_utils.py`, `quantization_utils.py`, `training_utils.py` |
| No training path for depth | `DepthAlbumentationsDataLoader`, depth metrics, `berhu_loss(mask_invalid=True)` | `dataset_utils.py`, `metrics.py`, `losses.py` |
| Every builder created its own backbone | `multi_task`: several heads on one backbone, named outputs | `_model_multi_task.py` |
| No builder exposed the embedding Mahalanobis needs | `add_embedding_output`, working on trained models | `layer_utils.py` |
| Mahalanobis scoring needed SciPy; no evaluation | `whiten_statistics`, `whitened_distance`, `sample_dense_embeddings`, `ood_detection_metrics` | `ood_utils.py` |
| The converter reorders the outputs of a multi-output model | `tflite_outputs` maps file positions to head names | `deploy_utils.py` |

Findings that remain:

- **The energy-score wrapper cannot run on the Axis accelerators** (`EXP`,
  `LOG`, `SUM` are not listed). Scores are computed on the host from the
  exported logits; the deployment guide says how.
- **`PAD` is not on the ARTPEC-8 list as read**, and MobileNetV4 and
  MobileOne exports contain it. This affects every head on those backbones.
  It needs the device check of roadmap section 1 before anything is changed.
- **`classifier` uses swish in its hidden layer**, which needs a sigmoid on
  ARTPEC-8; `activation='relu'` avoids it.
- **The image-quality head stays in floating point** with its default GeM
  pooling.
- **Nothing here was trained.** Builds, exports and hand-computed values
  were tested; depth accuracy and out-of-distribution detection quality on
  real data are unmeasured.
