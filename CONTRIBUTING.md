# Contributing

## Setup

    python3 -m venv .venv && . .venv/bin/activate
    pip install -e ".[test,docs]"

Python 3.9 or newer. The EfficientNet-Lite backbones are optional
(`[efficientnet-lite]` extra, Python 3.11 at most); weight conversion needs
the `[porting]` extra (torch, timm).

Always `import swiss_army_keras` before `import tensorflow` (see the README).
`tests/conftest.py` does the equivalent for the test suite.

## Tests

    pytest -m "not slow"     # unit tests
    pytest -m slow           # full-integer TFLite conversions

- Tests build randomly initialized models only: use `weights=None`, never
  download weights or train for more than a few steps.
- Pin the seed (`tf.keras.utils.set_random_seed`) in any test that compares
  values produced by randomly initialized layers.
- Assert more than finiteness: a degenerate all-zero output is finite.
  Prefer "output is not constant" or a numeric reference.
- An untrained QAT-wrapped model can legitimately output exact zeros (its
  output quantizer rounds to steps of 12/255). When a test needs
  non-degenerate values from a random model, assert on
  `deploy_utils.strip_qat(model)` or on the nested backbone.
- Every new head or backbone family gets a case in
  `tests/test_tflite_export.py`.
- Export behaviour (profiles, lint, QAT stripping, reparameterization) is
  tested on small models in `tests/test_deploy_utils.py`.
- A new reparameterizable backbone family must be added to
  `deploy_utils._reparam_families` so it is fused at export.
- Do not edit files under `swiss_army_keras/` while the suite is running.
  Keras re-reads a custom layer's source from disk when it first traces it,
  so a concurrent edit makes unrelated tests fail.

## Documentation

    python -m sphinx -b html docs/source docs/build/html

The build must stay free of docutils warnings from docstrings.

- **Docstring convention.** A summary paragraph, then `Input` and `Output`
  sections, each underlined with dashes, with one `name: description` entry
  per argument. `docs/source/conf.py` renders this convention; do not mix in
  Google or NumPy style.
- Every public function and class has a docstring. Say what the function
  returns (logits or probabilities, output resolution, extra outputs).
- Docstrings describe the code as it is. History, debugging accounts and
  references to planning documents belong in the commit message or in
  `plan.md`, not in the docstring.
- When a component is added, update **one** inventory:
  `docs/source/model_zoo.rst`. `README.md` and `docs/source/intro.rst` only
  carry a short summary; update them only if a new area appears.
- Export-related behaviour goes in `docs/source/deployment.rst`.
- Add an entry to `CHANGELOG.md` under "Unreleased".

## Conventions

- One `_model_<name>.py` file per architecture; re-export full models from
  `models.py` and feature builders from `base.py`.
- Register backbones in `BACKBONE_REGISTRY` (`_backbone_zoo.py`) with their
  feature taps and preprocessing function.
- Wrap quantizable layers with `layer_utils.qat_annotate_layer` and end the
  builder with `quantize_apply`, as the existing heads do.
- No training artifacts in git: `.gitignore` excludes `*.h5`, `*.tflite`,
  checkpoints, `tblogs/` and datasets.
- The version lives in `swiss_army_keras/__init__.py` (`__version__`);
  `setup.py` and the docs read it from there.
