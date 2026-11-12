"""`TrainingDriver` end to end on a toy task: the two float phases, the
whole-model QAT phase, the record of the split and the export."""

import json
import os

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras import deploy_utils, models
from swiss_army_keras.training_utils import TrainingDriver

pytestmark = pytest.mark.slow

SIZE = 64


def _datasets(targets=1):
    rng = np.random.default_rng(0)
    images = rng.random((16, SIZE, SIZE, 3)).astype(np.float32)
    labels = (images[..., 0] > 0.5).astype(np.int32)
    target = labels if targets == 1 else tuple(labels for _ in range(targets))
    dataset = tf.data.Dataset.from_tensor_slices((images, target)).batch(4)
    return dataset, dataset, dataset


class _Loader:
    """Stands in for a data loader: only `save_split` is used."""

    def save_split(self, path):
        with open(path, "w") as f:
            json.dump({"version": 1, "train": [], "val": [], "test": []}, f)
        return path


def _loss():
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def test_driver_runs_a_qat_phase_and_exports_its_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = models.fpn_seg(tf.keras.Input((SIZE, SIZE, 3)), 2, backbone="ResNet18", weights=None)
    train, val, test = _datasets()
    driver = TrainingDriver(
        model, "toy", tf.keras.optimizers.Adam(1e-3), _loss(), [], train, val, test, epochs=1,
        unfreezed_epochs=0, plot_metrics=False, checkpoint_name="toy", quantizer_name="toy",
        export_profile="artpec8", qat_epochs=1, qat_learning_rate=1e-4, qat_calibration_batches=2,
        data_loader=_Loader(), quant_batches=2)
    driver.run()

    assert os.path.exists("toy_split.json")
    assert driver.qat_model is not None and driver.qat_model is not driver.model
    assert driver.qat_model.qat_per_channel is False
    assert "val_loss" in driver.qat_history.history
    assert os.path.exists("toy_qat.weights.index")
    # the exported file is the QAT model, with its wrappers, and matches the profile
    assert driver.quantizer.model is driver.qat_model and driver.quantizer.strip_qat is False
    assert not driver.quantizer.lint_report.errors
    assert os.path.exists(driver.quantizer.tflite_int_path)


def test_driver_qat_phase_with_named_outputs_and_selected_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = models.ddrnet(tf.keras.Input((SIZE, SIZE, 3)), 2, augment=True)
    names = model.output_names
    assert len(names) == 2
    train, val, test = _datasets(targets=2)
    driver = TrainingDriver(
        model, "toy2", tf.keras.optimizers.Adam(1e-3), {name: _loss() for name in names}, None, train, val, test,
        epochs=1, unfreezed_epochs=0, plot_metrics=False, checkpoint_name="toy2", quantizer_name="toy2",
        export_profile="artpec9", loss_weights={names[0]: 1.0, names[1]: 0.4}, export_outputs=0, qat_epochs=1,
        qat_calibration_batches=2, quant_batches=2)
    driver.run()

    assert driver.qat_model.qat_per_channel is True and len(driver.qat_model.outputs) == 2
    assert len(deploy_utils.tflite_outputs(driver.quantizer.tflite_ui8_model)) == 1
    assert not driver.quantizer.lint_report.errors
