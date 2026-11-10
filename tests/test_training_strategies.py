"""Schedules, weight-decay presets, per-group optimizers, weight averaging
and sharpness-aware training (added on 2026-11-10).

The fits run a few steps on a handful of random numbers: they check
mechanics, not learning.
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras.optimizers import (
    SAM,
    MultiOptimizer,
    OneCycle,
    WarmupCosine,
    adamw,
    differential_learning_rate,
    no_decay_variables,
    sgdw,
)
from swiss_army_keras.training_utils import (
    SharpnessAwareModel,
    WeightAveraging,
    recalibrate_batch_norm,
    sharpness_aware,
)

L = tf.keras.layers


def _toy(name="toy"):
    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input((8,))
    backbone_in = tf.keras.Input((8,))
    hidden = L.BatchNormalization(name="bb_bn")(L.Dense(16, name="bb_dense")(backbone_in))
    backbone = tf.keras.Model(backbone_in, L.Activation("relu")(hidden), name="Toy_backbone")
    out = L.Dense(2, name="head")(backbone(inp))
    return tf.keras.Model(inp, out, name=name)


def _data(n=32):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 8)).astype("float32")
    return x, (x[:, :2] * 2.0 + 1.0).astype("float32")


# --------------------------------------------------------------------------- #
# schedules

def test_warmup_cosine_shape():
    schedule = WarmupCosine(peak=1.0, total_steps=100, warmup_steps=10, warmup_start=0.1, final=0.01)
    values = [float(schedule(step)) for step in (0, 5, 10, 55, 100, 500)]
    np.testing.assert_allclose(values[0], 0.1, rtol=1e-6)
    np.testing.assert_allclose(values[1], 0.55, rtol=1e-6)          # halfway up
    np.testing.assert_allclose(values[2], 1.0, rtol=1e-6)
    np.testing.assert_allclose(values[3], 0.01 + 0.99 * 0.5, rtol=1e-5)   # halfway down the cosine
    np.testing.assert_allclose(values[4:], [0.01, 0.01], rtol=1e-5)
    assert WarmupCosine(1.0, 200, warmup_steps=0.05).warmup_steps == 10      # a fraction of the total
    clone = WarmupCosine.from_config(schedule.get_config())
    assert float(clone(55)) == pytest.approx(values[3])


def test_one_cycle_shape():
    schedule = OneCycle(peak=1.0, total_steps=100, pct_start=0.3, div=25.0, final_div=100.0)
    np.testing.assert_allclose(float(schedule(0)), 1 / 25, rtol=1e-6)
    np.testing.assert_allclose(float(schedule(30)), 1.0, rtol=1e-6)
    np.testing.assert_allclose(float(schedule(100)), 1 / 2500, rtol=1e-4)
    values = np.array([float(schedule(step)) for step in range(101)])
    assert values.argmax() == 30 and np.all(np.diff(values[:31]) > 0) and np.all(np.diff(values[30:]) < 0)


def test_schedule_drives_an_optimizer():
    model = _toy()
    optimizer = tf.keras.optimizers.SGD(WarmupCosine(0.1, total_steps=8, warmup_steps=2))
    model.compile(optimizer=optimizer, loss="mse")
    x, y = _data()
    model.fit(x, y, batch_size=8, epochs=2, verbose=0)
    assert int(optimizer.iterations) == 8


# --------------------------------------------------------------------------- #
# weight decay

def test_decay_presets_skip_biases_and_normalization_parameters():
    model = _toy()
    skipped = {v.name for v in no_decay_variables(model)}
    assert any("bias" in name for name in skipped) and any("gamma" in name for name in skipped)
    assert not any("kernel" in name for name in skipped)

    for make in (adamw, sgdw):
        model = _toy()
        optimizer = make(model, learning_rate=0.1, weight_decay=0.5)
        # a loss with zero gradient everywhere: only the weight decay acts
        model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: 0.0 * tf.reduce_mean(y_pred))
        before = {v.name: v.numpy().copy() for v in model.trainable_variables}
        x, y = _data(8)
        model.fit(x, y, batch_size=8, epochs=1, verbose=0)
        after = {v.name: v.numpy() for v in model.trainable_variables}
        for name in before:
            changed = np.abs(after[name] - before[name]).max() > 0
            assert changed == ("kernel" in name), name


# --------------------------------------------------------------------------- #
# per-group optimizers

def test_differential_learning_rate_moves_the_backbone_less():
    model = _toy()
    optimizer = differential_learning_rate(model, lambda lr: tf.keras.optimizers.SGD(lr), 0.1, backbone_factor=0.0)
    assert isinstance(optimizer, MultiOptimizer)
    model.compile(optimizer=optimizer, loss="mse")
    backbone_before = model.get_layer("Toy_backbone").get_layer("bb_dense").get_weights()[0].copy()
    head_before = model.get_layer("head").get_weights()[0].copy()
    x, y = _data()
    history = model.fit(x, y, batch_size=8, epochs=2, verbose=0).history

    assert history["loss"][-1] < history["loss"][0]
    np.testing.assert_array_equal(model.get_layer("Toy_backbone").get_layer("bb_dense").get_weights()[0], backbone_before)
    assert np.abs(model.get_layer("head").get_weights()[0] - head_before).max() > 0
    assert int(optimizer.iterations) == 8
    assert float(optimizer.learning_rate) == 0.0                       # the first group is the backbone

    with pytest.raises(ValueError, match="no nested"):
        plain = tf.keras.Sequential([tf.keras.Input((8,)), L.Dense(2)])
        differential_learning_rate(plain, lambda lr: tf.keras.optimizers.SGD(lr), 0.1)


def test_multi_optimizer_needs_an_owner_for_every_variable():
    model = _toy()
    optimizer = MultiOptimizer([(tf.keras.optimizers.SGD(0.1), lambda v: "head" in v.name)])
    with pytest.raises(ValueError, match="no optimizer accepts"):
        optimizer.build(model.trainable_variables)
    with pytest.raises(ValueError, match="at least one"):
        MultiOptimizer([])


# --------------------------------------------------------------------------- #
# weight averaging

class _Recorder(tf.keras.callbacks.Callback):
    """Stores the head kernel that later callbacks see at the end of each epoch."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def on_epoch_end(self, epoch, logs=None):
        self.seen.append(self.model.get_layer("head").get_weights()[0].copy())


def test_ema_weights_are_used_for_validation_checkpoints_and_the_final_model():
    model = _toy()
    model.compile(optimizer=tf.keras.optimizers.SGD(0.05), loss="mse")
    x, y = _data()
    start = model.get_layer("head").get_weights()[0].copy()
    recorder = _Recorder()
    averaging = WeightAveraging(mode="ema", decay=0.5, warmup=False)
    model.fit(x, y, batch_size=8, epochs=3, verbose=0, validation_data=(x, y), callbacks=[averaging, recorder])

    final = model.get_layer("head").get_weights()[0]
    np.testing.assert_allclose(final, averaging._average[-2].numpy() if final.shape != averaging._average[-1].shape
                               else averaging._average[-1].numpy(), atol=1e-7)
    np.testing.assert_allclose(recorder.seen[-1], final, atol=1e-7)     # a later callback saw the averaged weights
    assert np.abs(final - start).max() > 0


def test_ema_average_matches_the_recurrence():
    """With decay d and no warm-up: a_0 = w_0 (first update), then
    a_n = d * a_{n-1} + (1 - d) * w_n. Checked by replaying the weights.
    """
    model = _toy()
    model.compile(optimizer=tf.keras.optimizers.SGD(0.05), loss="mse")
    x, y = _data(16)
    history = []

    class Tap(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            history.append(self.model.get_layer("head").get_weights()[0].copy())

    averaging = WeightAveraging(mode="ema", decay=0.8, warmup=False)
    model.fit(x, y, batch_size=8, epochs=2, verbose=0, callbacks=[Tap(), averaging], shuffle=False)
    expected = history[0]
    for weights in history[1:]:
        expected = 0.8 * expected + 0.2 * weights
    np.testing.assert_allclose(model.get_layer("head").get_weights()[0], expected, rtol=1e-5, atol=1e-6)


def test_swa_is_the_mean_of_the_epoch_end_weights():
    model = _toy()
    model.compile(optimizer=tf.keras.optimizers.SGD(0.05), loss="mse")
    x, y = _data(16)
    ends = []

    class Tap(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            ends.append(self.model.get_layer("head").get_weights()[0].copy())

    averaging = WeightAveraging(mode="swa", start_epoch=1)
    model.fit(x, y, batch_size=8, epochs=4, verbose=0, callbacks=[Tap(), averaging], shuffle=False)
    np.testing.assert_allclose(model.get_layer("head").get_weights()[0], np.mean(ends[1:], axis=0),
                               rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="Unknown mode"):
        WeightAveraging(mode="median")


def test_recalibrate_batch_norm_estimates_statistics_without_training():
    model = _toy()
    x, _ = _data(64)
    batch_norm = model.get_layer("Toy_backbone").get_layer("bb_bn")
    kernel_before = model.get_layer("head").get_weights()[0].copy()
    dataset = tf.data.Dataset.from_tensor_slices(x).batch(16)

    assert recalibrate_batch_norm(model, dataset, steps=4) == 1
    dense = tf.keras.Model(model.get_layer("Toy_backbone").input,
                           model.get_layer("Toy_backbone").get_layer("bb_dense").output)
    activations = dense.predict(x, verbose=0)
    np.testing.assert_allclose(batch_norm.moving_mean.numpy(), activations.mean(axis=0), atol=1e-4)
    assert batch_norm.momentum == 0.99                                  # restored
    np.testing.assert_array_equal(model.get_layer("head").get_weights()[0], kernel_before)


# --------------------------------------------------------------------------- #
# sharpness-aware training

@pytest.mark.parametrize("adaptive", [False, True], ids=["sam", "asam"])
def test_sharpness_aware_model_trains_and_shares_layers(adaptive):
    model = _toy()
    wrapped = sharpness_aware(model, rho=0.5 if adaptive else 0.05, adaptive=adaptive)
    assert isinstance(wrapped, SharpnessAwareModel)
    wrapped.compile(optimizer=tf.keras.optimizers.SGD(0.05), loss="mse")
    x, y = _data()
    before = model.get_layer("head").get_weights()[0].copy()
    history = wrapped.fit(x, y, batch_size=8, epochs=3, verbose=0).history

    assert history["loss"][-1] < history["loss"][0]
    assert np.all(np.isfinite(history["loss"]))
    assert np.abs(model.get_layer("head").get_weights()[0] - before).max() > 0      # the original model is trained


def test_sam_adaptive_option_scales_the_perturbation_with_the_weights():
    variable = tf.Variable([1.0, 10.0])
    gradient = tf.constant([1.0, 1.0])
    plain, adaptive = SAM(tf.keras.optimizers.SGD(0.1), rho=1.0), SAM(tf.keras.optimizers.SGD(0.1), rho=1.0, adaptive=True)
    plain.first_step([gradient], [variable])
    np.testing.assert_allclose(variable.numpy() - [1.0, 10.0], [1 / np.sqrt(2)] * 2, rtol=1e-5)
    variable.assign([1.0, 10.0])
    adaptive.first_step([gradient], [variable])
    step = variable.numpy() - [1.0, 10.0]
    # ASAM: e = rho * w^2 g / ||w g||
    np.testing.assert_allclose(step, np.array([1.0, 100.0]) / np.sqrt(101.0), rtol=1e-5)
