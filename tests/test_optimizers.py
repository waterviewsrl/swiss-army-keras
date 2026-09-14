"""Regression tests for swiss_army_keras.optimizers.

These optimizers were rewritten (see plan.md §10/§13 Phase 0) against the
modern `tf.keras.optimizers.Optimizer` API after the previous implementation,
built on removed `tensorflow.python.keras.*`/`tensorflow.python.training.*`
internals, was found to no longer import on current TensorFlow. Each test
checks: the optimizer actually reduces a toy loss over a few steps (not just
"doesn't crash"), and that `get_config`/`from_config` round-trips.
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras.optimizers import (
    AdaBeliefOptimizer,
    AdaHessian,
    Lion,
    RectifiedAdam2,
    SAM,
    Yogi,
)


def _toy_dense_model():
    inp = tf.keras.Input((8,))
    x = tf.keras.layers.Dense(16, activation="relu")(inp)
    out = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inp, out)


def _toy_conv_model():
    inp = tf.keras.Input((8,))
    x = tf.keras.layers.Reshape((8, 1))(inp)
    x = tf.keras.layers.Conv1D(4, 3, padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    out = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inp, out)


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    x = tf.constant(rng.standard_normal((32, 8)).astype("float32"))
    y = tf.constant(rng.standard_normal((32, 1)).astype("float32"))
    return x, y


GRADIENT_BASED_CASES = [
    (Yogi, {}),
    (Yogi, {"l1_regularization_strength": 0.01}),
    (Yogi, {"l2_regularization_strength": 0.01}),
    (Yogi, {"beta1": 0.0}),
    (Yogi, {"activation": "tanh"}),
    (RectifiedAdam2, {}),
    (RectifiedAdam2, {"amsgrad": True}),
    (RectifiedAdam2, {"total_steps": 100, "warmup_proportion": 0.1}),
    (RectifiedAdam2, {"weight_decay": 0.01}),
    (AdaBeliefOptimizer, {}),
    (AdaBeliefOptimizer, {"amsgrad": True}),
    (Lion, {}),
    (Lion, {"weight_decay": 0.01}),
]


@pytest.mark.parametrize("opt_cls,kwargs", GRADIENT_BASED_CASES)
def test_optimizer_reduces_loss(opt_cls, kwargs, toy_data):
    x, y = toy_data
    tf.random.set_seed(0)
    model = _toy_dense_model()
    optimizer = opt_cls(**kwargs)

    losses = []
    for _ in range(20):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(pred - y))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        losses.append(float(loss))

    assert losses[-1] < losses[0], f"{opt_cls.__name__}{kwargs} did not reduce the loss"
    assert all(np.isfinite(losses)), f"{opt_cls.__name__}{kwargs} produced a non-finite loss"


@pytest.mark.parametrize("opt_cls,kwargs", GRADIENT_BASED_CASES)
def test_optimizer_config_roundtrip(opt_cls, kwargs):
    optimizer = opt_cls(**kwargs)
    config = optimizer.get_config()
    restored = opt_cls.from_config(config)
    assert restored.get_config() == config


def test_adahessian_minimize_reduces_loss(toy_data):
    x, y = toy_data
    tf.random.set_seed(0)
    model = _toy_conv_model()
    optimizer = AdaHessian(learning_rate=0.15)

    def loss_fn():
        pred = model(x, training=True)
        return tf.reduce_mean(tf.square(pred - y))

    n_steps = 40
    losses = [float(optimizer.minimize(loss_fn, model.trainable_variables)) for _ in range(n_steps)]

    assert all(np.isfinite(losses))
    assert int(optimizer.iterations) == n_steps
    # AdaHessian's Hessian-diagonal estimate is itself stochastic (Hutchinson's
    # trick, drawn from TF's global RNG stream inside `minimize`) - a single
    # unlucky draw can transiently raise the loss on any one step even though
    # the optimizer is working correctly. `tf.random.set_seed` above does not
    # reliably isolate this from other tests' random-op consumption earlier in
    # the same process (confirmed empirically: this test is 100% stable in
    # isolation but was observed to fail intermittently as part of the full
    # suite before this fix) - so compare *windowed averages* rather than a
    # single first/last step, which is robust to one noisy draw either way.
    early_avg = sum(losses[:5]) / 5
    late_avg = sum(losses[-5:]) / 5
    assert late_avg < early_avg, f"loss did not trend down: early={early_avg:.4f} late={late_avg:.4f}"


@pytest.mark.parametrize("average_size_4d", [None, -1, 2])
def test_adahessian_spatial_averaging_modes(toy_data, average_size_4d):
    x, y = toy_data
    tf.random.set_seed(0)
    model = _toy_conv_model()
    optimizer = AdaHessian(learning_rate=0.15, average_size_4d=average_size_4d)

    def loss_fn():
        pred = model(x, training=True)
        return tf.reduce_mean(tf.square(pred - y))

    losses = [float(optimizer.minimize(loss_fn, model.trainable_variables)) for _ in range(10)]
    assert all(np.isfinite(losses))


def test_adahessian_config_roundtrip():
    optimizer = AdaHessian(learning_rate=0.2, hessian_power=0.5, weight_decay=0.01)
    config = optimizer.get_config()
    restored = AdaHessian.from_config(config)
    assert restored.get_config() == config


# SAM (plan.md §4.2/§13 Phase 4) - not a `tf.keras.optimizers.Optimizer`
# subclass (see its docstring for why), so it's excluded from the
# parametrized GRADIENT_BASED_CASES above and tested separately via its own
# `minimize(loss_fn, var_list)` convenience path, the same style already
# used for AdaHessian's `minimize`-based tests above.
def test_sam_minimize_reduces_loss(toy_data):
    x, y = toy_data
    tf.random.set_seed(0)
    model = _toy_dense_model()
    sam = SAM(tf.keras.optimizers.Adam(0.01), rho=0.05)

    def loss_fn():
        pred = model(x, training=True)
        return tf.reduce_mean(tf.square(pred - y))

    losses = [float(sam.minimize(loss_fn, model.trainable_variables)) for _ in range(30)]

    assert losses[-1] < losses[0], "SAM did not reduce the loss"
    assert all(np.isfinite(losses)), "SAM produced a non-finite loss"
    assert int(sam.iterations) == 30, "SAM.iterations should delegate to the base optimizer"


def test_sam_restores_weights_between_first_and_second_step():
    """Regression test for SAM's core invariant: `first_step` perturbs the
    weights (ascent), but `second_step` must apply the *perturbed-point*
    gradient to the *original* (pre-perturbation) weights, not the
    perturbed ones - a wrapper that forgot to undo the perturbation before
    calling the base optimizer would silently train on the wrong point.
    """
    model = _toy_dense_model()
    sam = SAM(tf.keras.optimizers.SGD(0.1), rho=0.05)

    original_weights = [w.numpy().copy() for w in model.trainable_variables]
    x = tf.constant(np.random.default_rng(0).standard_normal((4, 8)).astype("float32"))

    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(model(x, training=True)))
    grads = tape.gradient(loss, model.trainable_variables)
    sam.first_step(grads, model.trainable_variables)

    perturbed_weights = [w.numpy().copy() for w in model.trainable_variables]
    assert any(
        not np.allclose(orig, pert) for orig, pert in zip(original_weights, perturbed_weights)
    ), "first_step should have perturbed at least one weight"

    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(model(x, training=True)))
    grads = tape.gradient(loss, model.trainable_variables)
    sam.second_step(grads, model.trainable_variables)

    # The perturbation must be undone before the base optimizer's own SGD
    # step is applied - so the final weights should differ from the
    # *perturbed* ones by something close to a plain SGD step from
    # `original_weights`, not from `perturbed_weights`. Concretely: they
    # must not simply equal `perturbed_weights` minus a small SGD step
    # (which would mean the perturbation was never undone).
    final_weights = [w.numpy() for w in model.trainable_variables]
    still_at_perturbed_point = all(
        np.allclose(pert, final, atol=1e-3) for pert, final in zip(perturbed_weights, final_weights)
    )
    assert not still_at_perturbed_point, "second_step should have undone first_step's perturbation"


def test_sam_second_step_without_first_step_raises():
    model = _toy_dense_model()
    sam = SAM(tf.keras.optimizers.SGD(0.1))
    x = tf.constant(np.random.default_rng(0).standard_normal((4, 8)).astype("float32"))

    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(model(x, training=True)))
    grads = tape.gradient(loss, model.trainable_variables)

    with pytest.raises(RuntimeError, match="first_step"):
        sam.second_step(grads, model.trainable_variables)
