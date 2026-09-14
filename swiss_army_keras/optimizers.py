from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, Dict, Union

import numpy as np
import tensorflow as tf


Number = Union[float, int, np.floating, np.integer]
FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]

# All four optimizers below target the modern `tf.keras.optimizers.Optimizer`
# base class (the `build()`/`update_step()` API introduced in TF 2.11, still
# the public optimizer API on the legacy-Keras-2 `tf.keras` this package
# forces via `TF_USE_LEGACY_KERAS`, see `swiss_army_keras/__init__.py`).
# Earlier versions of this file targeted a mix of removed private
# `tensorflow.python.keras.*`/`tensorflow.python.training.*` internals and
# would not import on any reasonably current TensorFlow.


def _solve(a, b, c):
    """Closed-form solution of the L1/L2 proximal step used by `Yogi`.

    Solves f(w) = argmin_w{1/2 * a * w^2 + b * w + c * |w|}.

    Input
    ----------
        a: A Tensor
        b: A Tensor
        c: A Tensor with one element.

    Output
    ----------
        w: A Tensor, the solution to the minimization above.
    """
    w = (c * tf.sign(b) - b) / a
    w = tf.cast(tf.abs(b) > c, dtype=b.dtype) * w
    return w


class Yogi(tf.keras.optimizers.Optimizer):
    """Optimizer that implements the Yogi algorithm.

    See Algorithm 2 of
    https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization.pdf.

    Input
    ----------
        learning_rate: A float or a schedule. The learning rate.
        beta1: exponential decay rate for the 1st moment estimate. Set to 0
               to disable momentum (plain adaptive-LR update on `v` only).
        beta2: exponential decay rate for the 2nd moment (`v`) estimate.
        epsilon: a constant trading off adaptivity and noise.
        l1_regularization_strength: L1 proximal regularization strength, >= 0.
        l2_regularization_strength: L2 proximal regularization strength, >= 0.
        initial_accumulator_value: starting value for the `v` accumulator.
        activation: 'sign' or 'tanh' — how `v` is updated from `g^2 - v`.
    """

    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable] = 0.01,
        beta1: FloatTensorLike = 0.9,
        beta2: FloatTensorLike = 0.999,
        epsilon: FloatTensorLike = 1e-3,
        l1_regularization_strength: FloatTensorLike = 0.0,
        l2_regularization_strength: FloatTensorLike = 0.0,
        initial_accumulator_value: FloatTensorLike = 1e-6,
        activation: str = "sign",
        name: str = "Yogi",
        **kwargs,
    ):
        if activation not in ("sign", "tanh"):
            raise ValueError("activation must be 'sign' or 'tanh'")
        super().__init__(name=name, **kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.l1_regularization_strength = l1_regularization_strength
        self.l2_regularization_strength = l2_regularization_strength
        self.initial_accumulator_value = initial_accumulator_value
        self.activation = activation

    def build(self, var_list):
        if getattr(self, "_built", False):
            return
        super().build(var_list)
        self._built = True
        self._v = []
        self._m = []
        for var in var_list:
            init_v = tf.fill(
                tf.shape(var), tf.cast(self.initial_accumulator_value, var.dtype)
            )
            self._v.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v", initial_value=init_v
                )
            )
            if self.beta1 > 0.0:
                self._m.append(
                    self.add_variable_from_reference(model_variable=var, variable_name="m")
                )

    def update_step(self, gradient, variable):
        var_dtype = variable.dtype
        lr = tf.cast(self.learning_rate, var_dtype)
        beta1 = tf.cast(self.beta1, var_dtype)
        beta2 = tf.cast(self.beta2, var_dtype)
        epsilon = tf.cast(self.epsilon, var_dtype)
        l1 = tf.cast(self.l1_regularization_strength, var_dtype)
        l2 = tf.cast(self.l2_regularization_strength, var_dtype)

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta1_power = tf.pow(beta1, local_step)
        beta2_power = tf.pow(beta2, local_step)
        step_size = lr * tf.sqrt(1 - beta2_power) / (1 - beta1_power)

        var_key = self._var_key(variable)
        v = self._v[self._index_dict[var_key]]

        dense_grad = (
            tf.convert_to_tensor(gradient)
            if isinstance(gradient, tf.IndexedSlices)
            else gradient
        )
        grad2 = tf.square(dense_grad)
        if self.activation == "sign":
            sign = tf.sign(grad2 - v)
        else:
            sign = tf.tanh(10 * (grad2 - v))
        v.assign_add((1 - beta2) * sign * grad2)
        per_coord_lr = step_size / (tf.sqrt(v) + epsilon)

        if self.beta1 > 0.0:
            m = self._m[self._index_dict[var_key]]
            m.assign(beta1 * m + (1 - beta1) * dense_grad)
            update = m
        else:
            update = dense_grad

        new_var = variable - per_coord_lr * update
        if self.l1_regularization_strength > 0:
            new_var = _solve(1 + l2 * per_coord_lr, -new_var, l1 * per_coord_lr)
        elif self.l2_regularization_strength > 0:
            new_var = new_var / (1 + l2 * per_coord_lr)
        variable.assign(new_var)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "l1_regularization_strength": self.l1_regularization_strength,
                "l2_regularization_strength": self.l2_regularization_strength,
                "initial_accumulator_value": self.initial_accumulator_value,
                "activation": self.activation,
            }
        )
        return config


class RectifiedAdam2(tf.keras.optimizers.Optimizer):
    """Rectified Adam (RAdam), with optional warmup.

    Liyuan Liu et al., "On The Variance Of The Adaptive Learning Rate And
    Beyond" (https://arxiv.org/pdf/1908.03265v1.pdf).

    Input
    ----------
        learning_rate: A float or a schedule.
        beta_1, beta_2: exponential decay rates for the 1st/2nd moments.
        epsilon: small constant for numerical stability.
        weight_decay: decoupled weight decay factor.
        amsgrad: whether to use the AMSGrad variant.
        sma_threshold: threshold on the simple moving average length below
                       which the update falls back to plain SGD-with-momentum.
        total_steps: enables linear warmup + linear decay when > 0.
        warmup_proportion: fraction of `total_steps` spent warming up.
        min_lr: floor for the learning rate once decay is enabled.
    """

    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable] = 0.001,
        beta_1: FloatTensorLike = 0.9,
        beta_2: FloatTensorLike = 0.999,
        epsilon: FloatTensorLike = 1e-7,
        weight_decay: FloatTensorLike = 0.0,
        amsgrad: bool = False,
        sma_threshold: FloatTensorLike = 5.0,
        total_steps: int = 0,
        warmup_proportion: FloatTensorLike = 0.1,
        min_lr: FloatTensorLike = 0.0,
        name: str = "RectifiedAdam",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.sma_threshold = sma_threshold
        self.total_steps = total_steps
        self.warmup_proportion = warmup_proportion
        self.min_lr = min_lr

    def build(self, var_list):
        if getattr(self, "_built", False):
            return
        super().build(var_list)
        self._built = True
        self._m = []
        self._v = []
        self._vhat = []
        for var in var_list:
            self._m.append(self.add_variable_from_reference(model_variable=var, variable_name="m"))
            self._v.append(self.add_variable_from_reference(model_variable=var, variable_name="v"))
            if self.amsgrad:
                self._vhat.append(
                    self.add_variable_from_reference(model_variable=var, variable_name="vhat")
                )

    def update_step(self, gradient, variable):
        var_dtype = variable.dtype
        lr = tf.cast(self.learning_rate, var_dtype)
        beta_1 = tf.cast(self.beta_1, var_dtype)
        beta_2 = tf.cast(self.beta_2, var_dtype)
        epsilon = tf.cast(self.epsilon, var_dtype)
        weight_decay = tf.cast(self.weight_decay, var_dtype)

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.pow(beta_1, local_step)
        beta_2_power = tf.pow(beta_2, local_step)

        if self.total_steps > 0:
            total_steps = tf.cast(self.total_steps, var_dtype)
            warmup_steps = total_steps * tf.cast(self.warmup_proportion, var_dtype)
            min_lr = tf.cast(self.min_lr, var_dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr) / decay_steps
            lr = tf.where(
                local_step <= warmup_steps,
                lr * (local_step / warmup_steps),
                lr + decay_rate * tf.minimum(local_step - warmup_steps, decay_steps),
            )

        dense_grad = (
            tf.convert_to_tensor(gradient)
            if isinstance(gradient, tf.IndexedSlices)
            else gradient
        )

        var_key = self._var_key(variable)
        idx = self._index_dict[var_key]
        m = self._m[idx]
        v = self._v[idx]

        m.assign(beta_1 * m + (1 - beta_1) * dense_grad)
        v.assign(beta_2 * v + (1 - beta_2) * tf.square(dense_grad))

        m_corr_t = m / (1 - beta_1_power)

        if self.amsgrad:
            vhat = self._vhat[idx]
            vhat.assign(tf.maximum(vhat, v))
            v_corr_t = tf.sqrt(vhat / (1 - beta_2_power))
        else:
            v_corr_t = tf.sqrt(v / (1 - beta_2_power))

        sma_inf = 2.0 / (1 - beta_2) - 1.0
        sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1 - beta_2_power)

        r_t = tf.sqrt(
            tf.maximum((sma_t - 4.0) * (sma_t - 2.0) * sma_inf, 0.0)
            / tf.maximum((sma_inf - 4.0) * (sma_inf - 2.0) * sma_t, epsilon)
        )

        var_t = tf.where(
            sma_t >= self.sma_threshold,
            r_t * m_corr_t / (v_corr_t + epsilon),
            m_corr_t,
        )

        if self.weight_decay > 0:
            var_t += weight_decay * variable

        variable.assign_sub(lr * var_t)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "weight_decay": self.weight_decay,
                "amsgrad": self.amsgrad,
                "sma_threshold": self.sma_threshold,
                "total_steps": self.total_steps,
                "warmup_proportion": self.warmup_proportion,
                "min_lr": self.min_lr,
            }
        )
        return config


class AdaBeliefOptimizer(tf.keras.optimizers.Optimizer):
    """Optimizer that implements the AdaBelief algorithm.

    Zhuang et al., "AdaBelief Optimizer: Adapting Stepsizes by the Belief in
    Observed Gradients" (https://arxiv.org/abs/2010.07468).

    AdaBelief tracks the variance of `gradient - momentum` (the "belief" in
    the observed gradient) rather than the variance of the gradient itself,
    which Adam/AdaBelief's sibling optimizers do.

    Input
    ----------
        learning_rate: A float or a schedule.
        beta1, beta2: exponential decay rates for the 1st/2nd moments.
        epsilon: small constant for numerical stability. The default 1e-8 is
                 not always a good choice; e.g. 1.0 or 0.1 works better for
                 some ImageNet training runs.
        amsgrad: whether to use the AMSGrad variant.
        name: optional name for the operations created when applying gradients.
    """

    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable] = 0.001,
        beta1: FloatTensorLike = 0.9,
        beta2: FloatTensorLike = 0.999,
        epsilon: FloatTensorLike = 1e-8,
        amsgrad: bool = False,
        name: str = "AdaBelief",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def build(self, var_list):
        if getattr(self, "_built", False):
            return
        super().build(var_list)
        self._built = True
        self._m = []
        self._v = []
        self._vhat = []
        for var in var_list:
            self._m.append(self.add_variable_from_reference(model_variable=var, variable_name="m"))
            self._v.append(self.add_variable_from_reference(model_variable=var, variable_name="v"))
            if self.amsgrad:
                self._vhat.append(
                    self.add_variable_from_reference(model_variable=var, variable_name="vhat")
                )

    def update_step(self, gradient, variable):
        var_dtype = variable.dtype
        lr = tf.cast(self.learning_rate, var_dtype)
        beta1 = tf.cast(self.beta1, var_dtype)
        beta2 = tf.cast(self.beta2, var_dtype)
        epsilon = tf.cast(self.epsilon, var_dtype)

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta1_power = tf.pow(beta1, local_step)
        beta2_power = tf.pow(beta2, local_step)
        step_size = lr * tf.sqrt(1 - beta2_power) / (1 - beta1_power)

        dense_grad = (
            tf.convert_to_tensor(gradient)
            if isinstance(gradient, tf.IndexedSlices)
            else gradient
        )

        var_key = self._var_key(variable)
        idx = self._index_dict[var_key]
        m = self._m[idx]
        v = self._v[idx]

        m.assign(beta1 * m + (1 - beta1) * dense_grad)
        v.assign(beta2 * v + (1 - beta2) * tf.square(dense_grad - m) + epsilon)

        if self.amsgrad:
            vhat = self._vhat[idx]
            vhat.assign(tf.maximum(vhat, v))
            v_sqrt = tf.sqrt(vhat)
        else:
            v_sqrt = tf.sqrt(v)

        variable.assign_sub(step_size * m / (v_sqrt + epsilon))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config


class AdaHessian(tf.keras.optimizers.Optimizer):
    """Optimizer that implements AdaHessian, a second-order (Hessian-diagonal
    aware) optimizer.

    Yao et al., "ADAHESSIAN: An Adaptive Second Order Optimizer for Machine
    Learning" (https://arxiv.org/abs/2006.00719).

    Unlike the other optimizers in this module, AdaHessian needs a
    Hutchinson-trace estimate of the Hessian diagonal, which requires a
    second backward pass through a nested `tf.GradientTape`. It therefore
    **cannot** be used as a drop-in `model.compile(optimizer=...)` /
    `model.fit()` optimizer — call `minimize(loss_fn, var_list)` from a
    custom training step instead (see example below).

    Example
    ----------
        opt = AdaHessian(learning_rate=0.15)
        for x, y in dataset:
            def loss_fn():
                return compiled_loss(y, model(x, training=True))
            opt.minimize(loss_fn, model.trainable_variables)

    Input
    ----------
        learning_rate: A float or a schedule.
        beta_1, beta_2: exponential decay rates for the 1st moment / Hessian
                        diagonal 2nd-moment estimates.
        epsilon: small constant for numerical stability.
        weight_decay: decoupled weight decay factor (AdamW-style).
        hessian_power: exponent applied to the Hessian diagonal estimate;
                       1.0 is standard, 0.5 sometimes performs better.
        average_size_1d/2d/3d/4d: spatial averaging of the Hessian diagonal
            estimate for parameters of that rank. `None` disables averaging
            for that rank; `-1` uses AdaHessian's suggested default (average
            over all-but-the-last-2-dims for conv kernels); a positive int
            sets a custom block size.
    """

    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable] = 0.15,
        beta_1: FloatTensorLike = 0.9,
        beta_2: FloatTensorLike = 0.999,
        epsilon: FloatTensorLike = 1e-4,
        weight_decay: FloatTensorLike = 0.0,
        hessian_power: FloatTensorLike = 1.0,
        average_size_1d=None,
        average_size_2d=None,
        average_size_3d=-1,
        average_size_4d=-1,
        name: str = "AdaHessian",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.hessian_power = hessian_power
        self.average_size_1d = average_size_1d
        self.average_size_2d = average_size_2d
        self.average_size_3d = average_size_3d
        self.average_size_4d = average_size_4d

    def build(self, var_list):
        if getattr(self, "_built", False):
            return
        super().build(var_list)
        self._built = True
        self._m = []
        self._v = []
        for var in var_list:
            self._m.append(self.add_variable_from_reference(model_variable=var, variable_name="m"))
            self._v.append(self.add_variable_from_reference(model_variable=var, variable_name="v"))

    def _hessian_diag_estimate(self, tape, loss, var_list):
        """Hutchinson's stochastic estimator of the Hessian diagonal.

        Requires a `persistent=True` outer `tf.GradientTape` that has NOT yet
        been used to compute `tape.gradient(loss, var_list)` — this method
        computes both the gradients and the Hessian-diagonal estimate for
        the caller, since both need the same inner tape.
        """
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(var_list)
            loss_value = loss()
        grads = inner_tape.gradient(loss_value, var_list)

        rademacher = [
            tf.sign(tf.random.uniform(tf.shape(g), -1.0, 1.0)) for g in grads
        ]
        grad_dot_v = tf.add_n(
            [tf.reduce_sum(g * v) for g, v in zip(grads, rademacher)]
        )
        hvp = tape.gradient(grad_dot_v, var_list)
        hessian_diag = [tf.abs(hv * v) for hv, v in zip(hvp, rademacher)]
        return loss_value, grads, hessian_diag

    def _spatial_average(self, hess, rank):
        resize = {
            1: self.average_size_1d,
            2: self.average_size_2d,
            3: self.average_size_3d,
            4: self.average_size_4d,
        }.get(rank)

        if resize is None:
            return hess
        if resize == -1:
            if rank in (1, 2):
                return hess
            axes = list(range(rank - 1))
            hess_avg = tf.reduce_mean(hess, axes, keepdims=True)
            return tf.broadcast_to(hess_avg, tf.shape(hess))
        if resize <= 0:
            raise ValueError("average_size must be None, -1, or a positive int")
        flat = tf.reshape(hess, [resize, -1])
        avg = tf.reduce_mean(flat, axis=0)
        return tf.reshape(tf.tile(avg, [resize]), tf.shape(hess))

    def _update_step_hessian(self, grad, hess, variable):
        var_dtype = variable.dtype
        lr = tf.cast(self.learning_rate, var_dtype)
        beta_1 = tf.cast(self.beta_1, var_dtype)
        beta_2 = tf.cast(self.beta_2, var_dtype)
        epsilon = tf.cast(self.epsilon, var_dtype)

        var_key = self._var_key(variable)
        idx = self._index_dict[var_key]
        m = self._m[idx]
        v = self._v[idx]

        m.assign_add((grad - m) * (1 - beta_1))

        hess_avg = self._spatial_average(hess, len(variable.shape))
        v.assign_add((tf.square(hess_avg) - v) * (1 - beta_2))

        local_step = tf.cast(self.iterations + 1, var_dtype)
        bias_correct_1 = 1 - tf.pow(beta_1, local_step)
        bias_correct_2 = 1 - tf.pow(beta_2, local_step)

        if self.weight_decay > 0:
            variable.assign_sub(lr * tf.cast(self.weight_decay, var_dtype) * variable)

        denom = tf.pow(tf.sqrt(v / bias_correct_2), self.hessian_power) + epsilon
        variable.assign_sub(lr * m / bias_correct_1 / denom)

    def minimize(self, loss, var_list):
        """Compute gradients + Hessian-diagonal estimate for `loss` and apply
        one AdaHessian update step to every variable in `var_list`.

        Input
        ----------
            loss: a zero-argument callable returning the scalar loss tensor.
            var_list: list of `tf.Variable` to update.

        Output
        ----------
            The scalar loss value computed for this step.
        """
        var_list = list(var_list)
        with tf.GradientTape(persistent=True) as tape:
            loss_value, grads, hessian_diag = self._hessian_diag_estimate(
                tape, loss, var_list
            )
        del tape

        self.build(var_list)
        for grad, hess, var in zip(grads, hessian_diag, var_list):
            if grad is None:
                continue
            self._update_step_hessian(grad, hess, var)
        self.iterations.assign_add(1)
        return loss_value

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "weight_decay": self.weight_decay,
                "hessian_power": self.hessian_power,
            }
        )
        return config


class Lion(tf.keras.optimizers.Optimizer):
    """Optimizer that implements Lion (EvoLved Sign Momentum).

    Chen, X. et al., 2023. "Symbolic Discovery of Optimization Algorithms"
    (https://arxiv.org/abs/2302.06675). Discovered via program search rather
    than hand-derived, Lion tracks only *one* momentum buffer per variable
    (half of what the Adam-family optimizers above track), and its update is
    the *sign* of an interpolation between the current gradient and that
    momentum buffer, rather than a magnitude-scaled step — memory-cheap and
    competitive with (or exceeding) AdamW in the paper's own benchmarks, a
    reasonable default for edge-model training loops where optimizer-state
    memory matters.

    Input
    ----------
        learning_rate: A float or a schedule. Lion's effective step size is
                       roughly an order of magnitude larger than AdamW's for
                       a comparable update magnitude (its update is a raw
                       sign, not scaled by gradient magnitude) — the paper
                       recommends starting from an AdamW learning rate
                       divided by ~10 (or scaling `weight_decay` up ~10x
                       instead) as a first guess, not reusing an AdamW
                       learning rate unchanged.
        beta_1: interpolation coefficient between the momentum buffer and
                the current gradient used to form the *sign* update (0.9 in
                the paper).
        beta_2: the momentum buffer's own decay rate (0.99 in the paper) —
                note this decays *slower* than `beta_1`'s interpolation,
                the reverse of Adam's `beta_1 < beta_2`... `beta_2`
                convention (Adam's `beta_2` is also the slower-decaying
                one, so the naming lines up — it's the numeric defaults
                that differ, 0.9/0.99 here vs. Adam's usual 0.9/0.999).
        weight_decay: decoupled weight decay (AdamW-style), applied
                      alongside the sign update, not folded into the
                      gradient itself.
        name: optional name for the operations created when applying gradients.
    """

    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable] = 1e-4,
        beta_1: FloatTensorLike = 0.9,
        beta_2: FloatTensorLike = 0.99,
        weight_decay: FloatTensorLike = 0.0,
        name: str = "Lion",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay

    def build(self, var_list):
        if getattr(self, "_built", False):
            return
        super().build(var_list)
        self._built = True
        self._momentums = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(model_variable=var, variable_name="m")
            )

    def update_step(self, gradient, variable):
        var_dtype = variable.dtype
        lr = tf.cast(self.learning_rate, var_dtype)
        beta_1 = tf.cast(self.beta_1, var_dtype)
        beta_2 = tf.cast(self.beta_2, var_dtype)
        weight_decay = tf.cast(self.weight_decay, var_dtype)

        gradient = (
            tf.convert_to_tensor(gradient) if isinstance(gradient, tf.IndexedSlices) else gradient
        )

        var_key = self._var_key(variable)
        idx = self._index_dict[var_key]
        m = self._momentums[idx]

        update = tf.sign(beta_1 * m + (1.0 - beta_1) * gradient)
        variable.assign_sub(lr * (update + weight_decay * variable))
        m.assign(beta_2 * m + (1.0 - beta_2) * gradient)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "weight_decay": self.weight_decay,
            }
        )
        return config


class SAM:
    """Sharpness-Aware Minimization (SAM), wrapping any base optimizer.

    Foret, P. et al., 2021. "Sharpness-Aware Minimization for Efficiently
    Improving Generalization" (https://arxiv.org/abs/2010.01412). Improves
    cross-domain generalization by seeking parameters whose entire
    neighborhood — not just the point itself — has low loss, a genuinely
    different mechanism from a learning-rate or regularization change, so it
    composes with any base optimizer (`Yogi`/`RectifiedAdam2`/
    `AdaBeliefOptimizer`/`AdaHessian`/`Lion` above, or a plain `tf.keras`
    optimizer) rather than replacing it — matching plan.md §4.2's own
    framing, "a thin gradient-ascent-then-descent wrapper around any base
    optimizer."

    Unlike every optimizer class in this module, SAM cannot be expressed as
    a single `apply_gradients(grads, vars)` call from one gradient
    computation — it needs the loss/gradient evaluated *twice* per step:
    once at the current weights (to find the ascent direction), once at a
    perturbed point (whose gradient is what actually gets descended on).
    This is therefore not a `tf.keras.optimizers.Optimizer` subclass (that
    API's contract is exactly one gradient in, one update out per call) —
    it exposes the two SAM steps directly (`first_step`/`second_step`) for a
    custom training loop, plus a `minimize(loss_fn, var_list)` convenience
    method that performs the full two-pass dance in one call for the common
    case.

    Input
    ----------
        base_optimizer: any `tf.keras.optimizers.Optimizer` instance — SAM's
                         actual descent step (`second_step`) delegates to
                         this optimizer's own `apply_gradients`.
        rho: the neighborhood size (L2-ball radius) SAM searches within for
             the worst-case ascent direction — 0.05, the paper's own default
             across most of its experiments.
        epsilon: numerical-stability constant added to the gradient-norm
                 denominator when computing the ascent step size.

    Usage
    ----------
        sam = SAM(tf.keras.optimizers.Adam(1e-3), rho=0.05)
        for x, y in dataset:
            def loss_fn():
                return loss_object(y, model(x, training=True))
            sam.minimize(loss_fn, model.trainable_variables)
    """

    def __init__(self, base_optimizer, rho: FloatTensorLike = 0.05, epsilon: FloatTensorLike = 1e-12):
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.epsilon = epsilon
        self._e_ws = None  # perturbations applied in `first_step`, undone in `second_step`

    def first_step(self, grads, var_list):
        """Ascent step: perturb every variable toward the worst-case point
        within the `rho`-ball of the current gradient direction. Call
        `second_step` with the gradients evaluated *at the perturbed point*
        to complete one SAM update.
        """
        grads = list(grads)
        var_list = list(var_list)

        grad_norm = tf.sqrt(
            tf.add_n([tf.reduce_sum(tf.square(g)) for g in grads if g is not None])
        )
        scale = self.rho / (grad_norm + self.epsilon)

        self._e_ws = []
        for grad, var in zip(grads, var_list):
            if grad is None:
                self._e_ws.append(None)
                continue
            e_w = grad * scale
            var.assign_add(e_w)
            self._e_ws.append(e_w)

    def second_step(self, grads, var_list):
        """Descent step: undo `first_step`'s perturbation, then apply the
        gradients evaluated at the perturbed point (`grads`) to the
        *original* weights via the wrapped `base_optimizer`.
        """
        if self._e_ws is None:
            raise RuntimeError("second_step called before first_step")

        var_list = list(var_list)
        for e_w, var in zip(self._e_ws, var_list):
            if e_w is not None:
                var.assign_sub(e_w)
        self._e_ws = None

        pairs = [(g, v) for g, v in zip(grads, var_list) if g is not None]
        self.base_optimizer.apply_gradients(pairs)

    def minimize(self, loss_fn, var_list):
        """Convenience wrapper performing the full two-pass SAM update:
        gradient at the current point -> `first_step` -> gradient at the
        perturbed point -> `second_step`.

        Input
        ----------
            loss_fn: a zero-argument callable returning the scalar loss.
            var_list: list of `tf.Variable` to update.

        Output
        ----------
            The scalar loss value computed at the *perturbed* point (the
            one whose gradient was actually descended on) — matching the
            reference SAM implementation's own convention for what "the
            step's loss" means.
        """
        var_list = list(var_list)

        with tf.GradientTape() as tape:
            loss_value = loss_fn()
        grads = tape.gradient(loss_value, var_list)
        self.first_step(grads, var_list)

        with tf.GradientTape() as tape:
            loss_value = loss_fn()
        grads = tape.gradient(loss_value, var_list)
        self.second_step(grads, var_list)

        return loss_value

    @property
    def iterations(self):
        return self.base_optimizer.iterations
