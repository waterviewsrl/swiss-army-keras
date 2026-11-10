'''Knowledge distillation: train a small student to imitate a larger teacher.

`Distiller` wraps a student and a frozen teacher that take the same input.
Its training loss is `alpha * task_loss + (1 - alpha) * distillation_loss`.
Three distillation losses are available:

- `'logits'` (Hinton et al., 2015): Kullback-Leibler divergence between the
  softened class distributions, `softmax(logits / temperature)`. Works for
  classifiers and, per pixel, for segmentation.
- `'channel'` (Shu et al., 2021, "Channel-wise Knowledge Distillation for
  Dense Prediction"): each channel of a dense output is turned into a
  distribution over positions, and the divergence is taken per channel.
  Usually stronger than the per-pixel form for segmentation.
- `'mse'`: mean squared error between the two outputs, for regression heads
  and for feature maps.

Two uses matter for edge models: a large float teacher for a small student,
and the float model as teacher of its own quantization-aware copy.
'''

import tensorflow as tf


def _spatial_match(teacher, student):
    '''Resize a dense teacher output to the student's spatial size.'''
    if teacher.shape.rank == 4 and (teacher.shape[1] != student.shape[1] or teacher.shape[2] != student.shape[2]):
        teacher = tf.image.resize(teacher, tf.shape(student)[1:3], method='bilinear')
    return teacher


def logit_distillation_loss(teacher_logits, student_logits, temperature=4.0):
    '''`temperature**2 * KL(softmax(teacher / T) || softmax(student / T))`
    over the last axis, averaged over every other axis.
    '''
    teacher_logits = _spatial_match(tf.cast(teacher_logits, student_logits.dtype), student_logits)
    teacher_probabilities = tf.nn.softmax(teacher_logits / temperature, axis=-1)
    divergence = tf.reduce_sum(teacher_probabilities * (
        tf.nn.log_softmax(teacher_logits / temperature, axis=-1)
        - tf.nn.log_softmax(student_logits / temperature, axis=-1)), axis=-1)
    return temperature ** 2 * tf.reduce_mean(divergence)


def channel_distillation_loss(teacher_map, student_map, temperature=4.0):
    '''Channel-wise distillation for `(batch, H, W, C)` maps: the softmax is
    taken over the `H * W` positions of each channel.
    '''
    teacher_map = _spatial_match(tf.cast(teacher_map, student_map.dtype), student_map)
    shape = tf.shape(student_map)
    channels = student_map.shape[-1]
    teacher = tf.reshape(tf.transpose(teacher_map, (0, 3, 1, 2)), (shape[0], channels, -1)) / temperature
    student = tf.reshape(tf.transpose(student_map, (0, 3, 1, 2)), (shape[0], channels, -1)) / temperature
    divergence = tf.reduce_sum(tf.nn.softmax(teacher, axis=-1) * (
        tf.nn.log_softmax(teacher, axis=-1) - tf.nn.log_softmax(student, axis=-1)), axis=-1)
    return temperature ** 2 * tf.reduce_mean(divergence)


def mse_distillation_loss(teacher_output, student_output, temperature=None):
    '''Mean squared error between teacher and student outputs.'''
    teacher_output = _spatial_match(tf.cast(teacher_output, student_output.dtype), student_output)
    return tf.reduce_mean(tf.square(teacher_output - student_output))


_DISTILLATION_LOSSES = {
    'logits': logit_distillation_loss,
    'channel': channel_distillation_loss,
    'mse': mse_distillation_loss,
}


class Distiller(tf.keras.Model):
    '''Train `student` on its task and on the outputs of a frozen `teacher`.

    Example::

        distiller = Distiller(student, teacher, mode='channel', alpha=0.5)
        distiller.compile(optimizer=optimizer, loss=task_loss, metrics=[MeanIOU(n)])
        distiller.fit(train_set, validation_data=val_set, epochs=50)
        student.save_weights('student.weights.h5')     # the student is trained in place

    Both models receive the same input batch, so they must expect the same
    preprocessing. Both outputs must be **logits** for the `'logits'` and
    `'channel'` modes. A dense teacher output of another resolution is
    resized to the student's.

    Input
    ----------
        student: the model to train.
        teacher: the model to imitate; it is frozen and run in inference
                 mode.
        mode: `'logits'`, `'channel'` or `'mse'` (see the module
              docstring), or a callable
              `(teacher_output, student_output, temperature) -> scalar`.
        alpha: weight of the task loss; `1 - alpha` weighs the
               distillation loss. With `alpha=0` no labels are needed.
        temperature: softening temperature of the `'logits'` and
                     `'channel'` modes.
        student_output, teacher_output: index of the output to distill,
                     for models with several outputs.

    Validation and `predict` use the student alone. The logged metrics are
    `loss` (the weighted total), `task_loss` and `distillation_loss`, and
    the compiled metrics of the student.
    '''

    def __init__(self, student, teacher, mode='logits', alpha=0.5, temperature=4.0,
                 student_output=0, teacher_output=0, **kwargs):
        super().__init__(**kwargs)
        if not callable(mode) and mode not in _DISTILLATION_LOSSES:
            raise ValueError(f"Unknown mode '{mode}' - expected one of {sorted(_DISTILLATION_LOSSES)} or a callable.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('alpha must be between 0 and 1.')
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False
        self.distillation_fn = mode if callable(mode) else _DISTILLATION_LOSSES[mode]
        self.alpha = alpha
        self.temperature = temperature
        self.student_output = student_output
        self.teacher_output = teacher_output
        self._total_tracker = tf.keras.metrics.Mean(name='loss')
        self._task_tracker = tf.keras.metrics.Mean(name='task_loss')
        self._distillation_tracker = tf.keras.metrics.Mean(name='distillation_loss')
        if hasattr(student, 'preprocessing'):
            self.preprocessing = student.preprocessing

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    @staticmethod
    def _pick(outputs, index):
        return outputs[index] if isinstance(outputs, (list, tuple)) else outputs

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        teacher_outputs = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_outputs = self.student(x, training=True)
            distillation = self.distillation_fn(
                tf.stop_gradient(self._pick(teacher_outputs, self.teacher_output)),
                self._pick(student_outputs, self.student_output), self.temperature)
            if self.alpha > 0.0:
                task = self.compiled_loss(y, student_outputs, sample_weight, regularization_losses=self.student.losses)
            else:
                task = tf.zeros((), dtype=distillation.dtype)
            total = self.alpha * tf.cast(task, distillation.dtype) + (1.0 - self.alpha) * distillation
        variables = self.student.trainable_variables
        self.optimizer.apply_gradients(zip(tape.gradient(total, variables), variables))

        self._total_tracker.update_state(total)
        self._task_tracker.update_state(task)
        self._distillation_tracker.update_state(distillation)
        if y is not None:
            self.compiled_metrics.update_state(y, student_outputs, sample_weight)
        return self._logs()

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        student_outputs = self.student(x, training=False)
        task = self.compiled_loss(y, student_outputs, sample_weight, regularization_losses=self.student.losses)
        self._total_tracker.update_state(task)
        self._task_tracker.update_state(task)
        self.compiled_metrics.update_state(y, student_outputs, sample_weight)
        logs = self._logs()
        logs.pop('distillation_loss', None)
        return logs

    def _logs(self):
        logs = {m.name: m.result() for m in (self._total_tracker, self._task_tracker, self._distillation_tracker)}
        for metric in self.compiled_metrics.metrics:
            logs[metric.name] = metric.result()
        return logs

    @property
    def metrics(self):
        return [self._total_tracker, self._task_tracker, self._distillation_tracker] + list(self.compiled_metrics.metrics)
