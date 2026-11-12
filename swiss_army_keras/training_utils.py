'''Training helpers: `TrainingDriver` (frozen/unfrozen fine-tuning followed by
TFLite export), `ModelBuilder`, and layer freezing utilities.
'''
import os
import datetime
from matplotlib import pyplot as plt

import itertools

import numpy as np
import tensorflow as tf

import logging

from swiss_army_keras.utils import unfreeze_model
from swiss_army_keras.quantization_utils import Quantizer
from swiss_army_keras import deploy_utils

from tensorflow.python.util.tf_export import keras_export


# NOT `from keras.models import Model` (the standalone Keras 3 package) -
# every model this library builds is a legacy-Keras-2 `tf.keras.models.Model`
# instance (`TF_USE_LEGACY_KERAS=1`, forced in `swiss_army_keras/__init__.py`),
# a *different* class hierarchy than standalone Keras 3's `Model`. The bare
# `keras` import made `isinstance(i, Model)` below always False for every
# model this library actually builds, silently breaking `freeze_layers`/
# `unfreeze_layers`'s recursion into nested backbone sub-models (in addition
# to the import itself being unresolvable without the standalone `keras`
# package installed at all) - see plan.md §10/§13 Phase 8.
from tensorflow.keras.models import Model


def freeze_layers(model, tlist=None):
    # `tlist=[]` as a default accumulated state across calls (the same list
    # object was reused), so a second `freeze_layers` call appended to the
    # first call's history and `unfreeze_layers` popped the wrong flags.
    '''Recursively set `trainable=False` on every layer of `model` (descending
    into nested sub-models such as a backbone), recording each layer's
    previous state.

    Input
    ----------
        model: a `tf.keras.Model`.
        tlist: internal accumulator for the recursion; leave as `None`.

    Output
    ----------
        (model, tlist): the same model, and the list of previous `trainable`
        flags, to be passed to `unfreeze_layers` to restore them.
    '''
    if tlist is None:
        tlist = []
    for i in model.layers:
        was = i.trainable
        i.trainable = False
        tlist.append(was)
        if isinstance(i, Model):
            freeze_layers(i, tlist)
    return model, tlist


def unfreeze_layers(model, tlist):
    '''Restore the `trainable` flags recorded by `freeze_layers`.

    Input
    ----------
        model: the model previously passed to `freeze_layers`.
        tlist: the list returned by `freeze_layers` (consumed in place).

    Output
    ----------
        model: the same model with its `trainable` flags restored.
    '''
    for i in model.layers:
        was = i.trainable
        i.trainable = tlist.pop()
        if isinstance(i, Model):
            unfreeze_layers(i, tlist)
    return model


@keras_export("keras.callbacks.WeightsModelCheckpoint")
class WeightsModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):

    '''A `ModelCheckpoint` variant that always saves weights only
    (`model.save_weights(filepath)`), on the same schedule as the parent
    class (`save_freq='epoch'` or every N batches). Kept for backward
    compatibility; `TrainingDriver` itself uses the stock
    `tf.keras.callbacks.ModelCheckpoint(save_weights_only=True)`.
    '''
    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            self._freeze_and_save_model(
                epoch=self._current_epoch, batch=batch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1

        if self.save_freq == "epoch":
            self._freeze_and_save_model(epoch=epoch, batch=None, logs=logs)

    def _freeze_and_save_model(self, epoch, batch, logs):
        self.model.save_weights(self.filepath)


class ModelBuilder:

    '''Wraps a model built on some input tensor into a new model with a fresh
    `Input` of a fixed `(width, height, channels)` shape, carrying over the
    `preprocessing` attribute set by `backbone_zoo`.

    Input
    ----------
        width, height, channels: shape of the new input, in that order
            (note: used as `(width, height, channels)`, so pass square
            sizes unless you have checked the layout).
    '''
    def __init__(self, width, height, channels):
        self.width = width
        self.height = height
        self.channels = channels

    def build(self, model):

        '''Input
        ----------
            model: a model returned by one of the `swiss_army_keras.models`
                   builders (must have a `preprocessing` attribute).

        Output
        ----------
            a `tf.keras.Model` with the new input, and `preprocessing` copied.
        '''
        input_tensor = tf.keras.layers.Input(
            shape=(self.width, self.height, self.channels))

        out = model(input_tensor)
        res = tf.keras.models.Model([input_tensor, ], out)
        res.preprocessing = model.preprocessing
        res.summary()

        return res


class TrainingDriver():

    '''Two-phase fine-tuning driver followed by TFLite export.

    `run()` performs, in order:

    1. **Frozen phase**: `compile` + `fit` for `epochs` epochs with the model
       exactly as given. Build the model with `freeze_backbone=True` so only
       the head/decoder trains here.
    2. **Unfrozen phase** (skipped when `unfreezed_epochs=0`): every layer is
       made trainable with `utils.unfreeze_model` (only BatchNormalization
       layers at the top level of the model stay frozen; those inside the
       nested backbone become trainable too), the optimizer's slots are
       rebuilt for the larger variable set, and the model is re-compiled and
       fitted for `unfreezed_epochs` more epochs with the same optimizer
       instance (so the same learning rate, unless a schedule or callback
       changes it).
    3. **Whole-model QAT phase** (only when `qat_epochs > 0`): the best
       weights are reloaded, the model is prepared with
       `deploy_utils.prepare_qat` for the weight granularity of
       `export_profile` (ranges calibrated on training batches), and
       fine-tuned for `qat_epochs` epochs with a small learning rate. The
       float model stays in `self.model`; the fine-tuned one is
       `self.qat_model`, and it is the one exported. Use it for a backbone
       that loses accuracy at export, above all with a per-tensor profile
       (see the deployment guide). An epoch costs about 1.7 times a float
       epoch.
    4. **Export**: the best weights checkpoint is reloaded and `Quantizer`
       writes `<quantizer_name>_quant_ui8.tflite` (full integer, uint8
       input/output) and `<quantizer_name>_quant_f16.tflite` (float16).

    Callbacks installed automatically: TensorBoard
    (`tblogs/logs_<model_name>_<UTC timestamp>`), best-only weights
    checkpoint, best-only full-model `.h5` checkpoint, and
    `EarlyStopping(monitor='val_loss', patience=patience,
    restore_best_weights=True)`. User `callbacks` are appended after these.

    All files are written relative to the current working directory.

    Input
    ----------
        model: the model to train (from `swiss_army_keras.models`).
        model_name: prefix for every output file and the TensorBoard run.
        optimizer: a `tf.keras.optimizers.Optimizer` instance (legacy
                   Keras 2 API), e.g. one from `swiss_army_keras.optimizers`.
        loss: loss function (see `swiss_army_keras.losses`; wrap with
              `losses.loss_adapter(..., from_logits=True)` for the decoders,
              which output raw logits).
        metrics: list of Keras metrics.
        train_set, val_set, test_set: `tf.data.Dataset` objects yielding
              `(images, labels)` batches, e.g. from
              `SegmentationAlbumentationsDataLoader.build_datasets()`.
              `test_set` is only used as the calibration set for export.
        epochs: number of epochs for the frozen phase.
        patience: `EarlyStopping` patience, in epochs.
        unfreezed_epochs: epochs for the unfrozen phase; `-1` (default) means
              "same as `epochs`", `0` skips the phase.
        callbacks: extra Keras callbacks.
        quant_batches: number of `test_set` batches used for calibration.
        plot_metrics: if True, save loss/accuracy/IoU curves as
              `<metrics_name>_<loss|acc|iou>_<frozen|unfrozen>.pdf`.
        metrics_name: prefix for those PDFs (default: model name + timestamp).
        checkpoint_name: prefix for the `.h5` and `.weights` checkpoints
              (default: model name + timestamp).
        quantizer_name: prefix for the `.tflite` files (default: model name
              + timestamp).
        export_profile, reparameterize, strip_qat, equalize,
        drop_output_activations: passed to
              `Quantizer` (`profile=`, `reparameterize=`, `strip_qat=`,
              `equalize=`, `drop_output_activations=`): the deployment
              target the TFLite files are exported and checked for, e.g.
              `export_profile='artpec8'`. After `run()`, the `Quantizer`
              is available as `self.quantizer` (with its `lint_report`).
        loss_weights: passed to `model.compile`. With a model that has
              several outputs, give `loss`, `metrics` and `loss_weights`
              as lists (one entry per output) or as dictionaries keyed by
              output name. The data loaders yield one target per image,
              which Keras hands to every output. Example for
              `pidnet(augment=True)`, whose outputs are the auxiliary
              logits, the main logits and the boundary logit::

                  loss=[ce, ce, pidnet_boundary_loss], loss_weights=[0.4, 1.0, 20.0],
                  metrics=[[], [MeanIOU(n_classes)], []], export_outputs=1

        export_outputs: which outputs of the model go into the exported
              files (`Quantizer(outputs=...)`); None exports all of them.
        export_options: a dict of further `Quantizer` arguments, e.g.
              `dict(bias_correction=False, strict=True)`.
        qat_epochs: epochs of whole-model quantization-aware fine-tuning
              before the export; 0 (default) skips the phase.
        qat_learning_rate: learning rate of that phase (Adam). A tenth to
              a hundredth of the training rate is the usual range.
        qat_optimizer: an optimizer instance to use instead of
              `Adam(qat_learning_rate)`.
        qat_calibration_batches: number of `train_set` batches used to set
              the activation ranges before the fine-tuning starts.
        data_loader: the data loader the three sets come from. When given,
              `run()` writes which images are in each set to
              `<checkpoint_name>_split.json` (`self.split_manifest_name`),
              so that the trained model can later be evaluated on images
              it did not see (`split_manifest=` of the loaders).
    '''
    def __init__(self, model, model_name, optimizer, loss, metrics, train_set, val_set, test_set, epochs, patience=20, unfreezed_epochs=-1, callbacks=(), quant_batches=1, plot_metrics=True, metrics_name=None, checkpoint_name=None, quantizer_name=None, export_profile='default', reparameterize=True, strip_qat=None, equalize=None, drop_output_activations=False, loss_weights=None, export_outputs=None, export_options=None, qat_epochs=0, qat_learning_rate=1e-5, qat_optimizer=None, qat_calibration_batches=4, data_loader=None):
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.epochs = epochs
        self.unfreezed_epochs = epochs if unfreezed_epochs == -1 else unfreezed_epochs
        self.quant_batches = quant_batches
        self.export_profile = export_profile
        self.reparameterize = reparameterize
        self.strip_qat = strip_qat
        self.equalize = equalize
        self.drop_output_activations = drop_output_activations
        self.loss_weights = loss_weights
        self.export_outputs = export_outputs
        self.export_options = dict(export_options or {})
        self.qat_epochs = qat_epochs
        self.qat_learning_rate = qat_learning_rate
        self.qat_optimizer = qat_optimizer
        self.qat_calibration_batches = qat_calibration_batches
        self.data_loader = data_loader
        self.qat_model = None
        self.qat_history = None
        self.user_callbacks = list(callbacks)
        self.plot_metrics = plot_metrics

        self.patience = patience

        self.callbacks = []

        self.datestr = datetime.datetime.now(datetime.timezone.utc).strftime("_%Y-%m-%d_%H-%M-%S")

        self.logsname = 'tblogs/logs_'+model_name+self.datestr
        self.checkpoint_name = checkpoint_name + '.h5' if checkpoint_name is not None else model_name + \
            self.datestr + '.h5'
        self.metrics_name = metrics_name if metrics_name is not None else model_name + \
            self.datestr
        self.weights_checkpoint_name = checkpoint_name + '.weights' if checkpoint_name is not None else model_name + \
            self.datestr + '.weights'
        self.quantizer_name = quantizer_name if quantizer_name is not None else model_name + self.datestr
        prefix = checkpoint_name if checkpoint_name is not None else model_name + self.datestr
        self.qat_weights_checkpoint_name = prefix + '_qat.weights'
        self.split_manifest_name = prefix + '_split.json'

        self.callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=self.logsname)
        )
        self.callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(self.weights_checkpoint_name,
                                               save_weights_only=True,
                                               save_best_only=True)
        )
        self.callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(self.checkpoint_name,
                                               save_best_only=True)
        )
        self.callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                mode="min",
                restore_best_weights=True,
            )
        )

        for c in callbacks:
            self.callbacks.append(c)

    def run(self):

        '''Run the frozen phase, the optional unfrozen phase and the TFLite export
        (see the class docstring). Blocking; returns nothing. The trained model
        is available afterwards as `self.model`.
        '''
        if self.data_loader is not None:
            self.data_loader.save_split(self.split_manifest_name)
            logging.warning('Split written to %s', self.split_manifest_name)

        self.model.compile(loss=self.loss,
                           loss_weights=self.loss_weights,
                           optimizer=self.optimizer,
                           metrics=self.metrics)

        self.model.summary()

        model_history = self.model.fit(self.train_set,
                                       epochs=self.epochs,
                                       validation_data=self.val_set,
                                       callbacks=self.callbacks,
                                       )

        if self.plot_metrics:

            try:

                key_names = ['loss', 'acc', 'iou']
                for k in key_names:
                    legend = []
                    for metric in model_history.history:
                        if k.upper() in str(metric).upper():
                            plt.plot(model_history.history[metric])
                            legend.append(metric)

                    plt.title(k)
                    plt.xlabel('epochs')
                    plt.legend(legend, loc='upper left')
                    filename = self.metrics_name + '_'+k+'_frozen.pdf'
                    # `os.makedirs('')` raises, so with the default
                    # (directory-less) `metrics_name` the plots were never
                    # saved - the error was only logged below.
                    if os.path.dirname(filename):
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                    plt.savefig(filename)
                    plt.clf()

            except Exception as e:
                logging.error(e)

        if self.unfreezed_epochs > 0:

            logging.warning('Unfreezing Model')

            self.model = unfreeze_model(self.model)

            # The optimizer was already built (its internal per-variable slots
            # created) against the frozen phase's smaller trainable-variable
            # list, by the first `fit()` call above. Unfreezing adds many more
            # trainable variables (the whole backbone); re-`compile`-ing with
            # the same already-built optimizer instance does NOT rebuild it,
            # so the second `fit()` crashes the first time it touches one of
            # the newly-unfrozen variables: `KeyError: The optimizer cannot
            # recognize variable ...`. Resetting `_built` and rebuilding
            # against the model's current (full) trainable variables is the
            # fix Keras' own error message points to (plan.md sec.10/sec.13
            # Phase 8).
            self.optimizer._built = False
            self.optimizer.build(self.model.trainable_variables)

            self.model.compile(loss=self.loss,
                               loss_weights=self.loss_weights,
                               optimizer=self.optimizer,
                               metrics=self.metrics)

            self.model.summary()

            model_history = self.model.fit(self.train_set,
                                           epochs=self.unfreezed_epochs,
                                           validation_data=self.val_set,
                                           callbacks=self.callbacks,
                                           )
            if self.plot_metrics:

                try:

                    key_names = ['loss', 'acc', 'iou']
                    for k in key_names:
                        legend = []
                        for metric in model_history.history:
                            if k.upper() in str(metric).upper():
                                plt.plot(model_history.history[metric])
                                legend.append(metric)

                        plt.title(k)
                        plt.xlabel('epochs')
                        plt.legend(legend, loc='upper left')
                        filename = self.metrics_name + '_'+k+'_unfrozen.pdf'
                        if os.path.dirname(filename):
                            os.makedirs(os.path.dirname(filename), exist_ok=True)
                        plt.savefig(filename)
                        plt.clf()

                except Exception as e:
                    logging.error(e)

        export_model, export_checkpoint = self.model, self.weights_checkpoint_name
        if self.qat_epochs > 0:
            self._run_qat_phase()
            export_model, export_checkpoint = self.qat_model, self.qat_weights_checkpoint_name

        logging.warning('Quantizing Model')

        q = Quantizer(self.test_set,
                      export_model,
                      self.quantizer_name,
                      batches=self.quant_batches,
                      weights_checkpoint_name=export_checkpoint,
                      append_datetime=False,
                      profile=self.export_profile,
                      reparameterize=self.reparameterize,
                      strip_qat=self.strip_qat,
                      equalize=self.equalize,
                      drop_output_activations=self.drop_output_activations,
                      outputs=self.export_outputs,
                      **self.export_options)

        q.quantize()
        self.quantizer = q

    def _run_qat_phase(self):
        '''Prepare the trained model for whole-model QAT and fine-tune it.'''
        logging.warning('Whole-model QAT fine-tuning')
        self.model.load_weights(self.weights_checkpoint_name)

        calibration = []
        for batch in itertools.islice(iter(self.train_set), self.qat_calibration_batches):
            calibration.append(np.asarray(batch[0] if isinstance(batch, (tuple, list)) else batch, dtype=np.float32))
        calibration = np.concatenate(calibration)

        profile = deploy_utils.get_profile(self.export_profile)
        prepared = deploy_utils.prepare_qat(self.model, calibration, per_channel=profile.per_channel)
        if prepared.unwrapped_layers:
            logging.warning('Layers left out of QAT: %s', ', '.join(prepared.unwrapped_layers))

        # losses and metrics given by output name follow the outputs,
        # whose layers were renamed by the preparation
        renamed = dict(zip(self.model.output_names, prepared.output_names))

        def by_output(spec):
            return {renamed.get(key, key): value for key, value in spec.items()} if isinstance(spec, dict) else spec

        optimizer = self.qat_optimizer if self.qat_optimizer is not None else tf.keras.optimizers.Adam(
            self.qat_learning_rate)
        prepared.compile(loss=by_output(self.loss), loss_weights=by_output(self.loss_weights),
                         optimizer=optimizer, metrics=by_output(self.metrics))

        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=self.logsname + '_qat'),
            tf.keras.callbacks.ModelCheckpoint(self.qat_weights_checkpoint_name, save_weights_only=True,
                                               save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, mode='min',
                                             restore_best_weights=True),
        ] + self.user_callbacks
        self.qat_history = prepared.fit(self.train_set, epochs=self.qat_epochs, validation_data=self.val_set,
                                        callbacks=callbacks)
        self.qat_model = prepared


# ========================= #
# Training strategies


class WeightAveraging(tf.keras.callbacks.Callback):
    '''Keep an average of the model weights during training and use it for
    validation, for checkpoints and for the final model.

    Averaged weights are more stable than the last iterate on small or
    noisy datasets, and flatter solutions tolerate quantization better.

    * `mode='ema'`: exponential moving average, updated every
      `update_every` batches: `average = decay * average + (1 - decay) *
      weights`. Batch-norm statistics are averaged with everything else, so
      nothing needs to be recomputed.
    * `mode='swa'`: stochastic weight averaging (Izmailov et al., 2018): the
      plain mean of the weights at the end of each epoch from `start_epoch`
      on. Batch-norm statistics of an average of models are not the average
      of their statistics: call `recalibrate_batch_norm` after training.

    How it interacts with `fit`: the averaged weights are swapped in when
    validation starts and swapped out again at the first batch of the next
    epoch. Callbacks that run at the end of an epoch (model checkpoints,
    early stopping on a validation metric) therefore see the averaged
    model. When training ends the averaged weights stay in the model, so an
    export that follows uses them.

    Input
    ----------
        mode: `'ema'` or `'swa'`.
        decay: EMA decay (0.999 suits a few thousand steps; use a smaller
               value for short runs).
        update_every: EMA update period, in batches.
        start_epoch: first epoch (0-based) whose weights enter the average.
        warmup: with EMA, use `min(decay, (1 + n) / (10 + n))` for the n-th
                update, so that the average is not dominated by the
                initial weights.
    '''

    def __init__(self, mode='ema', decay=0.999, update_every=1, start_epoch=0, warmup=True):
        super().__init__()
        if mode not in ('ema', 'swa'):
            raise ValueError(f"Unknown mode '{mode}' - expected 'ema' or 'swa'.")
        self.mode = mode
        self.decay = decay
        self.update_every = update_every
        self.start_epoch = start_epoch
        self.warmup = warmup
        self._average = None
        self._backup = None
        self._updates = 0
        self._epoch = 0

    # -- bookkeeping
    def _variables(self):
        return [v for v in self.model.weights if v.dtype.is_floating]

    def _reset(self):
        self._average = [tf.Variable(v, trainable=False) for v in self._variables()]
        self._updates = 0

    def _accumulate(self):
        variables = self._variables()
        if self._average is None or len(self._average) != len(variables):
            self._reset()
            return
        self._updates += 1
        if self.mode == 'ema':
            decay = min(self.decay, (1.0 + self._updates) / (10.0 + self._updates)) if self.warmup else self.decay
        else:
            decay = self._updates / (self._updates + 1.0)        # running mean
        for average, variable in zip(self._average, variables):
            average.assign(decay * average + (1.0 - decay) * variable)

    def _swap_in(self):
        if self._average is None or self._backup is not None:
            return
        variables = self._variables()
        self._backup = [v.numpy() for v in variables]
        for variable, average in zip(variables, self._average):
            variable.assign(average)

    def _swap_out(self):
        if self._backup is None:
            return
        for variable, value in zip(self._variables(), self._backup):
            variable.assign(value)
        self._backup = None

    # -- callback hooks
    def on_train_begin(self, logs=None):
        self._backup = None
        self._epoch_done = False
        if self._average is not None and len(self._average) != len(self._variables()):
            self._average = None

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch
        self._epoch_done = False

    def on_train_batch_begin(self, batch, logs=None):
        self._swap_out()

    def on_train_batch_end(self, batch, logs=None):
        if self.mode == 'ema' and self._epoch >= self.start_epoch and (batch + 1) % self.update_every == 0:
            self._accumulate()

    def _close_epoch(self):
        # runs once per epoch, on raw weights: before validation if there is one, else at the end
        if not self._epoch_done and self.mode == 'swa' and self._epoch >= self.start_epoch \
                and self._backup is None:
            self._accumulate()
        self._epoch_done = True
        self._swap_in()

    def on_test_begin(self, logs=None):
        self._close_epoch()

    def on_epoch_end(self, epoch, logs=None):
        self._close_epoch()

    def on_train_end(self, logs=None):
        self._swap_in()
        self._backup = None                 # the averaged weights are final


def recalibrate_batch_norm(model, dataset, steps=50):
    '''Recompute the batch-norm statistics of `model` on `dataset`.

    Needed after `WeightAveraging(mode='swa')`, and useful after any
    operation that changes weights without training (weight averaging,
    pruning). The moving statistics are reset and then estimated as the
    plain average over `steps` batches; no weight is trained.

    Input
    ----------
        model: a `tf.keras.Model`.
        dataset: yields batches of inputs, or `(inputs, ...)` tuples.
        steps: number of batches to use.

    Output
    ----------
        The number of batch-norm layers that were recalibrated.
    '''
    def batch_norms(m):
        found = []
        for layer in m.layers:
            inner = getattr(layer, 'layer', layer)
            if isinstance(inner, tf.keras.layers.BatchNormalization):
                found.append(inner)
            elif isinstance(layer, tf.keras.Model):
                found.extend(batch_norms(layer))
        return found

    layers = batch_norms(model)
    if not layers:
        return 0
    saved = [(layer.momentum, layer.trainable) for layer in layers]
    for layer in layers:
        layer.moving_mean.assign(tf.zeros_like(layer.moving_mean))
        layer.moving_variance.assign(tf.ones_like(layer.moving_variance))
    try:
        for step, batch in enumerate(dataset):
            if step >= steps:
                break
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            for layer in layers:
                # momentum k/(k+1) turns the moving average into a running mean
                layer.momentum = step / (step + 1.0)
            model(inputs, training=True)
    finally:
        for layer, (momentum, trainable) in zip(layers, saved):
            layer.momentum = momentum
            layer.trainable = trainable
    return len(layers)


class SharpnessAwareModel(tf.keras.Model):
    '''A model trained with sharpness-aware minimization inside `fit`.

    Each step computes the gradient, moves the weights to the worst point
    of a small neighbourhood, computes the gradient there, moves back and
    applies that second gradient with the compiled optimizer (Foret et al.,
    2021). `adaptive=True` gives ASAM (Kwon et al., 2021), whose
    neighbourhood scales with each weight. A step costs two forward and
    backward passes.

    Build it with `sharpness_aware(model)`. `optimizers.SAM` remains
    available for custom training loops.
    '''

    def __init__(self, *args, rho=0.05, adaptive=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = rho
        self.adaptive = adaptive

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        variables = self.trainable_variables

        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, y, self(x, training=True), sample_weight)
        gradients = tape.gradient(loss, variables)
        if self.adaptive:
            gradients = [None if g is None else tf.square(v) * g for g, v in zip(gradients, variables)]
            terms = [tf.reduce_sum(tf.square(g / (tf.abs(v) + 1e-12))) for g, v in zip(gradients, variables)
                     if g is not None]
        else:
            terms = [tf.reduce_sum(tf.square(g)) for g in gradients if g is not None]
        scale = self.rho / (tf.sqrt(tf.add_n(terms)) + 1e-12)
        perturbations = [None if g is None else g * scale for g in gradients]
        for variable, perturbation in zip(variables, perturbations):
            if perturbation is not None:
                variable.assign_add(perturbation)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        gradients = tape.gradient(loss, variables)
        for variable, perturbation in zip(variables, perturbations):
            if perturbation is not None:
                variable.assign_sub(perturbation)
        self.optimizer.apply_gradients((g, v) for g, v in zip(gradients, variables) if g is not None)
        return self.compute_metrics(x, y, y_pred, sample_weight)


def sharpness_aware(model, rho=0.05, adaptive=False):
    '''Wrap a functional model so that `fit` trains it with SAM or ASAM;
    see `SharpnessAwareModel`.

    Input
    ----------
        model: a functional `tf.keras.Model`.
        rho: radius of the neighbourhood (0.05 for SAM; ASAM uses 0.5 to 2).
        adaptive: use ASAM.

    Output
    ----------
        A `SharpnessAwareModel` sharing the layers of `model`: compile and
        fit it, then export `model`.
    '''
    wrapped = SharpnessAwareModel(inputs=model.inputs, outputs=model.outputs, name=model.name + '_sam',
                                  rho=rho, adaptive=adaptive)
    for attribute in ('preprocessing', 'head_names'):
        if hasattr(model, attribute):
            setattr(wrapped, attribute, getattr(model, attribute))
    return wrapped
