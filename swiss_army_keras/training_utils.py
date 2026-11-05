'''Training helpers: `TrainingDriver` (frozen/unfrozen fine-tuning followed by
TFLite export), `ModelBuilder`, and layer freezing utilities.
'''
import os
import datetime
from matplotlib import pyplot as plt

import tensorflow as tf

import logging

from swiss_army_keras.utils import unfreeze_model
from swiss_army_keras.quantization_utils import Quantizer

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
    3. **Export**: the best weights checkpoint is reloaded and `Quantizer`
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
    '''
    def __init__(self, model, model_name, optimizer, loss, metrics, train_set, val_set, test_set, epochs, patience=20, unfreezed_epochs=-1, callbacks=(), quant_batches=1, plot_metrics=True, metrics_name=None, checkpoint_name=None, quantizer_name=None, export_profile='default', reparameterize=True, strip_qat=None, equalize=None, drop_output_activations=False):
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
        self.model.compile(loss=self.loss,
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

        logging.warning('Quantizing Model')

        q = Quantizer(self.test_set,
                      self.model,
                      self.quantizer_name,
                      batches=self.quant_batches,
                      weights_checkpoint_name=self.weights_checkpoint_name,
                      append_datetime=False,
                      profile=self.export_profile,
                      reparameterize=self.reparameterize,
                      strip_qat=self.strip_qat,
                      equalize=self.equalize,
                      drop_output_activations=self.drop_output_activations)

        q.quantize()
        self.quantizer = q
