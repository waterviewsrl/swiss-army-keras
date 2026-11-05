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
    for i in model.layers:
        was = i.trainable
        i.trainable = tlist.pop()
        if isinstance(i, Model):
            unfreeze_layers(i, tlist)
    return model


@keras_export("keras.callbacks.WeightsModelCheckpoint")
class WeightsModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):

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

    def __init__(self, width, height, channels):
        self.width = width
        self.height = height
        self.channels = channels

    def build(self, model):

        input_tensor = tf.keras.layers.Input(
            shape=(self.width, self.height, self.channels))

        out = model(input_tensor)
        res = tf.keras.models.Model([input_tensor, ], out)
        res.preprocessing = model.preprocessing
        res.summary()

        return res


class TrainingDriver():

    def __init__(self, model, model_name, optimizer, loss, metrics, train_set, val_set, test_set, epochs, patience=20, unfreezed_epochs=-1, callbacks=(), quant_batches=1, plot_metrics=True, metrics_name=None, checkpoint_name=None, quantizer_name=None):
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
                    os.makedirs(os.path.dirname(
                        filename), exist_ok=True)
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
                        os.makedirs(os.path.dirname(
                            filename), exist_ok=True)
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
                      append_datetime=False)

        q.quantize()
