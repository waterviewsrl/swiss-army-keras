'''TFLite export: `Quantizer` prepares a trained model for a deployment profile
(reparameterization, QAT stripping), converts it to full-integer and float16
TFLite files, checks the result, and can visualize its predictions. The
building blocks live in `deploy_utils`.
'''
import tensorflow as tf
import numpy as np

from datetime import datetime

import logging

import matplotlib.pyplot as plt

from swiss_army_keras import deploy_utils


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def standard_totflite(val):
    '''Convert an image tensor normalized for a float TFLite model to a float32 numpy array.
    '''
    return val.numpy().astype(np.float32)


def symmetric_totflite(val):
    '''Same as `standard_totflite`; kept as a separate entry of `totflite_dict` for symmetric (`[-1, 1]`) normalization.
    '''
    return val.numpy().astype(np.float32)


def ui8_totflite(val):
    '''Convert an image tensor normalized to `[0, 1]` to the uint8 numpy array a full-integer TFLite model expects.
    '''
    return (val*255).numpy().astype(np.uint8)


totflite_dict = {}
totflite_dict[-1] = ui8_totflite
totflite_dict[0] = standard_totflite
totflite_dict[1] = symmetric_totflite


class Quantizer():
    '''
    TFLite export of a trained model for a deployment target.

    `quantize()` writes two files (prefix `name`, relative to the current
    working directory):

    * `<name>_quant_ui8.tflite` (`_quant_i8` for a profile with int8
      input/output): full-integer model. Only `TFLITE_BUILTINS_INT8` ops are
      allowed (conversion fails otherwise) and `batches` batches of `dataset`
      are used as the representative (calibration) dataset.
    * `<name>_quant_f16.tflite`: float16 weights, float32 input/output.

    Before converting, the model is prepared for deployment:

    1. **Reparameterization** (`reparameterize=True`): the training-time
       branches of a MobileOne, RepViT or FastViT backbone are fused into
       plain convolutions (`deploy_utils.reparameterize_model`). The result
       computes the same function with several times fewer operators.
    2. **QAT stripping** (`strip_qat`): the quantization-aware-training
       wrappers of the head are removed and the whole graph is quantized
       post-training from the calibration data
       (`deploy_utils.strip_qat`). This is needed for integer-only
       accelerators: with the wrappers left in, the converter can leave an
       operator in floating point where the head's learned ranges and the
       backbone's calibrated ranges disagree, and convolution weights of
       the head stay quantized per channel. By default (`strip_qat=None`)
       the wrappers are stripped for profiles marked `integer_only`
       (`artpec7`, `artpec8`, `artpec9`) and kept otherwise.

    3. **Cross-layer equalization** (`equalize`): for per-tensor profiles,
       consecutive convolutions are rescaled so each weight tensor has
       similar ranges in all its channels
       (`deploy_utils.equalize_cross_layer`). It does not change what the
       model computes; the original weights are restored after conversion.
       By default (`equalize=None`) it is applied when the wrappers were
       stripped and the profile quantizes weights per tensor (`artpec8`).

    After converting, the flatbuffer is checked against the profile
    (`deploy_utils.lint_tflite`); the report is logged and kept in
    `self.lint_report`.

    Input
    ----------
        dataset: a `tf.data.Dataset` yielding `(images, labels)` batches with
                 images already normalized the way the model expects, e.g.
                 the test set returned by a data loader's `build_datasets()`.
        model: a `tf.keras.Model`, or the path of a saved model loadable
               with `tf.keras.models.load_model`.
        name: prefix of the output files.
        append_datetime: append a local timestamp to `name`.
        batches: number of dataset batches used for calibration.
        weights_checkpoint_name: optional weights checkpoint to load into
               `model` before converting (ignored when `model` is a path).
        profile: name of a deployment profile (`deploy_utils.PROFILES`:
               'default', 'litert-cpu', 'artpec7', 'artpec8', 'artpec9') or
               a `deploy_utils.DeploymentProfile`. It sets the weight
               quantization granularity (per channel or per tensor), the
               input/output type, and what the exported file is checked
               against.
        reparameterize: fuse reparameterizable backbones before converting.
        strip_qat: True / False to force, None for the profile-dependent
               default described above.
        strict: raise `RuntimeError` if the lint report has errors or
               operators outside the profile's operator list, instead of
               only logging a warning.
        equalize: True / False to force cross-layer equalization, None for
               the profile-dependent default described above.
    '''
    def __init__(self, dataset, model, name, append_datetime=True, batches=1, weights_checkpoint_name=None,
                 profile='default', reparameterize=True, strip_qat=None, strict=False, equalize=None):
        self.dataset = dataset
        self.model = model
        if append_datetime:
            self.name = f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_'
        else:
            self.name = name + '_'
        self.saved_model_dirname = ''
        self.batches = batches

        self.tflite_ui8_model = None
        self.tflite_f16_model = None
        self.normalization = 0
        self.weights_checkpoint_name = weights_checkpoint_name

        self.profile = deploy_utils.get_profile(profile)
        self.reparameterize = reparameterize
        self.strip_qat = self.profile.integer_only if strip_qat is None else strip_qat
        self.strict = strict
        # Cross-layer equalization only makes sense on a plain (stripped)
        # model, and measurably helps per-tensor exports only.
        self.equalize = (self.strip_qat and not self.profile.per_channel) if equalize is None else equalize
        self.equalization = None
        self.export_model = None
        self.lint_report = None
        self.tflite_int_path = None
        self.tflite_f16_path = None

    def _prepare_export_model(self):
        '''Load weights, then fuse and strip the model as configured.'''
        if isinstance(self.model, str):
            model = tf.keras.models.load_model(self.model, compile=False)
            self.saved_model_dirname = self.model
        else:
            model = self.model
            if self.weights_checkpoint_name is not None:
                model.load_weights(self.weights_checkpoint_name)
            self.saved_model_dirname = self.name + 'saved_model'
        model.trainable = False

        if self.reparameterize:
            try:
                model = deploy_utils.reparameterize_model(model)
            except Exception as e:  # keep exporting: an unfused model is slower, not wrong
                logging.warning('Reparameterization failed (%s: %s); exporting the unfused model.',
                                type(e).__name__, e)
        if self.strip_qat:
            model = deploy_utils.strip_qat(model)
        return model

    def quantize(self):
        '''
        Prepare the model, convert it and write the full-integer and float16
        TFLite files. The flatbuffers are kept in `self.tflite_ui8_model`
        (the full-integer one, whatever its input/output type) and
        `self.tflite_f16_model`, the file paths in `self.tflite_int_path`
        and `self.tflite_f16_path`, the prepared Keras model in
        `self.export_model`, and the check of the full-integer file in
        `self.lint_report`. Sets `model.trainable = False` on the model as a
        side effect.
        '''

        def representative_data_gen():
            for i in range(self.batches):
                vals = self.dataset.__iter__().next()[0]
                for val in vals:
                    yield [tf.expand_dims(tf.cast(val, tf.float32), axis=0)]

        self.export_model = self._prepare_export_model()

        self.equalization = None
        if self.equalize:
            if deploy_utils.has_qat_layers(self.export_model):
                logging.warning('Cross-layer equalization skipped: the model still has QAT wrappers '
                                '(it needs `strip_qat=True`).')
            else:
                self.equalization = deploy_utils.equalize_cross_layer(self.export_model)
                logging.info('Cross-layer equalization: %d layer pairs, %d sweeps.',
                             self.equalization.pairs, self.equalization.iterations)
        try:
            self._convert_and_write(representative_data_gen)
        finally:
            # Layers may be shared with the model being trained: leave its
            # weights as they were.
            if self.equalization is not None:
                self.equalization.restore()

    def _convert_and_write(self, representative_data_gen):
        self.tflite_ui8_model = deploy_utils.convert_to_tflite(
            self.export_model, representative_data_gen, profile=self.profile)
        suffix = 'i8' if self.profile.io_dtype == 'int8' else 'ui8'
        self.tflite_int_path = f'{self.name}quant_{suffix}.tflite'
        with open(self.tflite_int_path, 'wb') as f:
            f.write(self.tflite_ui8_model)

        self.lint_report = deploy_utils.lint_tflite(self.tflite_ui8_model, profile=self.profile)
        if self.lint_report.ok:
            logging.info('%s', self.lint_report)
        else:
            logging.warning('%s', self.lint_report)
            if self.strict:
                raise RuntimeError('Exported model does not match profile \'{}\':\n{}'.format(
                    self.profile.name, self.lint_report))

        self.tflite_f16_model = deploy_utils.convert_to_tflite(self.export_model, float16=True)
        self.tflite_f16_path = f'{self.name}quant_f16.tflite'
        with open(self.tflite_f16_path, 'wb') as f:
            f.write(self.tflite_f16_model)

    def vizualize_ui8_results(self, num_images):

        '''Run the full-integer model on `num_images` images of one dataset batch
        and plot image, predicted mask and reference mask. Segmentation models
        only. Call `quantize()` first.
        '''
        self.vizualize_results(num_images, self.tflite_ui8_model, -1)

    def vizualize_f16_results(self, num_images):

        '''Same as `vizualize_ui8_results`, for the float16 model.
        '''
        self.vizualize_results(
            num_images, self.tflite_f16_model, self.normalization)

    def vizualize_results(self, num_images, model, normalization):

        '''Shared implementation of the two `vizualize_*` methods.

        Input
        ----------
            num_images: number of images of the first batch to show.
            model: TFLite flatbuffer bytes.
            normalization: key of `totflite_dict` selecting how a dataset image
                           is converted to the model's input type (`-1`: uint8).
        '''
        interpreter = tf.lite.Interpreter(model_content=model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        print(input_details)
        output_details = interpreter.get_output_details()
        print(output_details)

        it = next(self.dataset.__iter__())

        images = it[0]
        labels = it[1]

        fig = plt.figure(figsize=(22, 22))
        for i in range(num_images):
            interpreter.set_tensor(input_details[0]['index'], np.expand_dims(
                totflite_dict[normalization](images[i]), axis=0))
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = np.argmax(output_data, axis=3)[0]
            visualize(
                image=images[i],
                predicted_mask=prediction*255,
                reference_mask=np.argmax(labels[i], axis=-1)*255,
            )
