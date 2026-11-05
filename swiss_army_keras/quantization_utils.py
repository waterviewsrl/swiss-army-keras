'''TFLite export: `Quantizer` converts a trained model to full-integer (uint8
I/O) and float16 TFLite files and can visualize their predictions.
'''
import tensorflow as tf
import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt


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
    '''TFLite export of a trained model, in two flavours.

    `quantize()` writes two files (prefix `name`, relative to the current
    working directory):

    * `<name>_quant_ui8.tflite`: full-integer model. Only
      `TFLITE_BUILTINS_INT8` ops are allowed (conversion fails otherwise),
      input and output tensors are `uint8`, and `batches` batches of
      `dataset` are used as the representative (calibration) dataset.
    * `<name>_quant_f16.tflite`: float16 weights, float32 input/output.

    Models built by this library are already wrapped for quantization-aware
    training, so the fake-quant ranges learned during training are used; the
    representative dataset only calibrates tensors those wrappers do not
    cover.

    Current limitations (see `docs/source/deployment.rst`): weights are
    quantized per channel (the TFLite default; a per-tensor switch for
    targets such as the Axis ARTPEC-8 DLPU is not exposed yet), the integer
    I/O type is fixed to uint8, and structural reparameterization
    (`reparameterize_mobileone` etc.) is not applied automatically.

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
    '''
    def __init__(self, dataset, model, name, append_datetime=True, batches=1, weights_checkpoint_name=None):
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

    def quantize(self):

        '''Convert the model and write the `_quant_ui8.tflite` and
        `_quant_f16.tflite` files. The converted flatbuffers are also kept in
        `self.tflite_ui8_model` and `self.tflite_f16_model`. Sets
        `model.trainable = False` on the model as a side effect.
        '''
        def representative_data_gen():
            # for input_value in tf.data.Dataset.from_tensor_slices(test_input).batch(1).take(1):
            #    yield [tf.cast(input_value, tf.float32) /255.]
            for i in range(self.batches):
                vals = self.dataset.__iter__().next()[0]
                for val in vals:
                    yield [tf.expand_dims(tf.cast(val, tf.float32), axis=0)]

        if isinstance(self.model, str):
            loaded_model = tf.keras.models.load_model(
                self.model,
                compile=False)
            loaded_model.trainable = False
            converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
            self.saved_model_dirname = self.model
        else:
            if self.weights_checkpoint_name is not None:
                self.model.load_weights(self.weights_checkpoint_name)
            self.model.trainable = False
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            self.saved_model_dirname = self.name + 'saved_model'

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        # Ensure that if any ops can't be quantized, the converter throws an error
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Set the input and output tensors to uint8 (APIs added in r2.3)
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        self.tflite_ui8_model = converter.convert()

        with open(f'{self.name}quant_ui8.tflite', 'wb') as f:
            f.write(self.tflite_ui8_model)

        if isinstance(self.model, str):
            loaded_model = tf.keras.models.load_model(
                self.model,
                compile=False)
            loaded_model.trainable = False
            converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
            self.saved_model_dirname = self.model
        else:
            self.model.trainable = False
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            self.saved_model_dirname = self.name + 'saved_model'

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        # Ensure that if any ops can't be quantized, the converter throws an error
        self.tflite_f16_model = converter.convert()

        with open(f'{self.name}quant_f16.tflite', 'wb') as f:
            f.write(self.tflite_f16_model)

        '''params = tf.experimental.tensorrt.ConversionParams(
            precision_mode='INT8',
            maximum_cached_engines=1,
            use_calibration=True)
        converter = tf.experimental.tensorrt.Converter(
            input_saved_model_dir=self.saved_model_dirname, conversion_params=params)

        converter.convert(calibration_input_fn=representative_data_gen)
        converter.save(self.name + 'tensorrt_ui8')

        params = tf.experimental.tensorrt.ConversionParams(
            precision_mode='FP16',
            maximum_cached_engines=4)
        converter = tf.experimental.tensorrt.Converter(
            input_saved_model_dir=self.saved_model_dirname, conversion_params=params)

        converter.convert()
        converter.save(self.name + 'tensorrt_f16')'''

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
