"""Score every export variant of a trained segmentation model, without training.

Loads a trained model from a weights checkpoint, exports it to full-integer
TFLite in several ways (QAT wrappers kept, post-training per channel, per
tensor with and without batch-norm folding, cross-layer equalization and
bias correction) and reports the mean intersection-over-union of each on a
fixed set of images. This is how the "Measured task accuracy" table of
`docs/source/deployment.rst` was produced.

The two built-in model configurations are those of the example notebooks
(Oxford-IIIT Pets, three classes, 256 x 256):

    python scripts/measure_export_accuracy.py ppliteseg \
        --data docs/source/notebooks/oxford_iiit \
        --checkpoint docs/source/notebooks/pets_mobilenetv4_pp_liteseg_<stamp>.weights \
        --backbone-weights pretrained_weights/mobilenetv4/MobileNetV4_ConvS.weights.h5

    python scripts/measure_export_accuracy.py lraspp \
        --data docs/source/notebooks/oxford_iiit \
        --checkpoint docs/source/notebooks/pets_mobileone_lraspp_<stamp>.weights \
        --backbone-weights pretrained_weights/mobileone/MobileOne_S1.weights.h5

`--data` must contain `images/*.jpg` and `annotations/trimaps/*.png`. If the
images overlap the training set of the checkpoint, read the figures as a
comparison between variants, not as test accuracy.
"""

import argparse
import glob
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import swiss_army_keras  # noqa: E402,F401  (sets the Keras 2 environment before TensorFlow loads)
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402

from swiss_army_keras import deploy_utils as du  # noqa: E402
from swiss_army_keras.models import lr_aspp, pp_liteseg  # noqa: E402

N_CLASSES = 3


def build(which, size, backbone_weights, checkpoint):
    inp = tf.keras.layers.Input((size, size, 3), batch_size=1)
    if which == 'ppliteseg':
        model = pp_liteseg(inp, n_labels=N_CLASSES, backbone='MobileNetV4-Conv-S', weights=backbone_weights,
                           freeze_backbone=True, name='Pets')
        model.load_weights(checkpoint)
    else:
        model = lr_aspp(inp, n_labels=N_CLASSES, backbone='MobileOne-S1', num_filters=128, num_filters_shallow=48,
                        activation='ReLU', batch_norm=True, weights=backbone_weights, freeze_backbone=True,
                        name='Pets')
        # the notebook saved an outer model wrapping the decoder: load through the same structure
        tf.keras.models.Model([inp], model(inp)).load_weights(checkpoint)
    return model


def load_images(data, size, count, seed):
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(data, 'images', '*.jpg'))
                   if os.path.exists(os.path.join(data, 'annotations', 'trimaps', os.path.basename(p)[:-4] + '.png')))
    picked = np.random.default_rng(seed).permutation(len(names))[:count]
    images, labels = [], []
    for index in picked:
        name = names[index]
        image = Image.open(os.path.join(data, 'images', name + '.jpg')).convert('RGB').resize((size, size), Image.BILINEAR)
        label = Image.open(os.path.join(data, 'annotations', 'trimaps', name + '.png')).resize((size, size), Image.NEAREST)
        images.append(np.asarray(image, dtype=np.float32))
        labels.append(np.asarray(label, dtype=np.int64) - 1)
    return np.stack(images), np.stack(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('model', choices=('ppliteseg', 'lraspp'))
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--backbone-weights', required=True)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--eval-images', type=int, default=150)
    parser.add_argument('--calibration-images', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    model = build(args.model, args.size, args.backbone_weights, args.checkpoint)
    raw, labels = load_images(args.data, args.size, args.eval_images + args.calibration_images, args.seed)
    x = np.asarray(model.preprocessing(raw), dtype=np.float32)
    x_cal, x_eval, y_eval = x[:args.calibration_images], x[args.calibration_images:], labels[args.calibration_images:]

    def miou(prediction):
        confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        np.add.at(confusion, (y_eval.reshape(-1), prediction.reshape(-1)), 1)
        iou = np.diag(confusion) / (confusion.sum(0) + confusion.sum(1) - np.diag(confusion))
        return float(iou.mean())

    def keras_prediction(m):
        return np.concatenate([m.predict(x_eval[i:i + 1], verbose=0) for i in range(len(x_eval))]).argmax(-1)

    def tflite_prediction(flatbuffer):
        interpreter = tf.lite.Interpreter(model_content=flatbuffer)
        interpreter.allocate_tensors()
        inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        scale, zero = inp['quantization']
        info = np.iinfo(inp['dtype'])
        results = []
        for i in range(len(x_eval)):
            quantized = np.clip(np.round(x_eval[i:i + 1] / scale + zero), info.min, info.max).astype(inp['dtype'])
            interpreter.set_tensor(inp['index'], quantized)
            interpreter.invoke()
            results.append(interpreter.get_tensor(out['index']).argmax(-1))
        return np.concatenate(results)

    def representative():
        for i in range(len(x_cal)):
            yield [x_cal[i:i + 1]]

    def report(label, prediction, note=''):
        print(f'{label:62s} mIoU {miou(prediction):.4f}  {note}', flush=True)

    report('float, QAT wrappers (as trained)', keras_prediction(model))
    plain = du.strip_qat(du.reparameterize_model(model))
    report('float, fused and stripped', keras_prediction(plain))

    flatbuffer = du.convert_to_tflite(model, representative, profile='default')
    lint = du.lint_tflite(flatbuffer, profile='default')
    report('QAT wrappers kept, per channel (default profile)', tflite_prediction(flatbuffer),
           '' if lint.ok else 'lint: ' + '; '.join(lint.errors)[:80])
    report('post-training, per channel', tflite_prediction(du.convert_to_tflite(plain, representative, profile='artpec9')))
    report('post-training, per tensor', tflite_prediction(du.convert_to_tflite(plain, representative, profile='artpec8')))

    equalized = du.equalize_cross_layer(plain)
    report('per tensor, equalized', tflite_prediction(du.convert_to_tflite(plain, representative, profile='artpec8')))
    equalized.restore()

    folded = du.fold_batch_norms(plain)
    corrected = du.correct_bias(folded, x_cal, per_channel=False)
    report('per tensor, folded and bias-corrected',
           tflite_prediction(du.convert_to_tflite(folded, representative, profile='artpec8')))
    corrected.restore()
    equalized = du.equalize_cross_layer(folded)
    report('per tensor, folded and equalized',
           tflite_prediction(du.convert_to_tflite(folded, representative, profile='artpec8')))
    corrected = du.correct_bias(folded, x_cal, per_channel=False)
    report('per tensor, folded, equalized and bias-corrected',
           tflite_prediction(du.convert_to_tflite(folded, representative, profile='artpec8')))
    corrected.restore()
    equalized.restore()
    du.correct_bias(folded, x_cal, per_channel=True)
    report('per channel, folded and bias-corrected',
           tflite_prediction(du.convert_to_tflite(folded, representative, profile='artpec9')))


if __name__ == '__main__':
    main()
