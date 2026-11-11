"""Compare whole-model QAT with a plain float fine-tune of the same length.

Starting from a trained segmentation checkpoint, either

* `--mode qat`: prepares the model with `deploy_utils.prepare_qat`,
  fine-tunes it for a fixed wall-clock budget, and exports it with its
  wrappers, or
* `--mode control`: fine-tunes the float network that `prepare_qat` wraps
  (same graph, same images, same budget, no fake quantization), then
  exports it post-training, the usual way (with bias correction for per
  tensor).

The control matters: fine-tuning alone changes accuracy, so QAT has to be
judged against a float model that was fine-tuned for as long. Both modes
report the mean intersection-over-union of the exported file on held-out
images. This is how the "Whole-model QAT" table of
`docs/source/deployment.rst` was produced.

    python scripts/measure_qat_finetune.py lraspp --mode qat --per-tensor \
        --data docs/source/notebooks/oxford_iiit \
        --checkpoint docs/source/notebooks/pets_mobileone_lraspp_<stamp>.weights \
        --backbone-weights pretrained_weights/mobileone/MobileOne_S1.weights.h5 --seconds 300

Images are split from one seeded permutation: calibration first, then
evaluation, then fine-tuning, so the three sets are disjoint. The
checkpoint's own training split was not recorded, so the evaluation images
may have been seen when it was trained. This runs training: it takes the
`--seconds` budget plus a few minutes, and about 4 GB of memory at 256x256.
"""

import argparse
import os
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import swiss_army_keras  # noqa: E402,F401  (sets the Keras 2 environment before TensorFlow loads)
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from swiss_army_keras import deploy_utils as du  # noqa: E402

from measure_export_accuracy import N_CLASSES, build  # noqa: E402

import glob  # noqa: E402
from PIL import Image  # noqa: E402


def load(data, size, count, seed):
    '''Images and trimaps as one float32 and one uint8 array (no copies).'''
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(data, 'images', '*.jpg'))
                   if os.path.exists(os.path.join(data, 'annotations', 'trimaps', os.path.basename(p)[:-4] + '.png')))
    picked = np.random.default_rng(seed).permutation(len(names))[:count]
    images = np.empty((len(picked), size, size, 3), dtype=np.float32)
    labels = np.empty((len(picked), size, size), dtype=np.uint8)
    for row, index in enumerate(picked):
        name = names[index]
        images[row] = np.asarray(Image.open(os.path.join(data, 'images', name + '.jpg')).convert('RGB')
                                 .resize((size, size), Image.BILINEAR), dtype=np.float32)
        labels[row] = np.asarray(Image.open(os.path.join(data, 'annotations', 'trimaps', name + '.png'))
                                 .resize((size, size), Image.NEAREST), dtype=np.uint8) - 1
    return images, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('model', choices=('ppliteseg', 'lraspp'))
    parser.add_argument('--mode', choices=('qat', 'control'), required=True)
    granularity = parser.add_mutually_exclusive_group(required=True)
    granularity.add_argument('--per-channel', dest='per_channel', action='store_true')
    granularity.add_argument('--per-tensor', dest='per_channel', action='store_false')
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--backbone-weights', required=True)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--calibration-images', type=int, default=32)
    parser.add_argument('--eval-images', type=int, default=150)
    parser.add_argument('--train-images', type=int, default=480)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    model = build(args.model, args.size, args.backbone_weights, args.checkpoint)
    total = args.calibration_images + args.eval_images + args.train_images
    x, y = load(args.data, args.size, total, args.seed)
    for start in range(0, len(x), 32):                           # normalize in place, in chunks
        x[start:start + 32] = np.asarray(model.preprocessing(x[start:start + 32]), dtype=np.float32)
    n_cal, n_eval = args.calibration_images, args.eval_images
    x_cal = x[:n_cal]
    x_eval, y_eval = x[n_cal:n_cal + n_eval], y[n_cal:n_cal + n_eval]
    x_train, y_train = x[n_cal + n_eval:], y[n_cal + n_eval:]
    profile = 'artpec9' if args.per_channel else 'artpec8'
    tag = '{} {}'.format(args.mode, 'per channel' if args.per_channel else 'per tensor')

    def miou(prediction):
        confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        np.add.at(confusion, (y_eval.reshape(-1).astype(np.int64), prediction.reshape(-1)), 1)
        iou = np.diag(confusion) / (confusion.sum(0) + confusion.sum(1) - np.diag(confusion))
        return float(iou.mean())

    def keras_prediction(m):
        return np.concatenate([m.predict(x_eval[i:i + 8], verbose=0) for i in range(0, len(x_eval), 8)]).argmax(-1)

    def tflite_prediction(flatbuffer):
        interpreter = tf.lite.Interpreter(model_content=flatbuffer)
        interpreter.allocate_tensors()
        inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        scale, zero = inp['quantization']
        info = np.iinfo(inp['dtype'])
        results = []
        for i in range(len(x_eval)):
            interpreter.set_tensor(inp['index'], np.clip(np.round(x_eval[i:i + 1] / scale + zero), info.min,
                                                         info.max).astype(inp['dtype']))
            interpreter.invoke()
            results.append(interpreter.get_tensor(out['index']).argmax(-1))
        return np.concatenate(results)

    def representative():
        for i in range(len(x_cal)):
            yield [x_cal[i:i + 1]]

    prepared = du.prepare_qat(model, x_cal, per_channel=args.per_channel)
    trained = prepared if args.mode == 'qat' else prepared.float_model
    print('{}: float {:.4f}; prepared before fine-tuning: simulated {:.4f}, exported {:.4f}'.format(
        tag, miou(keras_prediction(prepared.float_model)), miou(keras_prediction(prepared)),
        miou(tflite_prediction(du.convert_to_tflite(prepared, representative, profile=profile)))), flush=True)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    @tf.function
    def step(images, labels):
        with tf.GradientTape() as tape:
            loss = loss_fn(labels, trained(images, training=True))
        variables = trained.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients([(g, v) for g, v in zip(gradients, variables) if g is not None])
        return loss

    order, start, steps, losses = np.random.default_rng(args.seed + 1), time.time(), 0, []
    while time.time() - start < args.seconds:
        permutation = order.permutation(len(x_train))
        for k in range(0, len(permutation) - args.batch_size + 1, args.batch_size):
            batch = permutation[k:k + args.batch_size]
            losses.append(float(step(x_train[batch], y_train[batch].astype(np.int32))))
            steps += 1
            if time.time() - start >= args.seconds:
                break
    elapsed = time.time() - start
    print('{}: {} steps in {:.0f} s ({:.2f} s/step), loss {:.4f} -> {:.4f}'.format(
        tag, steps, elapsed, elapsed / max(steps, 1), np.mean(losses[:10]), np.mean(losses[-10:])), flush=True)

    if args.mode == 'qat':
        flatbuffer = du.convert_to_tflite(prepared, representative, profile=profile)
        report = du.lint_tflite(flatbuffer, profile=profile)
        print('{}: after fine-tuning: simulated {:.4f}, exported {:.4f}{}'.format(
            tag, miou(keras_prediction(prepared)), miou(tflite_prediction(flatbuffer)),
            '' if not report.errors else '  lint: ' + '; '.join(report.errors)), flush=True)
    else:
        float_model = prepared.float_model
        line = '{}: after fine-tuning: float {:.4f}, post-training export {:.4f}'.format(
            tag, miou(keras_prediction(float_model)),
            miou(tflite_prediction(du.convert_to_tflite(float_model, representative, profile=profile))))
        if not args.per_channel:
            du.correct_bias(float_model, x_cal, per_channel=False)
            line += ', with bias correction {:.4f}'.format(
                miou(tflite_prediction(du.convert_to_tflite(float_model, representative, profile=profile))))
        print(line, flush=True)


if __name__ == '__main__':
    main()
