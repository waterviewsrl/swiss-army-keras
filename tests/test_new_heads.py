"""Heads added on 2026-11-10: the plain pyramid head (`fpn_seg`,
`density_head`), the coordinate-classification keypoint head (`simcc_head`),
the re-identification training wrapper, learned loss balancing for
`multi_task`, and the distillation wrapper.

The fit calls below run two or three optimizer steps on a handful of random
tensors: they check the mechanics (what is trained, what is logged), not
learning.
"""

import numpy as np
import pytest
import tensorflow as tf

from swiss_army_keras import losses
from swiss_army_keras.distillation_utils import Distiller, channel_distillation_loss, logit_distillation_loss
from swiss_army_keras.models import (
    density_head,
    fpn_seg,
    multi_task,
    reid_embedding_head,
    reid_training_model,
    simcc_decode,
    simcc_encode,
    simcc_head,
    uncertainty_weighted,
)

SIZE = 64


# --------------------------------------------------------------------------- #
# plain pyramid head

@pytest.mark.parametrize("backbone, n_merges", [("ResNet18", 3), ("RepViT-M0.9", 3), ("StrideFormer-Tiny", 2)])
def test_fpn_seg_builds_on_backbones_with_five_four_and_three_levels(backbone, n_merges):
    size = 128   # StrideFormer needs an input that stays even down to stride 32
    model = fpn_seg(tf.keras.Input((size, size, 3)), 4, backbone=backbone, weights=None, freeze_backbone=False)
    assert model.output_shape == (None, size, size, 4)
    # the additions are QAT-wrapped by `quantize_apply`, so count them by name
    assert sum('fpn_seg_merge' in layer.name for layer in model.layers) == n_merges


def test_fpn_seg_uses_only_convolutions_additions_and_resizes_by_two():
    model = fpn_seg(tf.keras.Input((SIZE, SIZE, 3)), 3, backbone="ResNet18", weights=None)
    inner = {type(getattr(layer, "layer", layer)).__name__ for layer in model.layers}
    assert inner <= {"InputLayer", "Functional", "Conv2D", "BatchNormalization", "Activation", "Add", "UpSampling2D"}
    resizes = [layer.layer for layer in model.layers
               if type(getattr(layer, "layer", None)).__name__ == "UpSampling2D"]
    assert resizes and all(tuple(layer.size) == (2, 2) for layer in resizes)


def test_fpn_seg_options_and_errors():
    regression = fpn_seg(tf.keras.Input((SIZE, SIZE, 3)), 1, output_activation="sigmoid", levels=(3, 4, 5),
                         num_filters=32, backbone="CSP-N", weights=None)
    out = regression.predict(np.random.rand(1, SIZE, SIZE, 3).astype("float32"), verbose=0)
    assert out.shape == (1, SIZE, SIZE, 1) and 0.0 <= out.min() and out.max() <= 1.0

    with pytest.raises(ValueError, match="output_activation"):
        fpn_seg(tf.keras.Input((SIZE, SIZE, 3)), 2, output_activation="tanh", backbone="CSP-N", weights=None)
    with pytest.raises(ValueError, match="Choose consecutive levels"):
        fpn_seg(tf.keras.Input((SIZE, SIZE, 3)), 2, levels=(2, 4), backbone="CSP-N", weights=None)
    with pytest.raises(ValueError, match="multiples of 32"):
        fpn_seg(tf.keras.Input((70, 70, 3)), 2, backbone="CSP-N", weights=None)


def test_density_head_is_non_negative_and_density_targets_sum_to_the_count():
    model = density_head(tf.keras.Input((SIZE, SIZE, 3)), backbone="CSP-N", weights=None, freeze_backbone=False)
    out = model.predict(np.random.rand(2, SIZE, SIZE, 3).astype("float32"), verbose=0)
    assert out.shape == (2, SIZE, SIZE, 1) and out.min() >= 0.0

    target = losses.points_to_density([(10, 12), (40.5, 30.2), (500, 3)], (SIZE, SIZE), sigma=3.0)
    assert target.shape == (SIZE, SIZE, 1)
    np.testing.assert_allclose(target.sum(), 2.0, rtol=1e-5)          # the third point is outside

    loss = losses.density_count_loss(count_weight=0.5)
    y_true = np.stack([target, target])
    assert float(loss(y_true, y_true)) == pytest.approx(0.0, abs=1e-7)
    # a map with the right shape but twice the mass: the count term sees one object too many per target object
    doubled = float(loss(y_true, 2 * y_true))
    assert doubled == pytest.approx(float(np.mean(np.square(y_true))) + 0.5 * 2.0, rel=1e-4)


# --------------------------------------------------------------------------- #
# coordinate-classification keypoints

def test_simcc_encode_decode_round_trip_and_missing_keypoints():
    keypoints = np.array([[[10.2, 20.7], [50.0, 100.0], [95.4, 127.2], [0.0, 0.0]]], dtype="float32")
    visible = np.array([[1, 1, 1, 0]])
    x_target, y_target = simcc_encode(keypoints, visible, (128, 96), split_ratio=2.0)

    assert x_target.shape == (1, 4, 192) and y_target.shape == (1, 4, 256)
    np.testing.assert_allclose(x_target.sum(-1), [[1, 1, 1, 0]], atol=1e-5)
    decoded, scores = simcc_decode(x_target, y_target, split_ratio=2.0)
    np.testing.assert_allclose(decoded[0, :3], keypoints[0, :3], atol=0.26)     # half-pixel bins
    assert scores.shape == (1, 4)


def test_simcc_loss_is_zero_for_a_perfect_prediction_and_ignores_unlabelled_rows():
    keypoints = np.array([[[30.0, 40.0], [60.0, 80.0]]], dtype="float32")
    x_target, _ = simcc_encode(keypoints, np.array([[1, 0]]), (128, 96))
    perfect = np.log(x_target + 1e-9)
    assert float(losses.simcc_loss(x_target, perfect)) < 1e-4
    uniform = float(losses.simcc_loss(x_target, np.zeros_like(x_target)))
    assert uniform > 1.0
    # garbage on the unlabelled keypoint changes nothing
    noisy = np.zeros_like(x_target)
    noisy[0, 1] = np.random.default_rng(0).normal(size=x_target.shape[-1]) * 10
    assert float(losses.simcc_loss(x_target, noisy)) == pytest.approx(uniform, rel=1e-5)


def test_simcc_head_outputs_and_attributes():
    model = simcc_head(tf.keras.Input((128, 96, 3)), 5, backbone="CSP-N", weights=None, freeze_backbone=False,
                       hidden=32, final_kernel=3)
    assert model.output_names == ["x_logits", "y_logits"]
    assert model.output_shape == [(None, 5, 192), (None, 5, 256)]
    assert (model.x_bins, model.y_bins, model.split_ratio) == (192, 256, 2.0)
    x_logits, y_logits = model.predict(np.random.rand(2, 128, 96, 3).astype("float32"), verbose=0)
    positions, _ = simcc_decode(x_logits, y_logits)
    assert positions.shape == (2, 5, 2)
    assert positions[..., 0].max() < 96 and positions[..., 1].max() < 128
    with pytest.raises(ValueError, match="feature_level"):
        simcc_head(tf.keras.Input((128, 96, 3)), 5, backbone="CSP-N", weights=None, feature_level=6)


# --------------------------------------------------------------------------- #
# re-identification

def test_reid_training_model_shares_the_embedding_and_outputs_cosines():
    embedding_model = reid_embedding_head(
        tf.keras.Input((SIZE, SIZE, 3)), embedding_size=16, size=32, backbone="CSP-N", weights=None,
        freeze_backbone=False, pre_pool_attention=None)
    training_model = reid_training_model(embedding_model, n_identities=7)

    assert training_model.output_names == ["embedding", "identity"]
    x = np.random.rand(3, SIZE, SIZE, 3).astype("float32")
    embedding, cosines = training_model.predict(x, verbose=0)
    np.testing.assert_allclose(embedding, embedding_model.predict(x, verbose=0), atol=1e-6)
    assert cosines.shape == (3, 7) and np.abs(cosines).max() <= 1.0 + 1e-5
    assert any(layer.name == "reid_train_bnneck" for layer in training_model.layers)
    assert not any("bnneck" in layer.name for layer in embedding_model.layers)


def test_margin_losses_hand_computed():
    y_true = tf.constant([[1.0, 0.0]])
    cosines = tf.constant([[0.5, 0.2]])
    scale, margin = 10.0, 0.3
    target = np.cos(np.arccos(0.5) + margin)
    expected_arc = -np.log(np.exp(scale * target) / (np.exp(scale * target) + np.exp(scale * 0.2)))
    np.testing.assert_allclose(float(losses.arcface_loss(margin, scale)(y_true, cosines)), expected_arc, rtol=1e-5)
    expected_cos = -np.log(np.exp(scale * (0.5 - margin)) / (np.exp(scale * (0.5 - margin)) + np.exp(scale * 0.2)))
    np.testing.assert_allclose(float(losses.cosface_loss(margin, scale)(y_true, cosines)), expected_cos, rtol=1e-5)
    # a margin makes the same prediction cost more than plain scaled softmax
    assert float(losses.arcface_loss(0.3, scale)(y_true, cosines)) > float(losses.arcface_loss(0.0, scale)(y_true, cosines))


# --------------------------------------------------------------------------- #
# learned loss balancing and distillation

def test_uncertainty_weighted_trains_the_shared_model_and_its_task_weights():
    tf.keras.utils.set_random_seed(0)
    base = multi_task(tf.keras.Input((SIZE, SIZE, 3)), {
        "seg": dict(type="fpn_seg", n_labels=3, num_filters=16),
        "cls": dict(type="classifier", n_classes=4, size=16, logits=True),
    }, backbone="CSP-N", weights=None, freeze_backbone=False)
    model = uncertainty_weighted(base)
    assert model.task_weights() == {"seg": 1.0, "cls": 1.0}

    cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss={"seg": cross_entropy, "cls": cross_entropy})
    x = np.random.rand(4, SIZE, SIZE, 3).astype("float32")
    y = {"seg": tf.one_hot(np.random.randint(0, 3, (4, SIZE, SIZE)), 3).numpy(),
         "cls": tf.one_hot(np.random.randint(0, 4, (4,)), 4).numpy()}
    before = base.get_layer("quant_seg_head").weights[0].numpy().copy()
    history = model.fit(x, y, batch_size=4, epochs=2, verbose=0).history

    assert {"loss", "seg_loss", "cls_loss", "weighted_loss"} <= set(history)
    # first step: all weights are 1, so the weighted loss equals the plain sum
    assert history["weighted_loss"][0] == pytest.approx(history["loss"][0], rel=1e-4)
    assert all(weight != 1.0 for weight in model.task_weights().values())
    assert np.abs(base.get_layer("quant_seg_head").weights[0].numpy() - before).max() > 0     # layers are shared
    with pytest.raises(ValueError, match="at least two outputs"):
        uncertainty_weighted(fpn_seg(tf.keras.Input((SIZE, SIZE, 3)), 2, backbone="CSP-N", weights=None))


def _toy_segmenter(width, name):
    inp = tf.keras.Input((32, 32, 3))
    hidden = tf.keras.layers.Conv2D(width, 3, padding="same", activation="relu")(inp)
    return tf.keras.Model(inp, tf.keras.layers.Conv2D(3, 1)(hidden), name=name)


def test_distillation_losses_are_zero_for_identical_outputs_and_positive_otherwise():
    rng = np.random.default_rng(0)
    a = tf.constant(rng.normal(size=(2, 8, 8, 3)).astype("float32"))
    b = tf.constant(rng.normal(size=(2, 8, 8, 3)).astype("float32"))
    for fn in (logit_distillation_loss, channel_distillation_loss):
        assert float(fn(a, a)) == pytest.approx(0.0, abs=1e-6)
        assert float(fn(a, b)) > 0.0
    # a teacher of another resolution is resized to the student's
    coarse = tf.constant(rng.normal(size=(2, 4, 4, 3)).astype("float32"))
    assert np.isfinite(float(channel_distillation_loss(coarse, b)))


@pytest.mark.parametrize("mode", ["logits", "channel", "mse"])
def test_distiller_trains_the_student_only(mode):
    tf.keras.utils.set_random_seed(0)
    teacher, student = _toy_segmenter(16, "teacher"), _toy_segmenter(4, "student")
    distiller = Distiller(student, teacher, mode=mode, alpha=0.5)
    distiller.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    x = np.random.rand(8, 32, 32, 3).astype("float32")
    y = tf.one_hot(np.random.randint(0, 3, (8, 32, 32)), 3).numpy()
    teacher_before, student_before = teacher.get_weights()[0].copy(), student.get_weights()[0].copy()
    history = distiller.fit(x, y, batch_size=4, epochs=1, verbose=0, validation_data=(x, y)).history

    assert {"loss", "task_loss", "distillation_loss", "accuracy", "val_loss", "val_accuracy"} <= set(history)
    assert "val_distillation_loss" not in history
    np.testing.assert_array_equal(teacher.get_weights()[0], teacher_before)
    assert np.abs(student.get_weights()[0] - student_before).max() > 0
    np.testing.assert_allclose(distiller.predict(x, verbose=0), student.predict(x, verbose=0), atol=1e-6)


def test_distiller_rejects_bad_arguments():
    teacher, student = _toy_segmenter(8, "t"), _toy_segmenter(4, "s")
    with pytest.raises(ValueError, match="Unknown mode"):
        Distiller(student, teacher, mode="hint")
    with pytest.raises(ValueError, match="alpha"):
        Distiller(student, teacher, alpha=1.5)


# --------------------------------------------------------------------------- #
# building without QAT wrappers

@pytest.mark.parametrize("build", [
    lambda i: fpn_seg(i, 3, backbone="CSP-N", weights=None),
    lambda i: multi_task(i, {"seg": dict(type="lr_aspp", n_labels=3), "cls": dict(type="classifier", n_classes=2, size=8)},
                         backbone="CSP-N", weights=None),
    lambda i: simcc_head(i, 4, backbone="CSP-N", weights=None, hidden=16, final_kernel=3),
    lambda i: __import__("swiss_army_keras.models", fromlist=["ddrnet"]).ddrnet(i, n_labels=3, augment=True),
    lambda i: __import__("swiss_army_keras.models", fromlist=["pidnet"]).pidnet(i, n_labels=3, augment=False),
    lambda i: __import__("swiss_army_keras._model_detection", fromlist=["x"]).fcos_lite_detection_base(
        i, n_classes=2, backbone="CSP-N", weights=None),
    lambda i: __import__("swiss_army_keras.models", fromlist=["x"]).fastdepth_head(i, backbone="CSP-N", weights=None),
], ids=["fpn_seg", "multi_task", "simcc", "ddrnet", "pidnet", "fcos_base", "fastdepth"])
def test_builders_return_plain_models_when_qat_is_disabled(build):
    from swiss_army_keras import deploy_utils
    from swiss_army_keras.layer_utils import qat_options

    with qat_options(enabled=False):
        model = build(tf.keras.Input((SIZE, SIZE, 3)))
    assert not deploy_utils.has_qat_layers(model)
    assert not any(layer.name.startswith("quant") for layer in model.layers)
    outputs = model.predict(np.random.rand(1, SIZE, SIZE, 3).astype("float32"), verbose=0)
    assert all(np.all(np.isfinite(o)) for o in (outputs if isinstance(outputs, list) else [outputs]))
