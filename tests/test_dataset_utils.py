"""`DepthAlbumentationsDataLoader` on a small synthetic dataset.

The loaders start a pool of augmentation worker processes that live until
the interpreter exits, so this file is marked `slow` and kept out of the
default run. It is the only test of the data pipeline: it also builds a
`SegmentationAlbumentationsDataLoader` on the same images, because the two
share the worker protocol.
"""

import json
import os

import numpy as np
import pytest
import tensorflow as tf

albu = pytest.importorskip("albumentations")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("IronDomo")

from swiss_army_keras.dataset_utils import (  # noqa: E402
    DepthAlbumentationsDataLoader,
    ClassificationAlbumentationsDataLoader,
    SegmentationAlbumentationsDataLoader,
    read_split_manifest,
)

pytestmark = pytest.mark.slow

SIZE = 64


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    """12 images whose left half is bright, near (2 m) and of class 1; the
    right half is dark, far (40 m) and of class 0. The top rows have no
    depth and the bottom rows are beyond `max_depth`.
    """
    root = tmp_path_factory.mktemp("toy_depth")
    for directory in ("images", "depth", "depth_npy", "annotations"):
        (root / directory).mkdir()
    for i in range(12):
        image = np.zeros((80, 80, 3), np.uint8)
        image[:, :40] = 200
        depth = np.full((80, 80), 40.0, np.float32)
        depth[:, :40] = 2.0
        depth[:10] = 0.0
        depth[70:] = 200.0
        mask = np.zeros((80, 80), np.uint8)
        mask[:, :40] = 1
        cv2.imwrite(str(root / "images" / f"{i:03d}.png"), image)
        cv2.imwrite(str(root / "depth" / f"{i:03d}.png"), (depth * 256).astype(np.uint16))
        np.save(root / "depth_npy" / f"{i:03d}.npy", depth)
        cv2.imwrite(str(root / "annotations" / f"{i:03d}.png"), mask)
    return str(root)


def _first_batch(loader):
    train, _, _ = loader.build_datasets()
    images, targets = next(iter(train))
    return images.numpy(), targets.numpy()


def _left_is_brighter(images):
    return images[0, SIZE // 2, :SIZE // 2].mean() > images[0, SIZE // 2, SIZE // 2:].mean()


@pytest.mark.parametrize("options, near, far", [
    (dict(depths_dir="depth", target="depth"), 2.0, 40.0),
    (dict(depths_dir="depth", target="normalized"), 2.0 / 80.0, 0.5),
    (dict(depths_dir="depth_npy", depth_extension="npy", depth_divisor=1.0, target="inverse", min_depth=1.0), 0.5, 0.025),
], ids=["png-depth", "png-normalized", "npy-inverse"])
def test_depth_targets_units_and_missing_pixels(dataset_root, options, near, far):
    loader = DepthAlbumentationsDataLoader(
        dataset_root, width=SIZE, height=SIZE, batch_size=4, max_depth=80.0, seed=0, **options)
    images, depth = _first_batch(loader)

    assert images.shape == (4, SIZE, SIZE, 3) and depth.shape == (4, SIZE, SIZE, 1)
    assert depth.dtype == np.float32
    assert _left_is_brighter(images)
    row = depth[0, SIZE // 2, :, 0]
    np.testing.assert_allclose(row[:SIZE // 2 - 2], near, rtol=1e-4)
    np.testing.assert_allclose(row[SIZE // 2 + 2:], far, rtol=1e-4)
    assert (depth[0, :6] == 0).all(), "missing depth must stay 0"
    assert (depth[0, -6:] == 0).all(), "depth beyond max_depth must be marked missing"


def test_geometric_augmentation_moves_image_and_depth_together(dataset_root):
    flip = albu.Compose([albu.HorizontalFlip(p=1.0)])
    loader = DepthAlbumentationsDataLoader(
        dataset_root, train_augmentations=flip, width=SIZE, height=SIZE, batch_size=4, seed=0)
    images, depth = _first_batch(loader)

    assert not _left_is_brighter(images)
    row = depth[0, SIZE // 2, :, 0]
    assert row[:SIZE // 2 - 2].mean() == pytest.approx(40.0) and row[SIZE // 2 + 2:].mean() == pytest.approx(2.0)


def test_depth_loader_rejects_bad_options(dataset_root):
    with pytest.raises(ValueError, match="target"):
        DepthAlbumentationsDataLoader(dataset_root, target="disparity")
    with pytest.raises(ValueError, match="depth_extension"):
        DepthAlbumentationsDataLoader(dataset_root, depth_extension="tiff")
    with pytest.raises(ValueError, match="min_depth"):
        DepthAlbumentationsDataLoader(dataset_root, min_depth=10.0, max_depth=1.0)


def test_segmentation_loader_still_yields_one_hot_masks(dataset_root):
    flip = albu.Compose([albu.HorizontalFlip(p=1.0)])
    loader = SegmentationAlbumentationsDataLoader(
        dataset_root, train_augmentations=flip, width=SIZE, height=SIZE, batch_size=4, num_classes=2, seed=0)
    images, masks = _first_batch(loader)

    assert masks.shape == (4, SIZE, SIZE, 2) and masks.dtype == np.float32
    assert not _left_is_brighter(images)
    row = masks[0, SIZE // 2, :, 1]
    assert row[:SIZE // 2 - 2].mean() == 0.0 and row[SIZE // 2 + 2:].mean() == 1.0


# --- the recorded train / validation / test split ---


def test_split_is_recorded_and_reused(dataset_root, tmp_path):
    manifest_path = str(tmp_path / "runs" / "split.json")
    options = dict(width=SIZE, height=SIZE, batch_size=2, num_classes=2, train_val_test_split=[0.5, 0.25, 0.25])
    loader = SegmentationAlbumentationsDataLoader(dataset_root, seed=3, split_manifest=manifest_path, **options)
    with pytest.raises(RuntimeError):
        loader.save_split(str(tmp_path / "too_early.json"))
    loader.build_datasets()

    split = loader.split
    assert [len(split[name]) for name in ("train", "val", "test")] == [6, 3, 3]
    everything = split["train"] + split["val"] + split["test"]
    assert sorted(everything) == [f"images/{i:03d}.png" for i in range(12)]
    # the same order as the seeded shuffle of the file list that defined the split before it was recorded
    shuffled = tf.data.Dataset.from_tensor_slices(loader.images).shuffle(12, seed=3, reshuffle_each_iteration=False)
    assert everything == ["images/" + os.path.basename(p.decode()) for p in shuffled.as_numpy_iterator()]

    manifest = read_split_manifest(manifest_path)
    assert manifest["seed"] == 3 and manifest["counts"] == {"train": 6, "val": 3, "test": 3}
    assert {name: manifest[name] for name in ("train", "val", "test")} == split

    # another seed and other fractions: the manifest wins
    again = SegmentationAlbumentationsDataLoader(
        dataset_root, seed=99, split_manifest=manifest_path, width=SIZE, height=SIZE, batch_size=1, num_classes=2)
    train, val, test = again.build_datasets()
    assert again.split == split
    assert sum(1 for _ in test) == 3 and sum(1 for _ in val) == 3

    # a manifest that names a file the dataset does not have is refused
    manifest["test"][0] = "images/not_there.png"
    broken = str(tmp_path / "broken.json")
    with open(broken, "w") as f:
        json.dump(manifest, f)
    again.split_manifest = broken
    with pytest.raises(FileNotFoundError):
        again.build_datasets()


def test_classification_split_is_recorded(tmp_path):
    root = tmp_path / "classes"
    for label, value in (("bright", 220), ("dark", 30)):
        (root / label).mkdir(parents=True)
        for i in range(6):
            cv2.imwrite(str(root / label / f"{i}.png"), np.full((40, 40, 3), value, np.uint8))
    manifest_path = str(tmp_path / "class_split.json")
    options = dict(width=32, height=32, batch_size=2, train_val_test_split=[0.5, 0.25, 0.25])
    loader = ClassificationAlbumentationsDataLoader(str(root), seed=1, split_manifest=manifest_path, **options)
    loader.build_datasets()
    assert sorted(loader.split["train"] + loader.split["val"] + loader.split["test"]) == sorted(
        f"{label}/{i}.png" for label in ("bright", "dark") for i in range(6))

    again = ClassificationAlbumentationsDataLoader(str(root), seed=7, split_manifest=manifest_path, **options)
    _, _, test = again.build_datasets()
    assert again.split == loader.split
    # labels still follow the files: bright images are class 0
    for images, labels in test:
        for image, label in zip(images.numpy(), labels.numpy()):
            assert int(np.argmax(label)) == (0 if image.mean() > 0.5 else 1)
