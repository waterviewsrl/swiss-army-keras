"""`DepthAlbumentationsDataLoader` on a small synthetic dataset.

The loaders start a pool of augmentation worker processes that live until
the interpreter exits, so this file is marked `slow` and kept out of the
default run. It is the only test of the data pipeline: it also builds a
`SegmentationAlbumentationsDataLoader` on the same images, because the two
share the worker protocol.
"""

import numpy as np
import pytest
import tensorflow as tf

albu = pytest.importorskip("albumentations")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("IronDomo")

from swiss_army_keras.dataset_utils import (  # noqa: E402
    DepthAlbumentationsDataLoader,
    SegmentationAlbumentationsDataLoader,
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
