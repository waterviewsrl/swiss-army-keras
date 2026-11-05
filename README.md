# swiss-army-keras

A Keras library for building, training and exporting compact vision models for
edge and IoT devices. It started as a fork of
[keras-unet-collection](https://github.com/yingkaisha/keras-unet-collection)
(see [README-keras-unet-collection.md](README-keras-unet-collection.md)) and
now covers the whole path from a folder of images to a quantized TFLite file.

The goal is to let you concentrate on the dataset, the architecture choice and
the hyper-parameters, without rebuilding the surrounding plumbing for every
experiment.

## What it provides

| Area | Content | Module |
|---|---|---|
| Dataset pipelines | `tf.data` loaders for segmentation and classification with [albumentations](https://albumentations.ai/) augmentation in worker processes | `dataset_utils` |
| Training | Frozen-backbone training, optional unfrozen fine-tuning, callbacks, TFLite export at the end | `training_utils` |
| Backbones | `tf.keras.applications` families, EfficientNet-Lite, and five edge-oriented families implemented here: MobileOne, MobileNetV4, FastViT, RepViT, StrideFormer | `_backbone_zoo` |
| Segmentation | DeepLabV3+, LR-ASPP, PP-LiteSeg, DDRNet, PIDNet, PP-MobileSeg, plus the inherited U-Net family | `models` |
| Detection | Anchor-free FCOS-lite and GFL-lite detectors, YOLACT-style instance segmentation | `models` |
| Other heads | Single-label, multi-label and ordinal classifiers, keypoint regression, re-identification embeddings, monocular depth, image-quality side head | `models` |
| Losses, optimizers, metrics | Region, boundary and focal losses; detection and depth losses; Lion, SAM, AdaBelief and others; boundary F-score, mean average precision | `losses`, `optimizers`, `metrics` |
| Deployment | QAT wrapping in every head, full-integer and float16 TFLite export, structural reparameterization, uint8 input wrapper, out-of-distribution scores, test-time augmentation | `quantization_utils`, `layer_utils`, `ood_utils`, `inference_utils` |

The complete inventory, with registry names and limitations, is in
[docs/source/model_zoo.rst](docs/source/model_zoo.rst).

## Installation

Python 3.9 or newer and TensorFlow 2.16 or newer (CPU or GPU).

    pip3 install swiss-army-keras

From a checkout:

    pip3 install -e .

The optional EfficientNet-Lite B0-B4 backbones need the
[waterviewsrl/efficientnet-lite-keras](https://github.com/waterviewsrl/efficientnet-lite-keras)
fork, which supports Python 3.11 at most:

    pip3 install "swiss-army-keras[efficientnet-lite]"

### Import order

The library runs on the legacy Keras 2 API (the `tf-keras` package, installed
automatically), because the quantization-aware-training layers of
`tensorflow_model_optimization` do not support Keras 3. Import the library
before TensorFlow so the switch takes effect:

```python
import swiss_army_keras   # sets TF_USE_LEGACY_KERAS=1
import tensorflow as tf
```

If TensorFlow was imported first, the library raises an `ImportError` that
explains the fix.

## Quick start

```python
import swiss_army_keras
import tensorflow as tf
import albumentations as albu

from swiss_army_keras import losses, metrics
from swiss_army_keras.dataset_utils import SegmentationAlbumentationsDataLoader
from swiss_army_keras.models import lr_aspp
from swiss_army_keras.optimizers import AdaBeliefOptimizer
from swiss_army_keras.training_utils import TrainingDriver

IMG_SIZE, N_LABELS = 256, 3

# 1. model: any registered backbone with any decoder
model = lr_aspp(
    tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 3)),
    n_labels=N_LABELS, backbone="MobileOne-S1", weights=None,
    batch_norm=True, freeze_backbone=True,
)

# 2. data: images/ and a directory of PNG masks holding class ids
loader = SegmentationAlbumentationsDataLoader(
    dataset_path="/data/pets", masks_dir="annotations/trimaps",
    num_classes=N_LABELS, label_shift=1,
    width=IMG_SIZE, height=IMG_SIZE, batch_size=16,
    train_augmentations=albu.Compose([albu.HorizontalFlip(p=0.5)]),
    normalization=model.preprocessing, seed=0,
)
train_set, val_set, test_set = loader.build_datasets()

# 3. train (frozen, then unfrozen) and export full-integer and float16 TFLite
TrainingDriver(
    model, "pets_mobileone_lraspp",
    AdaBeliefOptimizer(learning_rate=1e-3),
    losses.loss_adapter(losses.symmetrical_unified_focal_loss(), from_logits=True),
    [metrics.MeanIOU(num_classes=N_LABELS)],
    train_set, val_set, test_set,
    epochs=20, unfreezed_epochs=20,
).run()
```

The step-by-step version, the classification flow and the list of output
files are in [docs/source/getting_started.rst](docs/source/getting_started.rst).

## Pretrained weights

`weights='imagenet'` works for the `tf.keras.applications` backbones. For
MobileOne, MobileNetV4, FastViT and RepViT, ImageNet weights are converted
from the `timm` checkpoints with the scripts in [scripts/](scripts/) and then
passed as a file path; automatic download is not available yet. See the
"Pretrained weights" section of the model zoo page.

## Deployment

Models are exported as TFLite files, full-integer (uint8 input and output) and
float16. Before targeting an accelerator, read
[docs/source/deployment.rst](docs/source/deployment.rst): it covers
per-tensor quantization, structural reparameterization (not applied
automatically yet), the operators each model produces, and the constraints of
the Axis DLPU family and other targets.

## Documentation

| Document | Content |
|---|---|
| [Online documentation](https://swiss-army-keras.readthedocs.io/en/latest/index.html) | User guide, examples, API reference (sources in [docs/source/](docs/source/)) |
| [docs/source/notebooks/](docs/source/notebooks/) | Example notebooks |
| [use_case_recipes.md](use_case_recipes.md) | Loss and hyper-parameter recipes for specific tasks |
| [EDGE_AI_CANDIDATES.md](EDGE_AI_CANDIDATES.md) | Assessment of further networks, optimizers and deployment strategies, by target hardware |
| [ROADMAP.md](ROADMAP.md) | Open work, in priority order |
| [ASSESSMENT.md](ASSESSMENT.md) | Project reassessment of November 2026 |
| [plan.md](plan.md) | Engineering log of the 0.11 development cycle |
| [CHANGELOG.md](CHANGELOG.md), [CONTRIBUTING.md](CONTRIBUTING.md) | History; how to set up, test and document |

## Development

    pip3 install -r requirements-test.txt
    pytest -m "not slow"        # unit tests, about 15-20 minutes on a laptop CPU
    pytest -m slow              # TFLite full-integer export checks
    python -m sphinx -b html docs/source docs/build/html

## License

MIT, see [LICENSE](LICENSE).
