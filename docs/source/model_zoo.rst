Model zoo
=========

Everything the library can build, with the exact names to pass and the
limitations to expect. All builders are importable from
``swiss_army_keras.models``.

Backbones
*********

Pass the registry name as ``backbone=`` to any head. The full list at run
time is ``sorted(swiss_army_keras._backbone_zoo.BACKBONE_REGISTRY)``.

.. list-table::
   :header-rows: 1
   :widths: 18 34 12 18 18

   * - Family
     - Registry names
     - Taps (strides)
     - Pretrained weights
     - Structural reparameterization
   * - MobileNet (Keras)
     - ``MobileNetV2``, ``MobileNetV3Large`` (minimalistic variant)
     - 5 (2-32)
     - ``weights='imagenet'``
     - no
   * - ResNet / ResNetV2
     - ``ResNet50``, ``ResNet101``, ``ResNet152``, ``ResNet50V2``,
       ``ResNet101V2``, ``ResNet152V2``
     - 5 (2-32)
     - ``weights='imagenet'``
     - no
   * - DenseNet
     - ``DenseNet121``, ``DenseNet169``, ``DenseNet201``
     - 5 (2-32)
     - ``weights='imagenet'``
     - no
   * - EfficientNet
     - ``EfficientNetB0`` ... ``EfficientNetB7``
     - 5 (2-32)
     - ``weights='imagenet'``
     - no
   * - VGG
     - ``VGG16``, ``VGG19``
     - 5 (1-16)
     - ``weights='imagenet'``
     - no
   * - EfficientNet-Lite
     - ``EfficientNetLiteB0`` ... ``EfficientNetLiteB4`` (needs the
       ``[efficientnet-lite]`` extra, Python <= 3.11);
       ``hub://EfficientNet-lite0`` ... ``hub://EfficientNet-lite4``
       (TensorFlow Hub, fixed 513 x 513 input, not trainable)
     - 5 (2-32)
     - ``weights='imagenet'`` / bundled in the Hub module
     - no
   * - MobileOne
     - ``MobileOne-S0`` ... ``MobileOne-S4``
     - 5 (2-32)
     - local file
     - ``reparameterize_mobileone``
   * - MobileNetV4
     - ``MobileNetV4-Conv-S``, ``-Conv-M``, ``-Conv-L``, ``-Hybrid-M``,
       ``-Hybrid-L``
     - 5 (2-32)
     - local file
     - no
   * - FastViT
     - ``FastViT-T8``, ``-T12``, ``-S12``, ``-SA12``, ``-SA24``, ``-SA36``,
       ``-MA36``
     - 4 (4-32)
     - local file
     - ``reparameterize_fastvit``
   * - RepViT
     - ``RepViT-M0.9``, ``-M1.0``, ``-M1.1``, ``-M1.5``, ``-M2.3``
     - 4 (4-32)
     - local file
     - ``reparameterize_repvit``
   * - StrideFormer
     - ``StrideFormer-Tiny``, ``StrideFormer-Base``
     - 3 (8-32)
     - none available
     - no

The VGG taps are taken before each pooling layer, hence the different
strides.

.. _pretrained-weights:

Pretrained weights for the custom families
------------------------------------------

ImageNet weights for MobileOne, MobileNetV4, FastViT and RepViT are converted
from the ``timm`` PyTorch checkpoints by the scripts in ``scripts/`` and
verified numerically against the PyTorch model. They are **not** downloaded
automatically yet: ``weights='imagenet'`` raises ``NotImplementedError`` for
these families. Convert once, then pass the file path:

.. code-block:: bash

   pip install "swiss-army-keras[porting]"          # torch, timm, safetensors
   python scripts/port_mobileone_weights.py MobileOne-S1
   # -> pretrained_weights/mobileone/MobileOne_S1.weights.h5

.. code-block:: python

   model = lr_aspp(input_tensor, n_labels=3, backbone="MobileOne-S1",
                   weights="pretrained_weights/mobileone/MobileOne_S1.weights.h5")

Check the license of the upstream weights before redistributing converted
files (MobileOne is released under Apple's sample-code license).

Semantic segmentation
*********************

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Builder
     - Encoder
     - Notes
   * - ``deeplab_v3_plus``
     - any registered backbone, or built-in
     - ASPP with dilated convolutions; output at input resolution.
   * - ``deeplab_v3_plus_lite``
     - any registered backbone, or built-in
     - Same without the final upsampling: logits at the shallow tap's
       resolution.
   * - ``lr_aspp``
     - any registered backbone, or built-in
     - The lightest decoder; optional ``iqa_head=True`` side output;
       ``gate=False`` builds it without the sigmoid gate (see
       :doc:`deployment`).
   * - ``pp_liteseg``
     - any registered backbone, or built-in
     - SPPM + UAFM fusion over three taps (``feature_levels``); input of at
       least 128 x 128 and a multiple of 128 with the default bin sizes;
       ``fusion='add'`` replaces the attention fusion by an addition.
   * - ``pp_mobileseg``
     - a 3-tap backbone (designed for StrideFormer)
     - AAM fusion; ``feature_levels=(1, 2, 3)``.
   * - ``ddrnet``
     - own dual-resolution network
     - ``planes=32`` is DDRNet-23-slim, ``64`` is DDRNet-23; with
       ``augment=True`` returns an auxiliary output for deep supervision.
   * - ``pidnet``
     - own three-branch network
     - With ``augment=True`` returns auxiliary and boundary outputs; train
       with ``pidnet_boundary_loss`` / ``pidnet_boundary_attentive_loss``.

All of them return **raw logits**. The U-Net family inherited from
keras-unet-collection (``unet_2d``, ``vnet_2d``, ``unet_plus_2d``,
``r2_unet_2d``, ``att_unet_2d``, ``resunet_a_2d``, ``u2net_2d``,
``unet_3plus_2d``, ``transunet_2d``, ``swin_unet_2d``) is still available;
those models apply their own output activation and are not wrapped for
quantization-aware training.

Object detection and instance segmentation
******************************************

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Builder
     - Notes
   * - ``fcos_lite_detection``
     - Anchor-free FCOS-style head with centerness on a PAFPN-lite neck;
       box decoding and NMS in the graph.
   * - ``gfl_lite_detection``
     - Generalized-focal-loss head (distribution regression, ``reg_max``) on
       a Ghost-PAN neck, in the style of NanoDet-Plus / PP-PicoDet.
   * - ``fcos_lite_instance_segmentation``
     - YOLACT-style prototypes and per-detection mask coefficients on the
       FCOS-lite detector. Batch size 1.

For training, use the ``*_base`` variants in ``swiss_army_keras.base``, which
return the raw per-level head outputs, together with a target assigner
(``atss_assign_targets``, ``simota_assign_targets``, ``dsla_assign_targets``,
``assign_instance_targets`` in ``swiss_army_keras._model_detection``) and the
detection losses. The library does not provide a detection data loader or a
ready-made training loop yet.

Classification, regression and embeddings
*****************************************

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Builder
     - Output
   * - ``classifier``
     - Softmax over ``n_classes`` (``return_logits=True`` for logits);
       optional ``iqa_head=True`` side output.
   * - ``multilabel_classifier``
     - Independent sigmoid per class.
   * - ``ordinal_classifier``
     - CORAL-style ``n_classes - 1`` cumulative sigmoids for ordered labels.
   * - ``wise_srnet_classifier``
     - Classifier with a learned depthwise pooling of the last feature map.
   * - ``distiller_classifier``
     - Classifier on a few "macro-features" distilled from the last feature
       map.
   * - ``learnable_resizer_classifier``
     - ``classifier`` behind a learnable resizer.
   * - ``pose_regression_head``
     - ``n_keypoints`` (x, y) coordinates in [0, 1], single subject.
   * - ``reid_embedding_head``
     - L2-normalized embedding for re-identification / metric learning.
   * - ``lr_aspp_depth``, ``fastdepth_head``
     - Dense monocular depth map (sigmoid-bounded or linear). Both adapt to
       backbones with five, four or three feature levels.
   * - ``multi_task``
     - Several heads on one shared backbone, one named output per head.
       Head types: ``lr_aspp``, ``lr_aspp_depth``, ``fastdepth``,
       ``classifier``, ``iqa``, or a callable.

The classifier heads accept ``pre_pool_attention='gem' | 'eca' | 'coordatt'``
and ``pooling='avg' | 'max'``.

Losses
******

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Group
     - Functions (``swiss_army_keras.losses``)
   * - Region
     - ``dice``, ``tversky``, ``focal_tversky``, ``log_cosh_dice``,
       ``iou_seg``, ``lovasz_softmax``, ``combo_loss``,
       ``symmetrical_unified_focal_loss``
   * - Boundary and structure
     - ``boundary_loss``, ``structure_loss``, ``pidnet_boundary_loss``,
       ``pidnet_boundary_attentive_loss``, ``ms_ssim``
   * - Pixel-wise and hard-example
     - ``focal_categorical_crossentropy``, ``categorical_focal_loss``,
       ``modified_categorical_focal_loss``,
       ``ohem_categorical_crossentropy``, ``class_balanced_weights``
   * - Detection
     - ``iou_box`` (GIoU / DIoU / CIoU / SIoU modes), ``quality_focal_loss``,
       ``distribution_focal_loss``, ``varifocal_loss``, ``yolact_mask_loss``
   * - Depth
     - ``scale_invariant_log_loss``, ``berhu_loss``
   * - Other
     - ``triplet_1d``, ``crps2d_tf``
   * - Wrappers
     - ``loss_adapter`` (logits, label smoothing, resolution mismatch),
       ``deep_supervision_loss`` (multi-output models)

Task-specific combinations are discussed in ``use_case_recipes.md``.

Optimizers and metrics
**********************

* ``swiss_army_keras.optimizers``: ``Yogi``, ``RectifiedAdam2``,
  ``AdaBeliefOptimizer``, ``Lion``, ``AdaHessian`` (custom ``minimize``, not a
  drop-in for ``model.fit``) and the ``SAM`` sharpness-aware wrapper around
  any base optimizer.
* ``swiss_army_keras.metrics``: ``MeanIOU`` (one-hot aware),
  ``BoundaryFScore``, ``MeanAveragePrecision``, and for depth
  ``DepthAbsRel``, ``DepthRMSE`` and ``DepthDeltaAccuracy`` (pixels with a
  target of 0 are left out; ``align='median'`` for depth known up to scale).

Data loaders
************

``swiss_army_keras.dataset_utils``:
``SegmentationAlbumentationsDataLoader`` (image and class mask),
``DepthAlbumentationsDataLoader`` (image and depth map, 16-bit PNG or
``.npy``) and ``ClassificationAlbumentationsDataLoader`` (one directory per
class). All three run albumentations pipelines in a pool of worker
processes.

Auxiliary utilities
*******************

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Utility
     - Purpose
   * - ``layer_utils.uint8_input_wrapper``
     - Moves the cast and input normalization into the graph so the
       deployed model takes raw ``uint8`` frames.
   * - ``layer_utils.add_energy_score_output``, ``layer_utils.energy_score``
     - Adds an energy-based confidence output computed from the logits.
   * - ``layer_utils.add_embedding_output``
     - Adds the feature that feeds a head as an extra output of a trained
       model: the embedding the Mahalanobis statistics are fitted on.
   * - ``ood_utils``
     - Mahalanobis-distance out-of-distribution scores (image level and
       dense), a NumPy-only form for devices (``whiten_statistics``,
       ``whitened_distance``), sampling of dense embeddings, anomaly-map
       aggregation, threshold calibration, and ``ood_detection_metrics``
       (AUROC, false positive rate at 95% true positive rate).
   * - ``iqa_utils``
     - Teacher-labelled image-quality side task
       (``add_teacher_iqa_labels``, ``laplacian_variance_quality``).
   * - ``inference_utils.tta_predict``
     - Flip and multi-scale test-time augmentation for segmentation.
   * - ``deploy_utils``
     - Deployment profiles, ``reparameterize_model`` (fuses the
       training-time branches of a MobileOne / FastViT / RepViT backbone
       inside any model), ``strip_qat``, ``drop_output_activations``,
       ``equalize_cross_layer``, ``convert_to_tflite``, ``lint_tflite``,
       ``tflite_outputs`` and ``quantization_error`` (see
       :doc:`deployment`).
   * - ``reparam_utils`` and ``reparameterize_*``
     - The per-family fusion functions and kernel arithmetic that
       ``deploy_utils.reparameterize_model`` builds on.
