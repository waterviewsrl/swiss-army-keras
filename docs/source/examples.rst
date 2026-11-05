########
Examples
########

Sample notebooks showing the library in complete experiments. They are stored
with their outputs and are not re-executed when the documentation is built.

1. **Flowers classifier.** Classification with the ``distiller_classifier``
   head on a MobileNetV3 (minimalistic) backbone.
2. **DeepLabV3+ on Oxford-IIIT Pets.** Segmentation with the DeepLabV3+
   decoder.
3. **Model user guide.** The U-Net family models inherited from
   keras-unet-collection.
4. **RepViT + PP-LiteSeg water segmentation.** A newer backbone family with a
   lightweight decoder, a loss recipe from ``use_case_recipes.md`` and the
   Lion optimizer, on synthetic data.
5. **RepViT flowers classifier.** Example 1 redone with a RepViT backbone on
   the real ``flower_photos`` dataset.
6. **MobileOne + LR-ASPP on Oxford-IIIT Pets.** Example 2 redone with the
   MobileOne backbone and the LR-ASPP decoder.
7. **MobileNetV4 + PP-LiteSeg on Oxford-IIIT Pets.** The same task with the
   MobileNetV4 backbone and the PP-LiteSeg decoder.

The notebooks that train on real data expect the dataset to be downloaded
first and, for the newer backbone families, a locally converted weights file
(see :ref:`pretrained-weights`).

.. toctree::
   :maxdepth: 1
   :caption: Sample notebooks

   notebooks/flowers-classifier.ipynb
   notebooks/segmentation_deeplab-v3-plus_oxford-iiit.ipynb
   notebooks/user_guide_models.ipynb
   notebooks/repvit_pp-liteseg_water-segmentation.ipynb
   notebooks/classification_repvit_flowers.ipynb
   notebooks/segmentation_mobileone_lr-aspp_oxford-iiit.ipynb
   notebooks/segmentation_mobilenetv4_pp-liteseg_oxford-iiit.ipynb
