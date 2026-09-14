########
Examples
########

You can find here sample notebooks which show the usage of varius feature of the library in a production like experiment and setup.

1. Classification example: a classification experiment based on distiller classifier layer, with MobilenetV3Plus minimalistic feature extractor backbone.
2. Segmentation example: a segmentation experiment based on DeepLabv3+ segmentation decoder with MobilenetV3Plus minimalistic encoder.
3. Model User Guide. A notebook with documentation about the different possible models provided by this library
4. RepViT + PP-LiteSeg water segmentation: one of the new backbone families (RepViT) combined with one of the new lightweight decoders (PP-LiteSeg with UAFM), trained with a loss recipe from ``use_case_recipes.md`` and the new Lion optimizer, on synthetic data.
5. RepViT flowers classifier: a re-do of the classification example above with the RepViT backbone in place of MobileNetV3, trained on the real ``flower_photos`` dataset.
6. MobileOne + LR-ASPP pet segmentation: a re-do of the segmentation example above with the MobileOne backbone and the LR-ASPP decoder in place of MobileNetV3Plus/DeepLabv3+, trained on the real Oxford-IIIT Pets dataset.


.. toctree::
   :maxdepth: 1
   :caption: Sample Notebooks:

   notebooks/flowers-classifier.ipynb
   notebooks/segmentation_deeplab-v3-plus_oxford-iiit.ipynb
   notebooks/user_guide_models.ipynb
   notebooks/repvit_pp-liteseg_water-segmentation.ipynb
   notebooks/classification_repvit_flowers.ipynb
   notebooks/segmentation_mobileone_lr-aspp_oxford-iiit.ipynb
   


