'''Public entry points: every full model builder of the library.
'''
from __future__ import absolute_import

import warnings

from swiss_army_keras._model_unet_2d import unet_2d
from swiss_army_keras._model_vnet_2d import vnet_2d
from swiss_army_keras._model_unet_plus_2d import unet_plus_2d
from swiss_army_keras._model_r2_unet_2d import r2_unet_2d
from swiss_army_keras._model_att_unet_2d import att_unet_2d
from swiss_army_keras._model_resunet_a_2d import resunet_a_2d
from swiss_army_keras._model_u2net_2d import u2net_2d
from swiss_army_keras._model_unet_3plus_2d import unet_3plus_2d
try:
    from swiss_army_keras._model_transunet_2d import transunet_2d
except ImportError:
    warnings.warn('Cannot load _model_transunet_2d')
from swiss_army_keras._model_swin_unet_2d import swin_unet_2d
from swiss_army_keras._model_deeplab_v3_plus import deeplab_v3_plus, deeplab_v3_plus_lite
from swiss_army_keras._model_lr_aspp import lr_aspp
from swiss_army_keras._model_pp_liteseg import pp_liteseg
from swiss_army_keras._model_ddrnet import ddrnet
from swiss_army_keras._model_pidnet import pidnet
from swiss_army_keras._model_pp_mobileseg import pp_mobileseg
from swiss_army_keras._model_detection import fcos_lite_detection, gfl_lite_detection
from swiss_army_keras._model_detection import fcos_lite_instance_segmentation
from swiss_army_keras._model_depth import lr_aspp_depth, fastdepth_head
from swiss_army_keras._model_classifier import classifier, wise_srnet_classifier, distiller_classifier, learnable_resizer_classifier, multilabel_classifier, ordinal_classifier, pose_regression_head, reid_embedding_head
