# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import build_backbone, BACKBONE_REGISTRY




from .regnet import build_regnet_backbone, build_effnet_backbone
from .shufflenet import build_shufflenetv2_backbone
from .mobilenet import build_mobilenetv2_backbone
from .mobilenetv3 import build_mobilenetv3_backbone
from .repvgg import build_repvgg_backbone
from .vision_transformer import build_vit_backbone
from .vision_transformer_multiview_onebranch import build_multiview_vit_backbone_onebranch
from .vision_transformer_multiview_APAN import build_multiview_vit_backbone_APAN