# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import pdb

import torch
from torch import nn

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Baseline_multiview_APAN(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            heads,
            view_heads,
            local_heads,
            pixel_mean,
            pixel_std,
            use_secap=True,
            loss_kwargs=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone
        self.backbone = backbone

        # head
        self.heads = heads
        self.view_heads = view_heads
        self.local_heads = local_heads
        self.loss_kwargs = loss_kwargs

        self.use_secap = use_secap

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

        self.gem = nn.Sequential()  # 空容器也行
        self.gem.sigweight = nn.Parameter(torch.tensor(0.1))

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)
        
        cfg0 = cfg.clone()
        if cfg0.is_frozen(): cfg0.defrost()
        cfg0.MODEL.HEADS.NUM_CLASSES = 2
        view_heads = build_heads(cfg0)
        cfg0 = cfg.clone()

        local_heads = build_heads(cfg)

        return {
            'backbone': backbone,
            'heads': heads,
            'view_heads': view_heads,
            'local_heads': local_heads,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'use_secap': cfg.MODEL.USE_SECAP,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE,
                        'view_id': cfg.MODEL.LOSSES.CE.VIEW_ID,
                        'view_oreg': cfg.MODEL.LOSSES.CE.VIEW_OREG,
                        'view_lambda': cfg.MODEL.LOSSES.CE.VIEW_LAMBDA,
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    },
                    'adm':{
                        'beta': cfg.MODEL.LOSSES.ADM.BETA,
                        'alpha': cfg.MODEL.LOSSES.ADM.ALPHA,   
                    }
                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        # print(batched_inputs.keys())

        #摄像头ID
        if 'camids' in batched_inputs.keys():
            camids = batched_inputs['camids']
        else:
            camids = 0
        if 'viewids' in batched_inputs.keys():
            view = batched_inputs['viewids']
            view1_index = [index for index, content in enumerate(view) if content == 'Aerial']
            view2_index = [index for index, content in enumerate(view) if content == 'Ground']

        global_feats, local_feats, view_feats= self.backbone(images, camids)
        
        features = global_feats - view_feats
        # features = global_feats 
        

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            #视角标签
            temp = torch.zeros((targets.shape[0])).long().to(targets.device)
            #创建一个与当前 batch 中样本数量相同的一维张量，初始值全为 0。
            temp[view1_index] = 1#空中视角设置为1
            targets_view = temp

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()


            #对比VDT,训练和推理阶段都多了local的分类头
            #多分类头,不同的分类头训练不同的特征子空间
            outputs = self.heads(features, targets)#主分类头
            outputs_global = self.heads(global_feats, targets)#原始全局特征分类头
            local_outputs = self.local_heads(local_feats, targets)#局部特征分类头，通过更细粒度的特征来增强区分性
            view_outputs = self.view_heads(view_feats, targets_view)#区分不同视觉的分类头，帮助模型学习到区分不同视角特征的能力。

            losses = self.losses(outputs, outputs_global, local_outputs, view_outputs, targets, targets_view)
            return losses
        else:
            outputs = self.heads(features)
            local_outputs = self.local_heads(local_feats)
            # print("测试阶段")
            # print("outputs shape:", outputs.shape)
            # print("local_outputs shape:", local_outputs.shape)
            if self.use_secap:
                outputs = torch.cat((outputs, local_outputs), dim=1)
            return outputs

    #对输入图片进行归一化处理
    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):#若为字典
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):#若为tensor,则直接使用
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, outputs_global, outputs_local, outputs_view, gt_labels, view_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()#提取预测类别的logits（未经softmax）。
        cls_outputs = outputs['cls_outputs']#提取分类输出结果（通常是经过softmax的概率分布）
        pred_features = outputs['features']#提取特征向量
        
        view_pred_class_logits = outputs_view['pred_class_logits'].detach()
        view_cls_outputs = outputs_view['cls_outputs']
        view_pred_features = outputs_view['features']
        
        global_pred_class_logits = outputs_global['pred_class_logits'].detach()
        global_cls_outputs = outputs_global['cls_outputs']
        global_pred_features = outputs_global['features']

        local_pred_class_logits = outputs_local['pred_class_logits'].detach()
        local_cls_outputs = outputs_local['cls_outputs']
        local_pred_features = outputs_local['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)
        #这行代码记录了当前模型对分类任务的准确度，pred_class_logits 是模型的预测结果，gt_labels 是实际标签。也就是target

        loss_dict = {}#存储所有损失
        loss_names = self.loss_kwargs['loss_names']

        #获取视角相关配置参数
        view_kwargs = self.loss_kwargs.get('ce')
        view_id_flag = view_kwargs.get('view_id')#是否使用视角损失
        view_oreg_flag = view_kwargs.get('view_oreg')#是否使用视角正交损失
        view_lambda = view_kwargs.get('view_lambda')#视角损失权重参数

        if 'ADMLoss' in loss_names:
                admloss_kwarge=self.loss_kwargs.get('adm')
                loss_dict['loss_adm_id'] = admloss(
                    cls_outputs,
                    gt_labels,
                    admloss_kwarge.get('beta'),
                     admloss_kwarge.get('alpha'),

                ) 
                if self.use_secap:
                    loss_dict['loss_adm_proto'] = admloss(
                        local_cls_outputs,
                        gt_labels,
                        admloss_kwarge.get('beta'),
                        admloss_kwarge.get('alpha'),
         
                    ) 
                if view_id_flag:
                    loss_dict['loss_adm_view'] = admloss(
                        view_cls_outputs,
                        view_labels,
                        admloss_kwarge.get('beta'),
                         admloss_kwarge.get('alpha'),
                    ) * view_lambda

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet_id'] = triplet_loss(
                pred_features,#特征
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')
            if self.use_secap:
                loss_dict['loss_triplet_proto'] = triplet_loss(
                    local_pred_features,
                    gt_labels,
                    tri_kwargs.get('margin'),
                    tri_kwargs.get('norm_feat'),
                    tri_kwargs.get('hard_mining')
                ) * tri_kwargs.get('scale')

        # calc oreg loss part
        #视角正交损失
        if view_oreg_flag:
            loss_dict['loss_oreg'] = torch.cosine_similarity(pred_features, view_pred_features).abs().mean() * view_lambda

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')

        return loss_dict
