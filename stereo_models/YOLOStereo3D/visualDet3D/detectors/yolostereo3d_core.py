import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from visualDet3D.networks.lib.blocks import AnchorFlatten, ConvBnReLU
from visualDet3D.networks.lib.ghost_module import ResGhostModule, GhostModule
from visualDet3D.networks.lib.PSM_cost_volume import PSMCosineModule, CostVolume
from visualDet3D.networks.backbones import resnet, dlanet
from visualDet3D.networks.backbones.dlaup import DLAUp
from visualDet3D.networks.backbones.resnet import BasicBlock
from visualDet3D.networks.lib.look_ground import LookGround
from visualDet3D.networks.backbones.depth_estimator import ASPP
class CostVolumePyramid(nn.Module):
    """Some Information about CostVolumePyramid"""
    def __init__(self, depth_channel_4, depth_channel_8, depth_channel_16):
        super(CostVolumePyramid, self).__init__()
        self.depth_channel_4  = depth_channel_4 # 24
        self.depth_channel_8  = depth_channel_8 # 24
        self.depth_channel_16 = depth_channel_16 # 96

        input_features = depth_channel_4 # 24
        self.four_to_eight = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, 3, ratio=3),
            nn.AvgPool2d(2),
            #nn.Conv2d(3 * input_features, 3 * input_features, 3, padding=1, bias=False),
            #nn.BatchNorm2d(3 * input_features),
            #nn.ReLU(),
            BasicBlock(3 * input_features, 3 * input_features),
        )
        input_features = 3 * input_features + depth_channel_8 # 3 * 24 + 24 = 96
        self.eight_to_sixteen = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, 3, ratio=3),
            nn.AvgPool2d(2),
            BasicBlock(3 * input_features, 3 * input_features),
            #nn.Conv2d(3 * input_features, 3 * input_features, 3, padding=1, bias=False),
            #nn.BatchNorm2d(3 * input_features),
            #nn.ReLU(),
        )
        input_features = 3 * input_features + depth_channel_16 # 3 * 96 + 96 = 384
        self.depth_reason = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, kernel_size=3, ratio=3),
            BasicBlock(3 * input_features, 3 * input_features),
            #nn.Conv2d(3 * input_features, 3 * input_features, 3, padding=1, bias=False),
            #nn.BatchNorm2d(3 * input_features),
            #nn.ReLU(),
        )
        self.output_channel_num = 3 * input_features #1152

        self.depth_output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.output_channel_num, int(self.output_channel_num/2), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num/2)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(int(self.output_channel_num/2), int(self.output_channel_num/4), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num/4)),
            nn.ReLU(),
            nn.Conv2d(int(self.output_channel_num/4), 96, 1),
        )


    def forward(self, psv_volume_4, psv_volume_8, psv_volume_16):
        psv_4_8 = self.four_to_eight(psv_volume_4)
        psv_volume_8 = torch.cat([psv_4_8, psv_volume_8], dim=1)
        psv_8_16 = self.eight_to_sixteen(psv_volume_8)
        psv_volume_16 = torch.cat([psv_8_16, psv_volume_16], dim=1)
        psv_16 = self.depth_reason(psv_volume_16)
        if self.training:
            return psv_16, self.depth_output(psv_16)
        return psv_16, torch.zeros([psv_volume_4.shape[0], 1, psv_volume_4.shape[2], psv_volume_4.shape[3]])

class StereoMerging(nn.Module):
    def __init__(self, base_features):
        super(StereoMerging, self).__init__()
        self.cost_volume_0 = PSMCosineModule(downsample_scale=4, max_disp=96, input_features=base_features) #64
        PSV_depth_0 = self.cost_volume_0.depth_channel

        self.cost_volume_1 = PSMCosineModule(downsample_scale=8, max_disp=192, input_features=base_features * 2) #128
        PSV_depth_1 = self.cost_volume_1.depth_channel

        self.cost_volume_2 = CostVolume(downsample_scale=16, max_disp=192, input_features=base_features * 4, PSM_features=8) #256
        PSV_depth_2 = self.cost_volume_2.output_channel

        self.depth_reasoning = CostVolumePyramid(PSV_depth_0, PSV_depth_1, PSV_depth_2)
        self.final_channel = self.depth_reasoning.output_channel_num + base_features * 4

    def forward(self, left_x, right_x):
        PSVolume_0 = self.cost_volume_0(left_x[0], right_x[0])
        PSVolume_1 = self.cost_volume_1(left_x[1], right_x[1])
        PSVolume_2 = self.cost_volume_2(left_x[2], right_x[2])
        PSV_features, depth_output = self.depth_reasoning(PSVolume_0, PSVolume_1, PSVolume_2) # c = 1152
        features = torch.cat([left_x[2], PSV_features], dim=1) # c = 1152 + 256 = 1408
        return features, depth_output

class YoloStereo3DCore(nn.Module):
    """
        Inference Structure of YoloStereo3D
        Similar to YoloMono3D,
        Left and Right image are fed into the backbone in batch. So they will affect each other with BatchNorm2d.
    """
    def __init__(self, backbone_arguments):
        super(YoloStereo3DCore, self).__init__()
        self.backbone_left = resnet(**backbone_arguments)
        self.backbone_disp = resnet(**backbone_arguments)
        base_features = 256 if backbone_arguments['depth'] > 34 else 64
        self.neck = StereoMerging(base_features)

    def ddc(self, x, depth, dilated=1):
        padding = nn.ReflectionPad2d(dilated)
        pad_depth = padding(depth)
        _, _, h, w = x.size()
        pad_x = padding(x)
        filter = (pad_depth[:, :, dilated: dilated + h, dilated: dilated + w] * pad_x[:, :, dilated: dilated + h, dilated: dilated + w]).clone()
        for i in [-dilated, 0, dilated]:
            for j in [-dilated, 0, dilated]:
                if i != 0 or j != 0:
                    filter += (pad_depth[:, :, dilated + i: dilated + i + h, dilated + j: dilated + j + w] * pad_x[:, :, dilated + i: dilated + i + h, dilated + j: dilated + j + w]).clone()
        return filter / 9

    def generate_pseudo_right_feats(self, ori_left_img_feats, left_depth_feats):
        pseudo_right_feats = []
        for img_feat_l, disp_feat_l in zip(ori_left_img_feats, left_depth_feats):
            pseudo_feat_r = self.ddc(img_feat_l, disp_feat_l)
            pseudo_right_feats.append(pseudo_feat_r)
        return pseudo_right_feats

    def forward(self, images):
        left_images = images[:, 0:3, :, :]
        disp_images = images[:, 3:, :, :]
        left_features = self.backbone_left(left_images)
        disp_feats = self.backbone_disp(disp_images)
        pseudo_right_features = self.generate_pseudo_right_feats(left_features, disp_feats)
        features, depth_output = self.neck(left_features, pseudo_right_features)
        output_dict = dict(features=features, depth_output=depth_output)
        return output_dict
