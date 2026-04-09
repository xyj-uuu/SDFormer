# Copyright (c) Phigent Robotics. All rights reserved.
import math
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS
from mmdet3d.ops.bev_pool import bev_pool
from mmdet3d.ops.voxel_pooling import voxel_pooling
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import force_fp32
from torch.cuda.amp.autocast_mode import autocast
from projects.mmdet3d_plugin.utils.gaussian import generate_guassian_depth_target
from mmdet.models.backbones.resnet import BasicBlock
from projects.mmdet3d_plugin.utils.semkitti import semantic_kitti_class_frequencies, kitti_class_names, CE_ssc_loss
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from PIL import Image
import pdb
import sys
from .ViewTransformerLSSBEVDepth import *
from .semkitti_depthnet import SemKITTIDepthNet
from .temporal_retrieve  import *
norm_cfg = dict(type='GN', num_groups=2, requires_grad=True)
from .gwc_encoder import *
sys.path.append('projects/mmdet3d_plugin/occupancy/image2bev/')
from .LEAStereo.LEAStereo import LEA_encoder
from .manydepth.temporal_encoder import  temporal_encoder

class BEVGeomAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(BEVGeomAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, bev_prob):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x1)
        return self.sigmoid(x1+bev_prob)

class ChannelAttention(nn.Module):

    def __init__(self, input_channel, output_channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(input_channel, input_channel // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(input_channel // ratio, output_channel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResCBAMBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResCBAMBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes, planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class ProbNet(BaseModule):

    def __init__(
        self,
        in_channels=512,
        scale_factor=1,
        with_centerness=False,
        loss_weight=6.0,
        bev_size=None,
    ):
        super(ProbNet, self).__init__()
        self.loss_weight=loss_weight
        mid_channels=in_channels//2
        self.base_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.prob_conv = nn.Sequential(
            ResCBAMBlock(mid_channels, mid_channels),
            )
        self.mask_net = nn.Conv2d(mid_channels, 1, kernel_size=1, padding=0, stride=1)

        self.with_centerness=with_centerness
        # if with_centerness:
        #     self.centerness = bev_centerness_weight(bev_size[0],bev_size[1]).cuda()
        # self.dice_loss = DiceLoss(use_sigmoid=True, loss_weight=self.loss_weight)
        self.ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.13]))

    def forward(self, input):
        height_feat = self.base_conv(input)
        height_feat = self.prob_conv(height_feat)
        bev_prob = self.mask_net(height_feat)
        return bev_prob
# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(256, 64, kernel_size=1)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
#         self.relu2 = nn.ReLU()
#         self.conv3 = nn.Conv2d(32, 3, kernel_size=1)
#         self.relu3 = nn.ReLU()
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.relu1(out)
#         out = self.conv2(out)
#         out = self.relu2(out)
#         out = self.conv3(out)
#         out = self.relu3(out)
#         return out
class ConvNetModel(nn.Module):
    def __init__(self):
        super(ConvNetModel, self).__init__()
        self.conv1 = nn.Conv2d(256, 3, kernel_size=(12, 20), stride=4, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        return out

class attention3(nn.Module):

    def __init__(self, in_dim):
        super(attention3, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim  , kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, kv):
        q = q.unsqueeze(dim=1)
        kv = kv.unsqueeze(dim=1)
        x=kv
        m_batchsize, C, D, height, width = x.size()

        confidence  = F.softmax(q, dim=2)
        confidence = torch.max(confidence, dim=2)[0]
        confidence =confidence.view(m_batchsize, -1, width * height)


        proj_query = self.query_conv(q ).view(m_batchsize, -1, width * height).permute(0, 2, 1)

        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)

        attention = self.softmax(energy)

        attention = confidence*attention


        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        out = out.view(m_batchsize, C, D, height, width)

        out = self.gamma * out + x
        return out

class FusionModule(nn.Module):
    def __init__(self, in_channels=128):
        super(FusionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=7, padding=3)
        self.fc1 = nn.Linear(in_channels, in_channels)
        self.fc2 = nn.Linear(in_channels, in_channels)
        self.gelu = nn.GELU()

    def forward(self, x1, x2):
        # Element-wise sum of the two input tensors
        x = x1 + x2

        # Max Pooling
        max_pool = torch.max(x1, x2)

        # Mean Pooling
        mean_pool = (x1 + x2) / 2

        # Concatenate max_pool and mean_pool along the channel dimension
        cat_pool = torch.cat((max_pool, mean_pool), dim=1)

        # Apply 7x7 convolution
        conv_out = self.conv(cat_pool)

        # Element-wise multiplication and addition
        combined = x * conv_out + x

        # Flatten the tensor
        combined_flat = combined.view(combined.size(0), combined.size(1), -1).mean(dim=2)

        # Linear layers with GELU activation
        out = self.fc1(combined_flat)
        out = self.gelu(out)
        out = self.fc2(out)

        # Reshape the output back to the original spatial dimensions
        out = out.view(combined.size(0), combined.size(1), 1, 1) * combined

        return out

class volume_interaction(nn.Module):
    def __init__(self,  out_channels=1):
        super(volume_interaction, self).__init__()
        self.dres1 = nn.Sequential(convbn_3d(2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   hourglass(32))
        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.out3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(32, 1, 3, 1, 1))
    def forward(self, stereo_volume, lss_volume):
        stereo_volume=stereo_volume.unsqueeze(1)
        lss_volume=lss_volume.unsqueeze(1)
        all_volume=torch.cat( (stereo_volume, lss_volume ), dim=1)
        data1_ = self.dres1(all_volume)
        data2_ = self.dres2(data1_)
        data3 = self.dres3(data2_) + data1_
        data3 =  self.out3(data3)
        data3 = data3.squeeze(1)
        data3 = F.softmax(data3, dim=1)
        data1, data2=None, None
        return data3, [data1, data2]


class temporal_interaction(nn.Module):
    def __init__(self,  out_channels=1):
        super(volume_interaction, self).__init__()
        self.dres1 = nn.Sequential(convbn_3d(2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   hourglass(32))
        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.out3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(32, 1, 3, 1, 1))
    def forward(self, stereo_volume, lss_volume):
        stereo_volume=stereo_volume.unsqueeze(1)
        lss_volume=lss_volume.unsqueeze(1)
        all_volume=torch.cat( (stereo_volume, lss_volume ), dim=1)
        data1_ = self.dres1(all_volume)
        data2_ = self.dres2(data1_)
        data3 = self.dres3(data2_) + data1_
        data3 =  self.out3(data3)
        data3 = data3.squeeze(1)
        data3 = F.softmax(data3, dim=1)
        data1, data2=None, None
        return data3, [data1, data2]



@NECKS.register_module()
class ViewTransformerLiftSplatShootVoxel(ViewTransformerLSSBEVDepth):
    def __init__(
            self,
            loss_depth_weight,
            semkitti=False,
            imgseg=False,
            imgseg_class=20,
            lift_with_imgseg=False,
            point_cloud_range=None,
            loss_seg_weight=1.0,
            loss_depth_type='bce', ##'bce', smooth
            point_xyz_channel=0,
            point_xyz_mode='cat',
            depth_model="lea",
            temporal_num = 2,
            **kwargs,
        ):

        super(ViewTransformerLiftSplatShootVoxel, self).__init__(loss_depth_weight=loss_depth_weight, **kwargs)

        self.leamodel=LEA_encoder(maxdisp=192)
        self.volume_interaction = volume_interaction()
        self.temporal_deformable = multipatch_deformable( indim=3, outdim=3 )
        self.curr_patch = multi_patch2d(in_channel=64, depth=1)
        self.warped_patch = multi_patch3d(in_channel=3, depth=1)
        self.cossim = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.temporal_encoder = temporal_encoder( maxdisp=112, height=384, width=1280 )
        self.temporal_prehourglass = nn.Sequential(convbn_3d( temporal_num-1, 32, 3, 1, 1),  nn.ReLU(inplace=True),
                                                hourglass(32),
                                                convbn_3d(32, 64, 3, 1, 1),
                                                nn.ReLU(inplace=True),
                                                nn.Conv3d(64, 64, kernel_size=3, padding=1, stride=1, bias=False))

        self.temporal_hourglass = nn.Sequential(convbn_3d( 64, 32, 3, 1, 1),  nn.ReLU(inplace=True),
                                                hourglass(32),
                                                convbn_3d(32, 64, 3, 1, 1),
                                                nn.ReLU(inplace=True),
                                                nn.Conv3d(64, 384, kernel_size=3, padding=1, stride=1, bias=False))



        self.semkitti = semkitti

        self.loss_depth_type = loss_depth_type
        self.cam_depth_range = self.grid_config['dbound']
        self.constant_std = 0.5
        self.point_cloud_range = point_cloud_range

        ''' Extra input for Splating: except for the image features, the lifted points should also contain their positional information '''
        self.point_xyz_mode = point_xyz_mode
        self.point_xyz_channel = point_xyz_channel

        assert self.point_xyz_mode in ['cat', 'add']
        if self.point_xyz_mode == 'add':
            self.point_xyz_channel = self.numC_Trans

        if self.point_xyz_channel > 0:
            assert self.point_cloud_range is not None
            self.point_cloud_range = torch.tensor(self.point_cloud_range)

            mid_channel = self.point_xyz_channel // 2
            self.point_xyz_encoder = nn.Sequential(
                nn.Linear(in_features=3, out_features=mid_channel),
                nn.BatchNorm1d(mid_channel),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=mid_channel, out_features=self.point_xyz_channel),
            )


        ''' Auxiliary task: image-view segmentation '''
        self.imgseg = imgseg
        if self.imgseg:
            self.imgseg_class = imgseg_class
            self.loss_seg_weight = loss_seg_weight
            self.lift_with_imgseg = lift_with_imgseg

            # build a small segmentation head
            in_channels = self.numC_input
            self.img_seg_head = nn.Sequential(
                BasicBlock(in_channels, in_channels),
                BasicBlock(in_channels, in_channels),
                nn.Conv2d(in_channels, self.imgseg_class, kernel_size=1, padding=0),
            )

        self.forward_dic = {}
        self.up = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1),
        )
        self.fusion2 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        )
        self.fusionmodule = FusionModule()
        self.fusion1 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 112, kernel_size=3, stride=1, padding=1),
        )
        self.attention1 = attention3(in_dim=1)
        self.conv_feat = ConvNetModel()
        # self.prob = ProbNet(in_channels=112, with_centerness=True, bev_size=(48, 160))
        # self.sigmoid = nn.Sigmoid()
        # self.geom_att = BEVGeomAttention()

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape  ## [1, 1, 384, 1280]
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample) #
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values  #
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)

        # [min - step / 2, min + step / 2] creates min depth
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]
        gt_depths_vals = gt_depths.clone()

        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]

        return gt_depths_vals, gt_depths.float()

    def get_diff_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)

        # [min - step / 2, min + step / 2] creates min depth
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]
        gt_depths_vals = gt_depths.clone()

        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1)[:, :, :, 1:]

        mask = torch.max(gt_depths, dim=3).values > 0.0
        mask = mask.unsqueeze(3)
        gt_depths = gt_depths * mask

        gt_depths = gt_depths.permute(0, 3, 1, 2).contiguous()
        gt_depths = gt_depths.unsqueeze(1)


        return gt_depths.float()

    @force_fp32()
    def get_bce_depth_loss(self, depth_labels, depth_preds):
        _, depth_labels = self.get_downsampled_gt_depth(depth_labels)
        # depth_labels = self._prepare_depth_gt(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(depth_preds, depth_labels, reduction='none').sum() / max(1.0, fg_mask.sum())
        return depth_loss

    @force_fp32()
    def get_smooth_depth_loss(self, depth_labels, depth_preds):

        B,D,H,W = depth_preds.shape
        depth_labels =  F.interpolate(depth_labels, [ H, W], mode='bilinear', align_corners=False)

        with torch.cuda.device_of(depth_preds):
            disp = torch.reshape(torch.arange(0, D, device=torch.cuda.current_device(), dtype=torch.float32),[1,D,1,1])
            disp = disp.repeat(depth_preds.size()[0], 1, depth_preds.size()[2], depth_preds.size()[3])
            depth_preds = torch.sum(depth_preds * disp, 1).unsqueeze(1)

        mask = (depth_labels > 0)
        mask.detach_()
        loss = F.smooth_l1_loss(depth_preds[mask], depth_labels[mask], reduction='mean')
        return loss


    @force_fp32()
    def get_klv_depth_loss(self, depth_labels, depth_preds):
        depth_gaussian_labels, depth_values = generate_guassian_depth_target(depth_labels,
            self.downsample, self.cam_depth_range, constant_std=self.constant_std)

        depth_values = depth_values.view(-1)
        fg_mask = (depth_values >= self.cam_depth_range[0]) & (depth_values <= (self.cam_depth_range[1] - self.cam_depth_range[2]))

        depth_gaussian_labels = depth_gaussian_labels.view(-1, self.D)[fg_mask]
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)[fg_mask]

        depth_loss = F.kl_div(torch.log(depth_preds + 1e-4), depth_gaussian_labels, reduction='batchmean', log_target=False)

        return depth_loss

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        if self.loss_depth_type == 'bce':
            depth_loss = self.get_bce_depth_loss(depth_labels, depth_preds)

        elif self.loss_depth_type == 'kld':
            depth_loss = self.get_klv_depth_loss(depth_labels, depth_preds)

        elif self.loss_depth_type == 'smooth':
            depth_loss = self.get_smooth_depth_loss(depth_labels, depth_preds)

        else:
            pdb.set_trace()

        return self.loss_depth_weight * depth_loss

    @force_fp32()
    def get_seg_loss(self, seg_labels):
        class_weights = torch.from_numpy(1 / np.log(semantic_kitti_class_frequencies + 0.001)).type_as(seg_labels).float()
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=0, reduction="mean",
        )
        seg_preds = self.forward_dic['imgseg_logits']
        if seg_preds.shape[-2:] != seg_labels.shape[-2:]:
            seg_preds = F.interpolate(seg_preds, size=seg_labels.shape[1:])

        loss_seg = criterion(seg_preds, seg_labels.long())

        return self.loss_seg_weight * loss_seg

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_xyz = geom_feats.clone()
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        if self.point_xyz_channel > 0:
            geom_xyz = geom_xyz.view(Nprime, 3)
            geom_xyz = geom_xyz[kept]

            pc_range = self.point_cloud_range.type_as(geom_xyz) # normalize points to [-1, 1]
            geom_xyz = (geom_xyz - pc_range[:3]) / (pc_range[3:] - pc_range[:3])
            geom_xyz = (geom_xyz - 0.5) * 2
            geom_xyz_feats = self.point_xyz_encoder(geom_xyz)

            if self.point_xyz_mode == 'cat':
                # concatenate image features & geometric features
                x = torch.cat((x, geom_xyz_feats), dim=1)

            elif self.point_xyz_mode == 'add':
                x += geom_xyz_feats

            else:
                raise NotImplementedError

        final = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
        final = final.permute(0, 1, 3, 4, 2)

        return final

    def forward(self, input, gt, mode, imgl, imgr, left_input, right_input,sam_list,mask_list):

        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]

        ##########
        mask_seg = torch.tensor(mask_list)
        
        if self.training:
            seg_fea = torch.tensor([sam_list[0].cpu().numpy(), sam_list[1].cpu().numpy()]).squeeze(1)
            
        else:
            seg_fea = torch.tensor([sam_list[0].cpu().numpy()]).squeeze(1)

        mask_seg = self.up(F.softmax(mask_seg.unsqueeze(1).type(torch.cuda.FloatTensor), dim=1).squeeze(1))
        mask_seg = F.interpolate(mask_seg, (48, 160))
        pred_seg = torch.mul(seg_fea.type(torch.cuda.FloatTensor),
                             mask_seg.type(torch.cuda.FloatTensor).unsqueeze(0)).squeeze(0)

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        calib = input[16]

        if  imgl.shape[1]>1:
            imgl, imgr = imgl[:, -1, ...], imgr[:, -1, ...]
        imgl, imgr = F.interpolate(imgl.squeeze(1), size=[288, 960], mode='bilinear', align_corners=True), F.interpolate(imgr.squeeze(1), size=[288, 960], mode='bilinear', align_corners=True)
        stereo_volume = self.leamodel(imgl, imgr, calib )["classfy_volume"]
        stereo_volume = F.interpolate(stereo_volume, size=[ 112, H, W ], mode='trilinear', align_corners=True).squeeze(1)
        stereo_volume = F.softmax(-stereo_volume, dim=1)


        if self.imgseg:
            self.forward_dic['imgseg_logits'] = self.img_seg_head(x)

        x = self.depth_net(x, mlp_input)
        depth_digit = x[:, :self.D, ...]
        img_feat = x[:, self.D:self.D + self.numC_Trans, ...]
        depth_prob = self.get_depth_dist(depth_digit)

        img_feat = self.fusionmodule(self.fusion2(pred_seg), img_feat)
        depth_prob1 = self.attention1(depth_prob, self.fusion1(img_feat)).squeeze(1)
        depth_prob, auxility = self.volume_interaction(stereo_volume, depth_prob1)


        if self.imgseg and self.lift_with_imgseg:
            img_segprob = torch.softmax(self.forward_dic['imgseg_logits'], dim=1)
            img_feat = torch.cat((img_feat, img_segprob), dim=1)

        # Lift
        volume = depth_prob.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, -1, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans, bda)
        bev_feat = self.voxel_pooling(geom, volume)

        seg_fea1 = self.conv_feat(seg_fea.cuda())
        seg_fea1 = F.interpolate(seg_fea1, size=[384, 1280], mode='bilinear', align_corners=True)
        
        curr_feature, batch_waped_feature = self.temporal_encoder(ref_images=left_input.unsqueeze(1).permute(0, 1, 4, 2, 3).cuda(),
                              source_images=seg_fea1.unsqueeze(1).cuda(), intrinsics=intrins)

        curr_feature = F.interpolate(curr_feature, size=[H, W], mode='bilinear', align_corners=True)
        batch_waped_feature = F.interpolate(batch_waped_feature, size=[self.D, H, W], mode='trilinear', align_corners=True)

        t_volume = self.temporal_prehourglass( batch_waped_feature )

        t_volume = t_volume.view(B, N, -1, self.D, H, W)
        t_volume = t_volume.permute(0, 1, 3, 4, 5, 2)
        t_volume = self.voxel_pooling(geom, t_volume)
        t_volume =  self.temporal_hourglass(t_volume )
        t_voxel = [t_volume] 


        return bev_feat, depth_prob, t_voxel