
"""
MED-VT model.
Some parts of the code is taken from VisTR (https://github.com/Epiphqny/VisTR)
which was again Modified from DETR (https://github.com/facebookresearch/detr)
And modified as needed.
"""
from typing import Optional, List, OrderedDict, Dict
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torchvision

from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict
from mmaction.utils import get_root_logger
from mmcv.cnn import ConvModule

import numpy as np
import torch
from torch import nn
from einops import rearrange

from ..builder import RECOGNIZERS
from .base import BaseRecognizer
from mmaction.models.backbones.resnet3d import BasicBlock3d
import time

class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(32, out_channels)
        self.silu =  nn.SiLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, out_channels)
            )

    def forward(self, x):
        out = self.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.silu(out)
        return out

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 use_conv=True,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = channels//2
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv3d(
                self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        #x:B C T H W
        assert x.shape[1] == self.channels
        x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                mode='trilinear',align_corners=True)
        if self.use_conv:
            x = self.conv(x)
        return x


class Timeupsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 use_conv=True,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = channels//2
        self.use_conv = use_conv
        self.relu = nn.ReLU(inplace=True)
        if use_conv:
            self.conv = nn.Conv3d(
                self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        #x:B C T H W
        assert x.shape[1] == self.channels
        x = F.interpolate(
                x, (x.shape[2]*2, x.shape[3] * 2, x.shape[4] * 2),
                mode='trilinear',align_corners=True)
        if self.use_conv:
            x = self.conv(x)
            x = self.relu(x)
        return x

@RECOGNIZERS.register_module()
class Swinseg(BaseRecognizer):
    def __init__(self,
                 backbone,
                 num_frames,
                 n_class=1,
                 cls_head=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__(backbone=backbone, cls_head=cls_head, neck=neck,
                         train_cfg=train_cfg, test_cfg=test_cfg, **kwargs)
        self.num_frames = num_frames
        self.upsample_layer=nn.ModuleList()
        self.upsample_layer.append(Upsample(256))
        self.upsample_layer.append(Upsample(512))
        self.upsample_layer.append(Upsample(1024))
        
        self.encoder1 = BasicBlock3d(1024,1024)
        self.encoder2 = BasicBlock3d(512,512)
        self.encoder3 = BasicBlock3d(256,256)
        self.encoder4 = BasicBlock3d(128,128)
        self.encoder5 = BasicBlock3d(128,128)
        self.encoder6 = nn.Sequential(
                    nn.Conv3d(3, 64, 3, 1,1),
                    BasicBlock3d(64,32,downsample = 
                                      nn.Conv3d(64, 32, kernel_size=1, stride=1, padding=0, bias=False))
        )
        
        
        
        self.decoder1 = BasicBlock3d(1024,512,downsample = 
                                      nn.Conv3d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False))
        self.decoder2 = BasicBlock3d(512,256,downsample = 
                                      nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1, bias=False))
        self.decoder3 = BasicBlock3d(256,128,downsample = 
                                      nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1, bias=False))
        self.decoder4 =BasicBlock3d(256,128,downsample = 
                                      nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1, bias=False))
        self.outlayer = BasicBlock3d(64,32,downsample = 
                                      nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1, bias=False))

#         self.reslayer1 = BasicBlock3d(1024,512,downsample = 
#                                       nn.Conv3d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False))
#         self.reslayer2 = BasicBlock3d(512,256,downsample = 
#                                       nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0, bias=False))
#         self.reslayer3 = BasicBlock3d(256,128,downsample = 
#                                       nn.Conv3d(256, 128, kernel_size=1, stride=1, padding=0, bias=False))
#         self.inlayer  = nn.Conv3d(3, 64, 3, 1,1)
#         self.reslayer4 = BasicBlock3d(64,32,downsample = 
#                                       nn.Conv3d(64, 32, kernel_size=1, stride=1, padding=0, bias=False))
#         self.reslayer5 = BasicBlock3d(64,32,downsample = 
#                                       nn.Conv3d(64, 32, kernel_size=1, stride=1, padding=0, bias=False))
#         self.reslayer6 =BasicBlock3d(96,32,downsample = 
#                                       nn.Conv3d(96, 32, kernel_size=1, stride=1, padding=0, bias=False))

        self.upsampler = nn.Sequential(
            Timeupsample(128),
            nn.SiLU(),
            Upsample(64)
#             nn.SiLU()
        )

    def forward(self,imgs,label, return_loss=True):
        if return_loss:
            return self.forward_train(imgs, label)
        else:
            return self.forward_test(imgs)

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    def _do_test(self, imgs):
#         print('_do_test:', imgs.shape)
        all_outs = self._forward_one_samples(imgs)
        # mean_outs = all_outs.mean(0)
        return all_outs

    def _forward_one_samples(self, samples):
        #        samples shape: ([1, 1, 3, 25, 224, 224])
        x = rearrange(samples, 'b n c t h w ->(b n)c t h w')
        _, _, t, _, _ = x.shape
        #         t1 = time.time()
        x_ = self.encoder6(x)
        features = self.backbone(x)
        #         t2 = time.time()
        #         features len :4
        #         features[0].shape:([1, 128, 13, 56, 56])
        batch_size, _, num_frames, _, _ = features[0].shape
        
        x4 = features[4]
        x3 = features[3]
        x2 = features[2]
        x1 = features[1]
        x0 = features[0]
        
        x4 = self.encoder1(x4)
        x4 = self.upsample_layer[2](x4)
            
        x3 = self.encoder2(x3)
        x3 = torch.cat([x3,x4],dim=1)
        x3 = self.decoder1(x3)
        
        x3 = self.upsample_layer[1](x3)
        x2 = self.encoder3(x2)
        x2 = torch.cat([x2,x3],dim=1)
        x2 = self.decoder2(x2)
            
        x2 = self.upsample_layer[0](x2)
        x1 = self.encoder4(x1)
        x1 = torch.cat([x1,x2],dim=1)
        x1 = self.decoder3(x1)
            
        x0 = self.encoder5(x0)
        x0 = torch.cat([x0,x1],dim=1)
        x0 = self.decoder4(x0)

        x0 = self.upsampler(x0)
        if  x0.shape[2] > t:
            x0=x0[:,:,:t,:,:]
            
        x0 = torch.cat([x_,x0],dim=1)   
        x0 = self.outlayer(x0)
            
        outputs_seg_masks = self.cls_head(x0)
        out = outputs_seg_masks
        return out


    def forward_train(self, samples, labels, **kwargs):
        losses = dict()
        all_outs = self._forward_one_samples(samples)
        loss_cls = self.cls_head.loss(all_outs, labels, **kwargs)
        losses.update(loss_cls)
        return losses

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)
