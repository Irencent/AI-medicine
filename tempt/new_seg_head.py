import torch.nn as nn
from ..builder import HEADS
from .base import BaseHead
import torch
import numpy as np
from mmcv.cnn import normal_init,kaiming_init
from einops import rearrange
import torch.nn.functional as F

def calculate_miou(pred, label, num_classes):
    """
    """
    pred_softmax = F.softmax(pred, dim=-1)
    pred_label = torch.argmax(pred_softmax, dim=-1)
    iou_list = []
    for class_id in range(num_classes):
        true_class = label == class_id
        pred_class = pred_label == class_id

        intersection = torch.logical_and(true_class, pred_class)
        union = torch.logical_or(true_class, pred_class)

        union_sum = torch.sum(union, dim=[1, 2, 3])
        intersection_sum = torch.sum(intersection, dim=[1, 2, 3])

        iou = torch.where(union_sum > 0, intersection_sum / union_sum, torch.zeros_like(union_sum).float())
        iou_list.append(iou)

    miou = torch.mean(torch.stack(iou_list), dim=0)
    
    return miou

def calculate_iou(pred, label):
    pred = torch.nn.functional.softmax(pred, dim=-1)
    pred = torch.argmax(pred, dim=-1)
    predt = pred.long()
    labl = label.long()
    labl = labl.reshape(pred.shape)
    intersection = (predt & labl).sum((2, 3))
    union = (predt | labl).sum((2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    mean_iou = iou.mean()

    return mean_iou


@HEADS.register_module()
class SegHead(BaseHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='diceloss'),
                 spatial_type='avg',
                 dropout_ratio=0,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls)

        self.layer = nn.Sequential(
            nn.Conv3d(in_channels,16, 3, padding=(1, 1, 1), dilation=1),
            nn.GroupNorm(8, 16),
            nn.SiLU(),
            nn.Conv3d(16 ,num_classes, 3, padding=(1, 1, 1), dilation=1))
#             nn.GroupNorm(8, 16),
#             nn.ReLU(),
#             nn.Conv3d(16, 8, 3, padding=(1, 1, 1), dilation=1),
#             nn.GroupNorm(4, 8),
#             nn.ReLU(),
#             nn.Conv3d(16, 2, 3, padding=(1, 1, 1), dilation=1))
#             nn.ReLU())

        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.fc_cls = num_classes
        
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)
        kaiming_init(self.layer)
    #        normal_init(self.fc_code, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        x = self.layer(x)
        if self.dropout is not None:
            x = self.dropout(x)
        outputs_seg_masks = x
        outputs_seg_masks = rearrange(outputs_seg_masks, 'b n t h w->b t h w n')
        return outputs_seg_masks

    def loss(self, cls_score, labels, **kwargs):
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
#             print(labels.shape)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            labels = labels.unsqueeze(0)
#             print(labels.shape)
        loss_cls = self.loss_cls(cls_score, labels, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
#         losses['IoU'] = calculate_iou(cls_score, labels)
        losses['mIoU'] = calculate_miou(cls_score, labels,self.fc_cls)
        return losses
