"""
Loss function for FOTS model is consist of two types of loss: Detection loss and Recognition loss.
Detection loss is also consist of per pixel classification loss + bounding box regression loss.
Recognition loss is standard CTC loss for CRNN. This module defines all of these losses.
"""

import torch
from torch import nn
from torch.nn import CTCLoss
from torch.nn import functional as F


class DetectionLoss(nn.Module):
    """Definition of Detection Loss."""
    
    def forward(self, y_true_clf, y_pred_clf, y_true_reg, y_pred_reg):
        """
        Detection loss consists of classification loss and regression loss.

        y_true_clf and y_pred_clf are of shape 1 * img_size/4 * img_size/4 (pred of score map)
        y_true_reg and y_pred_reg are of shape 1 * img_size/4 * img_size/4 (pred of goe map)
        """

        # Classification loss
        clf_loss = self._cross_entropy_loss(y_true_clf, y_pred_clf)

        # Regression loss
        
        # 1. IoU loss
        # split the regression map by channel. Each channel has shape 1 * img_size/4 * img_size/4
        top_gt, right_gt, bottom_gt, left_gt, theta_gt = torch.split(
            y_true_reg, split_size_or_sections=1, dim=1
        )
        top_pred, right_pred, bottom_pred, left_pred, theta_pred = torch.split(
            y_pred_reg, split_size_or_sections=1, dim=1
        )

        # Per pixel area calculation for corresponding bbox
        # sum of left + right will give width and sum of top + bottom will give height for each pixel in bbox
        # and multiplication of height and width is area 
        area_gt = (top_gt + bottom_gt) * (right_gt + left_gt)
        area_pred = (top_pred + bottom_pred) * (right_pred + left_pred)

        # Now calc. area of intersection height and width and then area of intersection
        w_int = torch.min(right_gt, right_pred) + torch.min(left_gt, left_pred)
        h_int = torch.min(top_gt, top_pred) + torch.min(bottom_gt, bottom_pred)

        # Area of intersection between gt and prediction
        area_int = w_int * h_int

        # From simple set theory
        area_union = area_gt + area_pred - area_int

        iou_loss = -torch.log((area_int+1) / (1+area_union))  # +1 is to prevent 0 as log(0) = -inf.
        angle_loss = 1 - torch.cos(theta_pred - theta_gt)

        # Regression loss. It consists of IoU loss + bbox rotation angle loss
        lam_theta = 10  # from paper TODO: Make this configurable
        print(f'iou: {torch.isinf(iou_loss).sum()}, angle: {torch.isinf(angle_loss).sum()}, clf: {clf_loss}')
        regression_loss = iou_loss + lam_theta*angle_loss

        # For regression loss, only consider the loss for the pixels where the ground truth
        # bboxes are present.
        regression_loss = torch.mean(regression_loss * y_true_clf)

        # Merge the reg loss and clf loss using hyperparameter lambda reg. which
        # keeps balance between two losses
        lam_reg = 1  # Value is from paper TODO: Make this configurable
        detection_loss = clf_loss + lam_reg * regression_loss

        return detection_loss
    
    def _cross_entropy_loss(self, y_true_clf, y_pred_clf):
        """
        Calculates cross entropy loss between per pixel prediction score map
        and ground truths.
        """
        return F.binary_cross_entropy(y_pred_clf, y_true_clf)


class RecognitionLoss(nn.Module):
    """
    Definition of Recognition Loss.

    In FOTS, recognition loss is nothing by CTC loss. Implementation of CTC is
    provided by PyTorch OOB.
    Here are some references which explains CTC decoder/loss in detail.

    - https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c
    - https://distill.pub/2017/ctc/
    - https://stats.stackexchange.com/q/320868/176418
    - https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
    """

    def __init__(self):
        super().__init__()
        self.ctc_loss = CTCLoss(zero_infinity=True)
    
    def forward(self, *x):
        gt, pred = x[0], x[1]
        # ctc_loss expects preds, gt, length of preds seq, length of targets seq.
        return self.ctc_loss(pred[0], gt[0], pred[1], gt[1])


class FOTSLoss(nn.Module):
    """
    Definition of FOTS Loss, which is combination of recognition loss and
    detection loss.
    """
    
    def __init__(self):
        super().__init__()
        self.rec_loss = RecognitionLoss()
        self.det_loss = DetectionLoss()
    
    def forward(
        self,
        y_true_clf,
        y_pred_clf,
        y_true_reg,
        y_pred_reg,
        y_true_recog,
        y_pred_recog
    ):
        detection_loss = self.det_loss(y_true_clf, y_pred_clf, y_true_reg, y_pred_reg)

        # Comment following line for full training
        return detection_loss

        # Calculate only if something was supposed to recognized
        recognition_loss = 0
        if y_true_recog:
            recognition_loss = self.rec_loss(y_true_recog, y_pred_recog)
        
        print(f'rec: {recognition_loss}')
        # combine rec. loss and det. loss using lambda recognition.
        lam_recog = 1  # value from paper TODO: Make this configurable
        return detection_loss + lam_recog * recognition_loss
