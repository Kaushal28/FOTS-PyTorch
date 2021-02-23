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

    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def forward(self, y_true_clf, y_pred_clf, y_true_reg, y_pred_reg, training_mask):
        """
        Detection loss consists of classification loss and regression loss.

        y_true_clf and y_pred_clf are of shape 1 * img_size/4 * img_size/4 (pred of score map)
        y_true_reg and y_pred_reg are of shape 1 * img_size/4 * img_size/4 (pred of goe map)
        """

        # Classification loss
        # clf_loss = self._cross_entropy_loss(y_true_clf, y_pred_clf)
        clf_loss = self._dice_coefficient(y_true_clf, y_pred_clf, training_mask) * 0.01

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
        regression_loss = iou_loss + self.config["fots_hyperparameters"]["lam_theta"] * angle_loss

        # For regression loss, only consider the loss for the pixels where the ground truth
        # bboxes are present.
        regression_loss = torch.mean(regression_loss * y_true_clf * training_mask)

        # Merge the reg loss and clf loss using hyperparameter lambda reg. which
        # keeps balance between two losses
        detection_loss = clf_loss + self.config["fots_hyperparameters"]["lam_reg"] * regression_loss

        return detection_loss
    
    def _dice_coefficient(self, y_true_cls, y_pred_cls,
                         training_mask):
        eps = 1e-5
        intersection = torch.sum(y_true_cls * y_pred_cls * training_mask)
        union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls * training_mask) + eps
        loss = 1. - (2 * intersection / union)

        return loss
    
    def _cross_entropy_loss(self, y_true_clf, y_pred_clf, training_mask):
        """
        Calculates cross entropy loss between per pixel prediction score map
        and ground truths.
        """
        return torch.nn.functional.binary_cross_entropy(y_pred_clf*training_mask, (y_true_clf*training_mask))


# class DetectionLoss(nn.Module):
#     def __init__(self):
#         super(DetectionLoss, self).__init__()

#     def forward(self, y_true_cls, y_pred_cls,
#                 y_true_geo, y_pred_geo,
#                 training_mask):
#         classification_loss = self.__dice_coefficient(y_true_cls, y_pred_cls, training_mask)

#         #classification_loss = self.__cross_entroy(y_true_cls, y_pred_cls, training_mask)
#         # scale classification loss to match the iou loss part
#         classification_loss *= 0.01

#         # d1 -> top, d2->right, d3->bottom, d4->left
#         #     d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
#         d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(y_true_geo, 1, 1)
#         #     d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
#         d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(y_pred_geo, 1, 1)
#         area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
#         area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
#         w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
#         h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
#         area_intersect = w_union * h_union
#         area_union = area_gt + area_pred - area_intersect
#         L_AABB = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
#         L_theta = 1 - torch.cos(theta_pred - theta_gt)
#         L_g = L_AABB + 20 * L_theta

#         # return torch.sum(L_g * y_true_cls * training_mask)/ torch.count_nonzero(y_true_cls * training_mask) + classification_loss

#         return torch.mean(L_g * y_true_cls * training_mask) + classification_loss

#     def __dice_coefficient(self, y_true_cls, y_pred_cls,
#                          training_mask):
#         '''
#         dice loss
#         :param y_true_cls:
#         :param y_pred_cls:
#         :param training_mask:
#         :return:
#         '''
#         eps = 1e-5
#         intersection = torch.sum(y_true_cls * y_pred_cls * training_mask)
#         union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls * training_mask) + eps
#         loss = 1. - (2 * intersection / union)

#         return loss


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
    
    def __init__(self, config):
        super().__init__()
        self.rec_loss = RecognitionLoss()
        self.det_loss = DetectionLoss(config)
        self.config = config
    
    def forward(
        self,
        y_true_clf,
        y_pred_clf,
        y_true_reg,
        y_pred_reg,
        training_mask
        # ,y_true_recog,
        # y_pred_recog
    ):
        detection_loss = self.det_loss(y_true_clf, y_pred_clf, y_true_reg, y_pred_reg, training_mask)

        # Comment following line for full training
        return detection_loss

        # Calculate only if something was supposed to recognized
        recognition_loss = 0
        if y_true_recog:
            recognition_loss = self.rec_loss(y_true_recog, y_pred_recog)
        
        print(f'rec: {recognition_loss}')
        # combine rec. loss and det. loss using lambda recognition.
        return detection_loss + config["fots_hyperparameters"]["lam_recog"] * recognition_loss
