import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from opencood.loss.center_point_loss import CenterPointLoss
from icecream import ic


class CenterPointCodebookLoss(CenterPointLoss):
    def __init__(self, args):
        super(CenterPointCodebookLoss, self).__init__(args)
        self.codebook_weight = 1.0
        if 'codebook_weight' in args.keys():
            self.codebook_weight = args['codebook_weight']
        print('self.codebook_weight', self.codebook_weight)

        self.detection_fix = False
        if 'detection_fix' in args.keys() and args['detection_fix']:
            self.detection_fix = True

        self.detection_loss_codebook = False
        if 'detection_loss_codebook' in args.keys() and args['detection_loss_codebook']:
            self.detection_loss_codebook = True
        print('self.detection_loss_codebook', self.detection_loss_codebook)

    def forward(self, output_dict, target_dict, suffix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        if self.detection_fix:
            total_loss = 0
        else:
            total_loss = super().forward(output_dict, target_dict, suffix)

        ###### Codebook loss ######
        codebook_loss = output_dict['codebook_loss']
        total_loss += self.codebook_weight * codebook_loss
        self.loss_dict.update({'total_loss': total_loss.item(),
                               'codebook_loss': codebook_loss.item()})
        if self.detection_loss_codebook:
            cls_wo_channel_compression = output_dict['cls_wo_channel_compression']
            bbox_wo_channel_compression = output_dict['bbox_wo_channel_compression']
            cls_w_channel_compression = output_dict['cls_preds']
            bbox_w_channel_compression = output_dict['bbox_preds']
            cls_cb_loss = F.mse_loss(cls_w_channel_compression, cls_wo_channel_compression)
            bbox_cb_loss = F.mse_loss(bbox_w_channel_compression, bbox_wo_channel_compression)
            total_loss += cls_cb_loss + bbox_cb_loss
            self.loss_dict.update({'cls_cb_loss': cls_cb_loss, 'bbox_cb_loss': bbox_cb_loss})
            print('total_loss', total_loss, '||', 'codebook_loss', codebook_loss, '||', 'cls_cb_loss', cls_cb_loss, '||', 'bbox_cb_loss', bbox_cb_loss)

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        codebook_loss = self.loss_dict.get('codebook_loss', 0)
        reg_cb_loss = self.loss_dict.get('reg_cb_loss', 0)
        cls_cb_loss = self.loss_dict.get('cls_cb_loss', 0)
        

        print("[epoch %d][%d/%d]%s, || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Codebook Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, codebook_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss', reg_loss,
                              epoch * batch_len + batch_id)
            writer.add_scalar('Confidence_loss', cls_loss,
                              epoch * batch_len + batch_id)