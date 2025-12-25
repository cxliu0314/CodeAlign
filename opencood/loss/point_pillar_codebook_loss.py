# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.loss.point_pillar_loss import PointPillarLoss

class PointPillarCodebookLoss(PointPillarLoss):
    def __init__(self, args):
        super(PointPillarCodebookLoss, self).__init__(args)
        self.codebook_params = args['codebook']

        self.detection_fix = False
        if 'detection_fix' in args.keys() and args['detection_fix']:
            self.detection_fix = True

    def forward(self, output_dict, target_dict, suffix=""):
        if self.detection_fix:
            total_loss = 0
        else:
            total_loss = super().forward(output_dict, target_dict, suffix)

        codebook_loss = 0
        if self.codebook_params['type'] == 'RecLoss':
            codebook_loss = output_dict['codebook_loss']
        elif self.codebook_params['type'] == 'BalancedSimilarityLoss':
            codebook_loss = self.balanced_sim_loss(output_dict, target_dict, self.codebook_params['foreground_weight'], self.codebook_params['background_weight'])

        total_loss += self.codebook_params['weight'] * codebook_loss
        self.loss_dict.update({'total_loss': total_loss.item(),
                               'codebook_loss': codebook_loss})
        return total_loss

    def balanced_sim_loss(self, output_dict, target_dict, fg_weight=0.75, bg_weight=0.25):
        positives = target_dict['pos_equal_one']  # [B, H, W, 2]
        total_sim_loss = 0
        foreground_mask = torch.logical_or(positives[..., 0], positives[..., 1]).float()  # [B, H, W]
        features = output_dict['codebook_features_trans']  # [B, N, C, H, W]
        
        for b in range(len(features)):
            feature = features[b]  # [N, C, H, W]
            mask = foreground_mask[b]  # [H, W]
        
            # avg_feature = feature.mean(dim=1)  # [N, H, W]
            # norm_feature = F.normalize(avg_feature, p=2, dim=0)  # [N, H, W]
            # sim_matrix = torch.einsum('nhw,mhw->nmhw', norm_feature, norm_feature)  # [N, N, H, W]
            norm_feature = F.normalize(feature, p=2, dim=1)  # [N, C, H, W]
            sim_matrix = torch.einsum('nchw,mchw->nmhw', norm_feature, norm_feature)
            
            weight_mask = torch.where(mask > 0, fg_weight, bg_weight)  # [H, W]
            weighted_sim = sim_matrix * weight_mask.unsqueeze(0).unsqueeze(0)  # [N, N, H, W]
            
            n = weighted_sim.size(0)
            triu_mask = torch.triu(torch.ones(n, n, device=weighted_sim.device), diagonal=1).bool()  # [N, N]
            valid_sim = weighted_sim[triu_mask]  # [K, H, W] where K is number of upper triangular elements
            
            if valid_sim.numel() > 0:
                if bg_weight == 0:
                    fg_sim = valid_sim[valid_sim > 0]
                    if fg_sim.numel() > 0:
                        total_sim_loss += 1-fg_sim.mean()
                else:
                    total_sim_loss += 1-valid_sim.mean()
        
        return total_sim_loss / len(features)
    
    def logging(self, epoch, batch_id, batch_len, writer = None, suffix=""):
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        # iou_loss = self.loss_dict.get('iou_loss', 0)
        codebook_loss = self.loss_dict.get('codebook_loss', 0)

        print("[epoch %d][%d/%d]%s || Loss: %.4f || Cls Loss: %.4f"
              " || Reg Loss: %.4f || Dir Loss: %.4f || Codebook Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, dir_loss, codebook_loss))

        if not writer is None:
            writer.add_scalar('Reg_loss' + suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Cls_loss' + suffix, cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss' + suffix, dir_loss,
                            epoch*batch_len + batch_id)
            # writer.add_scalar('Iou_loss' + suffix, iou_loss,
            #                 epoch*batch_len + batch_id)
            writer.add_scalar('Codebook_loss' + suffix, total_loss,
                epoch*batch_len + batch_id)
            writer.add_scalar('Total_loss' + suffix, codebook_loss,
                epoch*batch_len + batch_id)
