# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss
import itertools

class PointPillarPyramidCodebookLoss(PointPillarPyramidLoss):
    def __init__(self, args):
        super(PointPillarPyramidCodebookLoss, self).__init__(args)
        self.codebook_params = args['codebook']

        self.detection_fix = False
        if 'detection_fix' in args.keys() and args['detection_fix']:
            self.detection_fix = True
            
        self.detection_loss_codebook = False
        if 'detection_loss_codebook' in args.keys() and args['detection_loss_codebook']:
            self.detection_loss_codebook = True

    def forward(self, output_dict, target_dict, suffix=""):
        if self.detection_fix:
            total_loss = 0
        else:
            total_loss = super().forward(output_dict, target_dict, suffix)

        if suffix != "": 
            return super().forward(output_dict, target_dict, suffix)
        assert suffix == ""

        codebook_loss = 0
        if 'ReconstructionLoss' in self.codebook_params.keys():
            recon_loss = output_dict['codebook_loss']
            codebook_loss += recon_loss * self.codebook_params['ReconstructionLoss']['weight']
            self.loss_dict.update({'recon_loss': recon_loss})
        if 'PixelSimLoss' in self.codebook_params.keys():
            pixelsim_loss = self.PixelSimLoss(output_dict, target_dict, self.codebook_params['PixelSimLoss']['mode'], self.codebook_params['PixelSimLoss']['foreground_weight'], self.codebook_params['PixelSimLoss']['background_weight'])
            codebook_loss += pixelsim_loss * self.codebook_params['PixelSimLoss']['weight']
            self.loss_dict.update({'pixelsim_loss': pixelsim_loss})
        if 'InstanceSimLoss' in self.codebook_params.keys():
            instance_loss = self.InstanceSimLoss(output_dict, target_dict, self.codebook_params['InstanceSimLoss']['pool_size'], self.codebook_params['InstanceSimLoss']['temperature'])
            codebook_loss += instance_loss * self.codebook_params['InstanceSimLoss']['weight']
            self.loss_dict.update({'instancesim_loss': instance_loss})
        if 'cosine' in self.codebook_params.keys():
            cosine_loss = self.FeatureSimLoss(output_dict, mode='cosine')
            codebook_loss += cosine_loss * self.codebook_params['cosine']['weight']
            self.loss_dict.update({'cosine_loss': cosine_loss})
        if 'l2' in self.codebook_params.keys():
            l2_loss = self.FeatureSimLoss(output_dict, mode='l2')
            codebook_loss += l2_loss * self.codebook_params['l2']['weight']
            self.loss_dict.update({'l2_loss': l2_loss})
        if 'smoothl1' in self.codebook_params.keys():
            smoothl1_loss = self.FeatureSimLoss(output_dict, mode='smoothl1')
            codebook_loss += smoothl1_loss * self.codebook_params['smoothl1']['weight']
            self.loss_dict.update({'smoothl1_loss': smoothl1_loss})
        if 'mahalanobis' in self.codebook_params.keys():
            mahalanobis_loss = self.FeatureSimLoss(output_dict, mode='mahalanobis')
            codebook_loss += mahalanobis_loss * self.codebook_params['mahalanobis']['weight']
            self.loss_dict.update({'mahalanobis_loss': mahalanobis_loss})
        if 'mmd' in self.codebook_params.keys():
            mmd_loss = self.FeatureSimLoss(output_dict, mode='mmd')
            codebook_loss += mmd_loss * self.codebook_params['mmd']['weight']
            self.loss_dict.update({'mmd_loss': mmd_loss})
        if 'coral' in self.codebook_params.keys():
            coral_loss = self.FeatureSimLoss(output_dict, mode='coral')
            codebook_loss += coral_loss * self.codebook_params['coral']['weight']
            self.loss_dict.update({'coral_loss': coral_loss})
        if 'js' in self.codebook_params.keys():
            js_loss = self.FeatureSimLoss(output_dict, mode='js')
            codebook_loss += js_loss * self.codebook_params['js']['weight']
            self.loss_dict.update({'js_loss': js_loss})


        total_loss += codebook_loss
        self.loss_dict.update({'total_loss': total_loss.item(),
                               'codebook_loss': codebook_loss})
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
    
    def logging(self, epoch, batch_id, batch_len, writer = None, suffix=""):
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)
        depth_loss = self.loss_dict.get('depth_loss', 0)
        pyramid_loss = self.loss_dict.get('pyramid_loss', 0)
        codebook_loss = self.loss_dict.get('codebook_loss', 0)

        print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || Depth Loss: %.4f || Pyramid Loss: %.4f || Codebook Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, dir_loss, iou_loss, depth_loss, pyramid_loss, codebook_loss))
        if 'ReconstructionLoss' in self.codebook_params.keys():
            print("Reconstruction Loss: %.4f" % self.loss_dict.get('recon_loss', 0))
        if 'PixelSimLoss' in self.codebook_params.keys():
            print("PixelSim Loss: %.4f" % self.loss_dict.get('pixelsim_loss', 0))
        if 'InstanceSimLoss' in self.codebook_params.keys():
            print("InstanceSim Loss: %.4f" % self.loss_dict.get('instancesim_loss', 0))
        if 'cosine' in self.codebook_params.keys():
            print("Cosine Loss: %.4f" % self.loss_dict.get('cosine_loss', 0))
        if 'l2' in self.codebook_params.keys():
            print("L2 Loss: %.4f" % self.loss_dict.get('l2_loss', 0))
        if 'smoothl1loss' in self.codebook_params.keys():
            print("SmoothL1 Loss: %.4f" % self.loss_dict.get('smoothl1_loss', 0))
        if 'mahalanobis' in self.codebook_params.keys():
            print("Mahalanobis Loss: %.4f" % self.loss_dict.get('mahalanobis_loss', 0))
        if 'mmd' in self.codebook_params.keys():
            print("MMD Loss: %.4f" % self.loss_dict.get('mmd_loss', 0))
        if 'coral' in self.codebook_params.keys():
            print("CORAL Loss: %.4f" % self.loss_dict.get('coral_loss', 0))
        if 'js' in self.codebook_params.keys():
            print("JS Loss: %.4f" % self.loss_dict.get('js_loss', 0))
            
        if not writer is None:
            writer.add_scalar('Reg_loss' + suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Cls_loss' + suffix, cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss' + suffix, dir_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss' + suffix, iou_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Depth_loss' + suffix, depth_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Pyramid_loss' + suffix, pyramid_loss,
                epoch*batch_len + batch_id)
            writer.add_scalar('Codebook_loss' + suffix, codebook_loss,
                epoch*batch_len + batch_id)
            writer.add_scalar('Total_loss' + suffix, total_loss,
                epoch*batch_len + batch_id)
            if 'ReconstructionLoss' in self.codebook_params.keys():
                writer.add_scalar('ReconstructionLoss' + suffix, self.loss_dict.get('recon_loss', 0),
                                epoch*batch_len + batch_id)
            if 'PixelSimLoss' in self.codebook_params.keys():
                writer.add_scalar('PixelSimLoss' + suffix, self.loss_dict.get('pixelsim_loss', 0),
                                epoch*batch_len + batch_id)
            if 'InstanceSimLoss' in self.codebook_params.keys():
                writer.add_scalar('InstanceSimLoss' + suffix, self.loss_dict.get('instancesim_loss', 0),
                                epoch*batch_len + batch_id)
            if 'cosine' in self.codebook_params.keys():
                writer.add_scalar('CosineLoss' + suffix, self.loss_dict.get('cosine_loss', 0),
                                epoch*batch_len + batch_id)
            if 'l2' in self.codebook_params.keys():
                writer.add_scalar('L2Loss' + suffix, self.loss_dict.get('l2_loss', 0),
                                epoch*batch_len + batch_id)
            if 'smoothl1' in self.codebook_params.keys():
                writer.add_scalar('SmoothL1Loss' + suffix, self.loss_dict.get('smoothl1_loss', 0),
                                epoch*batch_len + batch_id)
            if 'mahalanobis' in self.codebook_params.keys():
                writer.add_scalar('MahalanobisLoss' + suffix, self.loss_dict.get('mahalanobis_loss', 0),
                                epoch*batch_len + batch_id)
            if 'mmd' in self.codebook_params.keys():
                writer.add_scalar('MMDLoss' + suffix, self.loss_dict.get('mmd_loss', 0),
                                epoch*batch_len + batch_id)
            if 'coral' in self.codebook_params.keys():
                writer.add_scalar('CORALLoss' + suffix, self.loss_dict.get('coral_loss', 0),
                                epoch*batch_len + batch_id)
            if 'js' in self.codebook_params.keys():
                writer.add_scalar('JSLoss' + suffix, self.loss_dict.get('js_loss', 0),
                                epoch*batch_len + batch_id)
    
    def PixelSimLoss(self, output_dict, target_dict, mode='all', fg_weight=0.75, bg_weight=0.25):
        features = output_dict['single_feature_warped']  # [B, N, C, H, W]
        mask = output_dict['single_warp_mask'] # [B, N, H, W]

        positives = target_dict['pos_equal_one']  # [B, H, W, 2]
        foreground_mask = torch.logical_or(positives[..., 0], positives[..., 1]).float()  # [B, H, W]
        
        total_sim_loss = 0
        for b in range(len(features)):
            feature = features[b]  # [N, C, H, W]
            f_mask = foreground_mask[b]  # [H, W]
            feature_mask = mask[b].bool()  # [N, H, W], mask for each feature
        
            if mode == 'mean':
                avg_feature = feature.mean(dim=1)  # [N, H, W]
                norm_feature = F.normalize(avg_feature, p=2, dim=0)  # [N, H, W]
                sim_matrix = torch.einsum('nhw,mhw->nmhw', norm_feature, norm_feature)  # [N, N, H, W]
            elif mode == 'all':
                norm_feature = F.normalize(feature, p=2, dim=1)  # [N, C, H, W]
                sim_matrix = torch.einsum('nchw,mchw->nmhw', norm_feature, norm_feature)

            weight_mask = torch.where(f_mask > 0, fg_weight, bg_weight)  # [H, W]
            feature_mask_pairwise = feature_mask.unsqueeze(1) & feature_mask.unsqueeze(0)  # [N, N, H, W]
            combined_mask = weight_mask.unsqueeze(0).unsqueeze(0) * feature_mask_pairwise  # [N, N, H, W]
            
            weighted_sim = sim_matrix * combined_mask  # [N, N, H, W]
            valid_sim = weighted_sim.sum(dim=(2, 3))  # [N, N], sum over H and W for each pair of features
            n_valid_elements = combined_mask.sum(dim=(2, 3))  # [N, N]
            avg_sim = valid_sim / (n_valid_elements + 1e-8)  # Avoid division by zero

            triu_mask = torch.triu(torch.ones_like(avg_sim), diagonal=1).bool()  # Upper triangular mask            
            valid_mask = n_valid_elements > 0  # 有效区域标识
            final_mask = triu_mask & valid_mask
            if final_mask.any():
                final_sim = avg_sim[final_mask].mean()
                total_sim_loss += 1 - final_sim

        return total_sim_loss / len(features)
    
    
    def NTXentLoss(self, x1, x2, temperature=0.5):
        sim_matrix = torch.matmul(x1, x2.T) / temperature
        sim_matrix = sim_matrix - torch.max(sim_matrix, dim=-1, keepdim=True)[0]
        labels = torch.arange(x1.size(0)).to(x1.device)
        loss = nn.CrossEntropyLoss()((sim_matrix + 1e-7), labels)
        return loss

    def instance_contra_loss(self, img, gt, bbox_tensor, output_size=9, temperature=0.5):
        B, C, H, W = img.shape
        pooled_features1_batch = []
        pooled_features2_batch = []
        for batch_idx in range(B):
            pooled_features1 = []
            pooled_features2 = []
            for bbox_idx in range(bbox_tensor.shape[0]):
                xmin, ymin, xmax, ymax = bbox_tensor[bbox_idx]
                cropped_feature_map1 = img[batch_idx, :, ymin:ymax, xmin:xmax]
                cropped_feature_map2 = gt[batch_idx, :, ymin:ymax, xmin:xmax]
                if torch.any(torch.all(cropped_feature_map1.flatten(1) == 0, dim=0)) or torch.any(torch.all(cropped_feature_map2.flatten(1) == 0, dim=0)):
                    continue

                if cropped_feature_map1.shape[1] > 0 and cropped_feature_map1.shape[2] > 0:
                    pooled_feature1 = F.adaptive_avg_pool2d(cropped_feature_map1, (output_size, output_size))  # (C1, 9, 9)
                else:
                    pooled_feature1 = torch.zeros((img.shape[1], output_size, output_size)).cuda()

                if cropped_feature_map2.shape[1] > 0 and cropped_feature_map2.shape[2] > 0:
                    pooled_feature2 = F.adaptive_max_pool2d(cropped_feature_map2, (output_size, output_size))  # (C2, 9, 9)
                else:
                    pooled_feature2 = torch.zeros((gt.shape[1], output_size, output_size)).cuda()
                pooled_features1.append(pooled_feature1.unsqueeze(0)) # (1, C1, 9, 9)
                pooled_features2.append(pooled_feature2.unsqueeze(0))  # (1, C2, 9, 9)
            if not pooled_features1 or not pooled_features2:
                continue
            pooled_features1 = torch.cat(pooled_features1, dim=0)  # (N, C1, 9, 9)
            pooled_features2 = torch.cat(pooled_features2, dim=0)  # (N, C2, 9, 9)
            pooled_features1_batch.append(pooled_features1.unsqueeze(0))  # (1, N, C1, 9, 9)
            pooled_features2_batch.append(pooled_features2.unsqueeze(0))  # (1, N, C2, 9, 9)
        if not pooled_features1_batch or not pooled_features2_batch:
            return torch.tensor(0.0, device=img.device, requires_grad=True)
        pooled_features1_batch = torch.cat(pooled_features1_batch, dim=0) # (B, N, C1, 9, 9)
        pooled_features2_batch = torch.cat(pooled_features2_batch, dim=0)  # (B, N, C2, 9, 9)
        B, N, _, _, _ = pooled_features1_batch.shape

        gt_embeddings = pooled_features2_batch.view(B*N, 64, -1, 9, 9).max(dim=2)[0]  # max across dim=2
        img_embeddings = pooled_features1_batch.view(B*N, 64, -1, 9, 9).max(dim=2)[0]  # max across dim=2

        bev_embeddings_flat = img_embeddings.reshape(B*N, -1)
        gt_embeddings_flat = gt_embeddings.reshape(B*N, -1)
        bev_embeddings_flat = F.normalize(bev_embeddings_flat, dim=-1)
        gt_embeddings_flat = F.normalize(gt_embeddings_flat, dim=-1)
        loss = self.NTXentLoss(bev_embeddings_flat, gt_embeddings_flat, temperature)
        return loss        
        
    def InstanceSimLoss(self, output_dict, target_dict, output_size=9,  temperature=0.5):
        features = output_dict['single_feature_warped']
        bbox = target_dict['gt_boxes'] # n,4
        B = len(features)
        agent_modality_list = output_dict['agent_modality_list'] # n

        loss = 0
        loss_n = 0
        for b in range(B):
            feature = features[b]
            N = feature.shape[0]
            combinations = list(itertools.combinations(range(N), 2))
            for i, j in combinations:
                if agent_modality_list[i] == agent_modality_list[j]:
                    continue
                feature_i = feature[i].unsqueeze(0)
                feature_j = feature[j].unsqueeze(0)
                loss += self.instance_contra_loss(feature_i, feature_j, bbox[b], output_size, temperature)
                loss_n += 1
        if loss_n > 0:
            loss /= loss_n
        else:
            loss = torch.tensor(0).cuda()

        return loss


    def FeatureSimLoss(self, output_dict, mode='cosine'):
        features = output_dict['single_feature_warped']  # [B, N, C, H, W]
        masks = output_dict['single_warp_mask']          # [B, N, H, W]
        modality_list = output_dict['agent_modality_list']  # list of length N

        B = len(features)
        total_loss = 0.0
        count = 0

        for b in range(B):
            feature = features[b]  # [N, C, H, W]
            mask = masks[b]        # [N, H, W]
            N = feature.shape[0]

            for i in range(1, N):
                if modality_list[0] == modality_list[i]:
                    continue

                f0 = feature[0]  # [C, H, W]
                fi = feature[i]
                m = mask[i]      # [H, W]

                if mode == 'cosine':
                    loss = self.cosine_similarity_loss(f0, fi, m)
                elif mode == 'l2':
                    loss = self.l2_loss(f0, fi, m)
                elif mode == 'smoothl1':
                    loss = self.smooth_l1_loss(f0, fi, m)
                elif mode == 'mahalanobis':
                    loss = self.mahalanobis_loss(f0, fi, m)
                elif mode == 'mmd':
                    loss = self.mmd_loss(f0, fi, m)
                elif mode == 'coral':
                    loss = self.coral_loss(f0, fi, m)
                elif mode == 'js':
                    loss = self.js_divergence_loss(f0, fi, m)
                else:
                    raise ValueError(f"Unsupported loss mode: {mode}")

                total_loss += loss
                count += 1

        if count > 0:
            return total_loss / count
        else:
            return torch.tensor(0.0, device=feature.device)

    def cosine_similarity_loss(self, feat1, feat2, mask):
        eps = 1e-6
        mask = mask.unsqueeze(0)  # [1, H, W]

        # 正规化每个像素点的通道向量
        feat1_norm = torch.norm(feat1, dim=0, keepdim=True).clamp(min=eps)
        feat2_norm = torch.norm(feat2, dim=0, keepdim=True).clamp(min=eps)
        feat1_unit = feat1 / feat1_norm
        feat2_unit = feat2 / feat2_norm

        # 每个位置的 cosine 值：通道间点积
        cosine_map = (feat1_unit * feat2_unit).sum(dim=0)  # [H, W]
        cosine_map = torch.clamp(cosine_map, -1.0, 1.0)  # 数值保护

        # 损失越小，表示越相似
        loss_map = 1 - cosine_map
        loss = (loss_map * mask.squeeze(0)).sum() / (mask.sum() + eps)
        return loss

    def l2_loss(self, feat1, feat2, mask):
        mask = mask.unsqueeze(0)
        valid = mask.sum()
        if valid == 0:
            return torch.tensor(0.0, device=feat1.device)
        diff = (feat1 - feat2) ** 2 * mask
        return diff.sum() / (valid + 1e-6)
    
    def smooth_l1_loss(self, feat1, feat2, mask):
        mask = mask.unsqueeze(0)
        valid = mask.sum()
        if valid == 0:
            return torch.tensor(0.0, device=feat1.device)
        
        diff = (feat1 - feat2) * mask
        abs_diff = diff.abs()
        loss = torch.where(abs_diff < 1, 0.5 * diff ** 2, abs_diff - 0.5)
        return loss.sum() / (valid + 1e-6)

    def mahalanobis_loss(self, feat1, feat2, mask):
        mask = mask.bool()
        f1 = feat1[:, mask]  # [C, N]
        f2 = feat2[:, mask]
        
        if f1.shape[1] < 2:  # 样本数不足
            return torch.tensor(0.0, device=feat1.device)

        try:
            diff = (f1 - f2).T  # [N, C]
            data = torch.cat([f1, f2], dim=1)
            cov = torch.cov(data)  # [C, C]
            inv_cov = torch.linalg.pinv(cov)
            dist = torch.einsum('nc,cd,nd->n', diff, inv_cov, diff)
            return dist.mean()
        except Exception:
            return torch.tensor(0.0, device=feat1.device)
        
    def mmd_loss(self, feat1, feat2, mask, sigma=1.0, sample_k=256):
        mask = mask.bool()
        f1 = feat1[:, mask].T  # [N, C]
        f2 = feat2[:, mask].T

        if f1.shape[0] < 2 or f2.shape[0] < 2:
            return torch.tensor(0.0, device=feat1.device)

        # 随机采样 k 个点，避免 O(N^2) 的爆炸显存
        def sample(x, k):
            if x.shape[0] <= k:
                return x
            idx = torch.randperm(x.shape[0], device=x.device)[:k]
            return x[idx]

        f1 = sample(f1, sample_k)  # [k, C]
        f2 = sample(f2, sample_k)

        def rbf(x, y):
            dist = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(-1)
            return torch.exp(-dist / (2 * sigma ** 2))

        Kxx = rbf(f1, f1)
        Kyy = rbf(f2, f2)
        Kxy = rbf(f1, f2)

        return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

    def coral_loss(self, feat1, feat2, mask):
        mask = mask.bool()
        f1 = feat1[:, mask].T
        f2 = feat2[:, mask].T

        if f1.shape[0] < 2 or f2.shape[0] < 2:
            return torch.tensor(0.0, device=feat1.device)

        c1 = f1 - f1.mean(0)
        c2 = f2 - f2.mean(0)
        cov1 = (c1.T @ c1) / max(f1.shape[0] - 1, 1)
        cov2 = (c2.T @ c2) / max(f2.shape[0] - 1, 1)
        return ((cov1 - cov2) ** 2).mean()

    def js_divergence_loss(self, feat1, feat2, mask):
        mask = mask.bool()
        f1 = feat1[:, mask].T
        f2 = feat2[:, mask].T

        if f1.shape[0] < 1 or f2.shape[0] < 1:
            return torch.tensor(0.0, device=feat1.device)

        p = F.softmax(f1, dim=1).clamp(min=1e-8)
        q = F.softmax(f2, dim=1).clamp(min=1e-8)
        m = 0.5 * (p + q)

        js = 0.5 * (F.kl_div(m.log(), p, reduction='batchmean') +
                    F.kl_div(m.log(), q, reduction='batchmean'))
        return torch.nan_to_num(js, nan=0.0, posinf=1.0, neginf=1.0)


