""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, ScaledDotProductAttention
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
import torchvision
from opencood.models.sub_modules.codebook import ChannelCompressor
from opencood.models.sub_modules.codebook import UMGMQuantizer
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.fuse_modules.fusion_in_one import regroup
import os

class HeterCodebookSharedHead(nn.Module):
    def __init__(self, args, train_flag=True):
        super(HeterCodebookSharedHead, self).__init__()
        self.args = args
        self.train_flag = train_flag
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list
        
        self.voxel_size = args['voxel_size']
        self.out_size_factor = args['out_size_factor']
        self.cav_lidar_range  = args['lidar_range']
        
        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        self.cam_crop_info = {} 

        local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            """
            Encoder building
            """
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            """
            Backbone building 
            """
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))

            # Load encoder ckpt
            if self.train_flag and args.get("fix_encoder", False):
                model_path = model_setting['model_path']
                if model_path != '':
                    checkpoint = torch.load(model_path,map_location='cuda:{}'.format(local_rank))
                    encoder_state_dict = {k.replace(f'encoder_{modality_name}.', ''): v for k, v in checkpoint.items() if f'encoder_{modality_name}' in k}
                    backbone_state_dict = {k.replace(f'backbone_{modality_name}.', ''): v for k, v in checkpoint.items() if f'backbone_{modality_name}' in k}
                    encoder_model = getattr(self, f"encoder_{modality_name}")
                    backbone_model = getattr(self, f"backbone_{modality_name}")
                    encoder_model.load_state_dict(encoder_state_dict, strict=False)
                    backbone_model.load_state_dict(backbone_state_dict, strict=False)
                    print(f"Encoder and Backbone for modality {modality_name} loaded successfully.")

            """
            Aligner building
            """
            if args.get("aligner", False):
                setattr(self, f"aligner_{modality_name}", AlignNet(args['aligner']))

            # camera mask setting
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                    f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
                }

        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        """
        Codebook
        """
        self.codebook_flag = True if 'codebook' in args else False
        if self.codebook_flag:
            channel = 64
            p_rate = 0.0
            seg_num = args['codebook']['seg_num']
            dict_size = [args['codebook']['dict_size'] for _ in range(args['codebook']['r'])]
            self.codebook = UMGMQuantizer(channel, seg_num, dict_size, p_rate,
                            {"latentStageEncoder": lambda: nn.Linear(channel, channel), "quantizationHead": lambda: nn.Linear(channel, channel),
                            "latentHead": lambda: nn.Linear(channel, channel), "restoreHead": lambda: nn.Linear(channel, channel),
                            "dequantizationHead": lambda: nn.Linear(channel, channel), "sideHead": lambda: nn.Linear(channel, channel)})
            print("codebook:", self.codebook_flag)
            print("seg_num: ", seg_num)        
            print("dict_size: ", args['codebook']['dict_size'])
        
            if self.train_flag and args.get("load_codebook", False):
                model_path = args['codebook_path']
                if model_path != '':
                    checkpoint = torch.load(model_path,map_location='cuda:{}'.format(local_rank))
                    codebook_state_dict = {k.replace('codebook.', ''): v for k, v in checkpoint.items() if 'codebook' in k}
                    codebook_model = getattr(self, 'codebook')
                    codebook_model.load_state_dict(codebook_state_dict, strict=True)
                    print(f"Codebook loaded successfully.")

        """
        Fusion
        """
        self.shrink_flag = False
        self.fusion_method = args['fusion']['method']
        if self.fusion_method == 'maxfusion':
            self.fusion_backbone = MaxFusion()
        elif self.fusion_method == 'attfusion':
            self.fusion_backbone = AttFusion(args['fusion']['backbone'])
        elif self.fusion_method == 'pyramidfusion':
            self.fusion_backbone = PyramidFusion(args['fusion']['backbone'])
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['fusion']['shrink_head'])

        """
        Shared Heads
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        
        # Load backend ckpt
        if self.train_flag and args.get("fix_backend", False):
            model_path = model_setting['model_path']
            if model_path != '':
                checkpoint = torch.load(model_path,map_location='cuda:{}'.format(local_rank))
                # Load fusion state dict
                fusion_state_dict = {k.replace(f'fusion_backbone.', ''): v for k, v in checkpoint.items() if f'fusion_backbone.' in k}
                self.fusion_backbone.load_state_dict(fusion_state_dict, strict=True)
                print(f"Fusion backbone loaded successfully.")
                # Load heads state dict
                self.cls_head.load_state_dict({k.replace(f'cls_head.', ''): v for k, v in checkpoint.items() if f'cls_head' in k}, strict=True)
                self.reg_head.load_state_dict({k.replace(f'reg_head.', ''): v for k, v in checkpoint.items() if f'reg_head' in k}, strict=True)
                self.dir_head.load_state_dict({k.replace(f'dir_head.', ''): v for k, v in checkpoint.items() if f'dir_head' in k}, strict=True)
                print(f"Heads loaded successfully.")
                # Load shrink_conv state dict if needed
                if self.fusion_method == 'pyramidfusion':
                    shrink_state_dict = {k.replace(f'shrink_conv.', ''): v for k, v in checkpoint.items() if f'shrink_conv' in k}
                    self.shrink_conv.load_state_dict(shrink_state_dict, strict=True)
                    print(f"Shrink conv loaded successfully.")

        # self.model_train_init()

    def model_train_init(self):
        if 'fix_encoder' in self.args and self.args['fix_encoder']:
            self.fix_encoder()
        if 'fix_backend' in self.args and self.args['fix_backend']:
            self.fix_backend()

        check_trainable_module(self)
        print('----------- Training Parameters -----------')
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)
        print('-------------------------------------------')

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
            
    def fix_encoder(self):
        for modality_name in self.modality_name_list:
            eval(f"self.encoder_{modality_name}").eval()
            eval(f"self.backbone_{modality_name}").eval()
            for p in eval(f"self.encoder_{modality_name}").parameters():
                p.requires_grad_(False)
            for p in eval(f"self.backbone_{modality_name}").parameters():
                p.requires_grad_(False)
    def fix_backend(self):
        self.fusion_backbone.eval()
        for p in self.fusion_backbone.parameters():
            p.requires_grad_(False)
        if self.fusion_method == 'pyramidfusion':
            self.shrink_conv.eval()
            for p in self.shrink_conv.parameters():
                p.requires_grad_(False)
        self.cls_head.eval()
        self.reg_head.eval()
        self.dir_head.eval()
        for p in self.cls_head.parameters():
            p.requires_grad_(False)
        for p in self.reg_head.parameters():
            p.requires_grad_(False)
        for p in self.dir_head.parameters():
            p.requires_grad_(False)

    def forward(self, data_dict, vis_path=''):
        output_dict = {}
        vis_dict = {}
        agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        print(agent_modality_list)
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for i, modality_name in enumerate(self.modality_name_list):
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            if vis_path!='':
                vis_dict[f'encoder_{i}'] = feature
            if self.args.get("aligner", False):
                feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        """
        Crop/Padd camera feature map.
        """
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })

        """
        Assemble heter features
        """
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            if modality_name=='m5':
                spatial_aligned_feature = F.interpolate(modality_feature_dict[modality_name][feat_idx].unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
            else:
                spatial_aligned_feature = modality_feature_dict[modality_name][feat_idx]
            heter_feature_2d_list.append(spatial_aligned_feature)
            counting_dict[modality_name] += 1
        heter_feature_2d = torch.stack(heter_feature_2d_list)
        
        if vis_path!='':
            vis_dict['aligner'] = heter_feature_2d

        """
        Codebook Part
        """
        N, C, H, W = heter_feature_2d.shape
        output_dict.update({'affine_matrix': affine_matrix,
                            'agent_modality_list': agent_modality_list,})
        if self.codebook_flag:
            heter_feature_2d = heter_feature_2d.permute(0, 2, 3, 1).contiguous().view(-1, C)
            heter_feature_2d, code, _, codebook_loss = self.codebook(heter_feature_2d)
            heter_feature_2d = heter_feature_2d.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()
            output_dict.update({'codebook_loss': codebook_loss})
            if vis_path!='':
                vis_dict['codebook'] = heter_feature_2d.clone()
                vis_dict['code'] = code[0].view(-1, H, W, 1).permute(0, 3, 1, 2)


        """
        Feature transformation
        """
        if self.args.get("pixel_sim", False) or vis_path!='':
            # warp to ego
            _, C, H, W = heter_feature_2d.shape
            B, L = affine_matrix.shape[:2]
            single_split_x = regroup(heter_feature_2d, record_len)
            single_feature_transed = []
            single_warp_mask = []
            for b in range(B):
                N = record_len[b]
                t_matrix = affine_matrix[b][:N, :N, :, :]
                single_feature_in_ego = warp_affine_simple(single_split_x[b], t_matrix[0, :, :, :], (H, W), mode='nearest', align_corners=False)
                single_feature_transed.append(single_feature_in_ego)
                single_warp_mask.append((single_feature_in_ego.sum(dim=1) != 0).float())
            output_dict.update({'single_feature_warped': single_feature_transed})
            output_dict.update({'single_warp_mask': single_warp_mask})
            if vis_path!='':
                vis_dict['codebook_warp'] = single_feature_transed[0]
                vis_dict['warp_mask'] = single_warp_mask

                # code = code[0].view(-1, H, W, 1).permute(0, 3, 1, 2).float()
                # _, C, H, W = code.shape
                # B, L = affine_matrix.shape[:2]
                # single_split_x = regroup(code, record_len)
                # single_feature_transed = []
                # single_warp_mask = []
                # for b in range(B):
                #     N = record_len[b]
                #     t_matrix = affine_matrix[b][:N, :N, :, :]
                #     single_feature_in_ego = warp_affine_simple(single_split_x[b], t_matrix[0, :, :, :], (H, W), mode='nearest', align_corners=False)
                #     single_feature_transed.append(single_feature_in_ego)
                # vis_dict['code_warp'] = single_feature_transed[0]


        """
        Fusion net
        """
        if self.fusion_method == 'pyramidfusion':
            fused_feature, occ_outputs = self.fusion_backbone.forward_collab(
                                                heter_feature_2d,
                                                record_len, 
                                                affine_matrix, 
                                                agent_modality_list, 
                                                self.cam_crop_info
                                            )
            fused_feature = self.shrink_conv(fused_feature)      
            output_dict.update({'occ_single_list': occ_outputs})
            output_dict['pyramid'] = 'collab'
        else:
            fused_feature = self.fusion_backbone.forward(heter_feature_2d, record_len, affine_matrix)

        if vis_path!='':
            vis_dict['fusion'] = fused_feature
        
        """
        Detection heads
        """
        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        if vis_path!='':
            vis_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
            return vis_dict
        

        return output_dict