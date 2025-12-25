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
from opencood.models.sub_modules.codebook import UMGMQuantizer
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.fuse_modules.fusion_in_one import regroup
import os
from opencood.utils.codebook_utils import gumbelSoftmax
# from opencood.models.fuse_modules.adapter import Adapter
from opencood.models.c2c_module import Translator
import random


class HeterCodebookSharedHeadReorderC2C(nn.Module):
    def __init__(self, args, train_flag=True):
        super(HeterCodebookSharedHeadReorderC2C, self).__init__()
        self.args = args
        self.train_flag = train_flag

        ### add by czc 用于忽略部分模型 与forward配套
        ignored_modality = set(args.get("ignored_modality", []))
        modality_name_list = list(args.keys() - ignored_modality)
        ### 

        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list
        
        self.voxel_size = args['voxel_size']
        self.out_size_factor = args['out_size_factor']
        self.cav_lidar_range  = args['lidar_range']
        
        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        self.cam_crop_info = {} 

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

            """
            Aligner building
            """
            if args.get("aligner", False):
                setattr(self, f"aligner_{modality_name}", AlignNet(args['aligner']))

            """
            Load ego model
            """
            # if self.train_flag:
            model_path = model_setting['model_path']
            if model_path != '':
                # Replace direct rank retrieval with conditional check
                if torch.distributed.is_initialized():
                    local_rank = torch.distributed.get_rank()
                else:
                    local_rank = 0  # Default to rank 0 if not using distributed
                checkpoint = torch.load(model_path, map_location='cuda:{}'.format(local_rank))
                # checkpoint = torch.load(model_path)
                encoder_state_dict = {k.replace(f'encoder_{modality_name}.', ''): v for k, v in checkpoint.items() if f'encoder_{modality_name}'==k.split('.')[0]}
                backbone_state_dict = {k.replace(f'backbone_{modality_name}.', ''): v for k, v in checkpoint.items() if f'backbone_{modality_name}'==k.split('.')[0]}
                encoder_model = getattr(self, f"encoder_{modality_name}")
                backbone_model = getattr(self, f"backbone_{modality_name}")
                encoder_model.load_state_dict(encoder_state_dict, strict=True)
                backbone_model.load_state_dict(backbone_state_dict, strict=True)
                print(f"Encoder and Backbone for modality {modality_name} loaded successfully.")
                if args.get("aligner", False):
                    aligner_state_dict = {k.replace(f'aligner_{modality_name}.', ''): v for k, v in checkpoint.items() if f'aligner_{modality_name}' in k}
                    aligner_model = getattr(self, f"aligner_{modality_name}", None)
                    aligner_model.load_state_dict(aligner_state_dict, strict=True)
                    print(f"Aligner for modality {modality_name} loaded successfully.")

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
        C2C Translator
        """
        self.translator = Translator(args["translator"])

        self.codebook_flag = True if 'codebook' in args else False
        self.shrink_flag = False
        self.fusion_method = args['fusion']['method']

        """
        Backend Models
        """
        from torch.nn import ModuleDict
        self.backend_models = ModuleDict()
        for mod in args['backend_modality']:
            model_path = args['backend'][mod]
            if model_path != '':
                if torch.distributed.is_initialized():
                    local_rank = torch.distributed.get_rank()
                else:
                    local_rank = 0
                checkpoint = torch.load(model_path, map_location='cuda:{}'.format(local_rank))

                channel = 64
                p_rate = 0.0
                backend_modules = {
                    'codebook': UMGMQuantizer(
                        channel,
                        args['codebook']['seg_num'],
                        [args['codebook']['dict_size'] for _ in range(args['codebook']['r'])],
                        p_rate,
                        {
                            "latentStageEncoder": lambda: nn.Linear(channel, channel),
                            "quantizationHead": lambda: nn.Linear(channel, channel),
                            "latentHead": lambda: nn.Linear(channel),
                            "restoreHead": lambda: nn.Linear(channel, channel),
                            "dequantizationHead": lambda: nn.Linear(channel, channel),
                            "sideHead": lambda: nn.Linear(channel, channel)
                        }
                    ).to(local_rank),
                    'fusion': PyramidFusion(args['fusion']['backbone']).to(local_rank) if self.fusion_method == 'pyramidfusion' else MaxFusion().to(local_rank),
                    'cls_head': nn.Conv2d(args['in_head'], args['anchor_number'], kernel_size=1).to(local_rank),
                    'reg_head': nn.Conv2d(args['in_head'], 7 * args['anchor_number'], kernel_size=1).to(local_rank),
                    'dir_head': nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1).to(local_rank)
                }
                if self.fusion_method == 'pyramidfusion':
                    backend_modules['shrink_conv'] = DownsampleConv(args['fusion']['shrink_head']).to(local_rank)
                    self.shrink_flag = True
                
                self.backend_models[mod] = ModuleDict(backend_modules)
                
                codebook_state_dict = {k.replace('multi_channel_compressor.', ''): v for k, v in checkpoint.items() if 'multi_channel_compressor' in k}
                self.backend_models[mod]['codebook'].load_state_dict(codebook_state_dict, strict=True)
                print(f"Codebook for {mod} loaded successfully.")
                # Load fusion state dict
                fusion_state_dict = {k.replace('pyramid_backbone.', ''): v for k, v in checkpoint.items() if 'pyramid_backbone' in k}
                self.backend_models[mod]['fusion'].load_state_dict(fusion_state_dict, strict=True)
                print(f"Fusion backbone for {mod} loaded successfully.")
                # Load heads state dict
                self.backend_models[mod]['cls_head'].load_state_dict({k.replace('cls_head.', ''): v for k, v in checkpoint.items() if 'cls_head' in k}, strict=True)
                self.backend_models[mod]['reg_head'].load_state_dict({k.replace('reg_head.', ''): v for k, v in checkpoint.items() if 'reg_head' in k}, strict=True)
                self.backend_models[mod]['dir_head'].load_state_dict({k.replace('dir_head.', ''): v for k, v in checkpoint.items() if 'dir_head' in k}, strict=True)
                print(f"Heads for {mod} loaded successfully.")
                # Load shrink_conv state dict if needed
                if self.fusion_method == 'pyramidfusion':
                    shrink_state_dict = {k.replace('shrink_conv.', ''): v for k, v in checkpoint.items() if 'shrink_conv' in k}
                    self.backend_models[mod]['shrink_conv'].load_state_dict(shrink_state_dict, strict=True)
                    print(f"Shrink conv for {mod} loaded successfully.")
            else:
                print(f"Warning: No model path provided for {mod}. Skipping initialization.")

        if 'fix_encoder' in args and args['fix_encoder']:
            self.fix_encoder()
        if 'only_train_translator' in args and args['only_train_translator']:
            self.only_train_translator()
        if 'only_train_embedding' in args and args['only_train_embedding']:
            self.only_train_embedding()
            if self.train_flag:
                model_path = args['translator_path']
                if model_path != '':
                    if torch.distributed.is_initialized():
                        local_rank = torch.distributed.get_rank()
                    else:
                        local_rank = 0
                    checkpoint = torch.load(model_path, map_location='cuda:{}'.format(local_rank))
                    translator_model = getattr(self, 'translator')
                    translator_state_dict = {k.replace('translator.', '', 1): v for k, v in checkpoint.items() if 'translator' in k and not k.startswith('translator.translator.emb_table')}
                    translator_model.load_state_dict(translator_state_dict, strict=False)
                    print(f"translator loaded successfully.")

        # check again which module is not fixed.
        if self.train_flag:
            print('----------- Training Parameters -----------')
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(name, param.data.shape)
            print('-------------------------------------------')
            print('----------- Fixed Parameters -----------')
            for name, param in self.named_parameters():
                if not param.requires_grad:
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
    
    def only_train_translator(self):
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        
        self.translator.train()
        for p in self.translator.parameters():
            p.requires_grad_(True)

    def only_train_embedding(self):
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

        self.translator.train()
        self.translator.translator.emb_table.requires_grad_(True)
        
        
    def forward(self, data_dict, vis_path=''):
        output_dict = {}
        vis_dict = {}
        agent_modality_list = data_dict['agent_modality_list'] 
        record_len = data_dict['record_len'] 


        ### add by czc 因为 需要添加ignore部分模型的部分 所以需要重新计算 index 用于排除 被ignore的模型 from stamp:
        # 这里与 line 38配套
        pairwise_t_matrix = data_dict["pairwise_t_matrix"]
        pairwise_t_matrix_new = torch.zeros_like(pairwise_t_matrix)
        agent_modality_list_filtered = []
        record_len_filtered = []
        cur = 0
        count = 0
        ptr = 0
        indices = []
        for m in agent_modality_list:
            if m in self.modality_name_list:
                agent_modality_list_filtered.append(m)
                count += 1
                indices.append(cur)
            cur += 1
            if record_len[ptr] == cur:
                record_len_filtered.append(count)
                if len(indices) > 0:
                    for i in range(len(indices)):
                        for j in range(len(indices)):
                            pairwise_t_matrix_new[ptr][i][j] = pairwise_t_matrix[ptr][indices[i]][indices[j]]
                cur = 0
                count = 0
                ptr += 1
                indices = []
        pairwise_t_matrix = pairwise_t_matrix_new
        affine_matrix = normalize_pairwise_tfm(pairwise_t_matrix, self.H, self.W, self.fake_voxel_size)
        record_len = torch.tensor(record_len_filtered)
        agent_modality_list = agent_modality_list_filtered

        # 替换掉了这个部分：
        # affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        ##### 

        


        print(agent_modality_list)
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}        
        
        # Randomly select backend modality
        if self.train_flag:
            backend_modality = random.choice(self.args['backend_modality'])
        else:
            backend_modality = agent_modality_list[0]
        # print(f"backend modality: {backend_modality}")

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
        Assemble heter features
        """
        if 'm5' in modality_feature_dict:
            modality_feature_dict['m5'] = F.interpolate(
                modality_feature_dict['m5'],
                size=(256, 256),
                mode='bilinear',
                align_corners=False
            )
        
        """
        Codebook Part
        """
        output_dict.update({'affine_matrix': affine_matrix,
                            'agent_modality_list': agent_modality_list,})
        if self.codebook_flag:
            codemap_dict = {}            
            for mod_name, feat in modality_feature_dict.items():
                N, C, H, W = feat.shape
                if mod_name == backend_modality:
                    heter_feature_2d = feat.permute(0, 2, 3, 1).contiguous().reshape(-1, C)
                    heter_feature_2d, code, _, _ = self.backend_models[backend_modality]['codebook'](heter_feature_2d)
                    heter_feature_2d = heter_feature_2d.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()
                    modality_feature_dict[mod_name] = heter_feature_2d
                    # print(f"encoder {mod_name} backend {backend_modality}, no translator")
                else:
                    feature_code2code = self.translator(feat, backend_modality)
                    _, C1, _, _ = feature_code2code.shape # C1: codebook size 
                    quantizeds_code2code = gumbelSoftmax(feature_code2code.permute(0, 2, 3, 1).contiguous().view(-1,1,C1), 1.0, True)
                    # Use selected modality's codebook
                    feature = self.backend_models[backend_modality]['codebook'].get_feature([quantizeds_code2code])
                    feature = feature.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()
                    modality_feature_dict[mod_name] = feature
                    # print(f"encoder {mod_name} backend {backend_modality}, with translator")

                # if vis_path!='':
                #     quantizeds_code2code = quantizeds_code2code.view(-1, H, W, C1).permute(0, 3, 1, 2)
                #     codemap_dict[mod_name] = quantizeds_code2code

        """
        Crop/Padd camera feature map.
        """
        for mod_name in list(modality_feature_dict.keys()):
            if self.sensor_type_dict[mod_name] == "camera":
                feat = modality_feature_dict[mod_name]  # [N_cam, C, H, W]
                _, _, H, W = feat.shape
                target_H = int(H * eval(f"self.crop_ratio_H_{mod_name}"))
                target_W = int(W * eval(f"self.crop_ratio_W_{mod_name}"))

                crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                modality_feature_dict[mod_name] = crop_func(feat)
                if eval(f"self.depth_supervision_{mod_name}"):
                    output_dict[f"depth_items_{mod_name}"] = getattr(self, f"encoder_{mod_name}").depth_items

            #     if vis_path!='':
            #         feat = codemap_dict[mod_name]
            #         feat_argmax = torch.argmax(feat, dim=1)  # [n, 128, 128]
            #         output = torch.full((feat.shape[0], 256, 256), -1, dtype=torch.float, device=feat.device)
            #         output[:, 64:192, 64:192] = feat_argmax
            #         codemap_dict[mod_name] = output.unsqueeze(1)  # [N, 1, 256, 256]
            # else: 
            #     if vis_path!='':
            #         feat = codemap_dict[mod_name]
            #         feat_argmax = torch.argmax(feat, dim=1)  # [n, 128, 128]
            #         codemap_dict[mod_name] = feat_argmax.unsqueeze(1).float()  # [N, 1, 256, 256]

        """
        Concat features
        """
        counting = {m: 0 for m in self.modality_name_list}
        heter_feats = []
        codemap_list = []
        for mod_name in agent_modality_list:
            idx = counting[mod_name]
            f = modality_feature_dict[mod_name][idx]
            heter_feats.append(f)
            # if vis_path!='':
            #     c = codemap_dict[mod_name][idx]
            #     codemap_list.append(c)
            counting[mod_name] += 1
        heter_feature_2d = torch.stack(heter_feats, dim=0)  # [N_agents, C, H', W']

        if vis_path!='':
            vis_dict["codebook"] = heter_feature_2d

        # if vis_path!='':
        #     vis_dict['codemap'] = torch.stack(codemap_list, dim=0)
        #     code = vis_dict['codemap']
        #     _, C, H, W = code.shape
        #     B, L = affine_matrix.shape[:2]
        #     single_split_x = regroup(code, record_len)
        #     single_feature_transed = []
        #     for b in range(B):
        #         N = record_len[b]
        #         t_matrix = affine_matrix[b][:N, :N, :, :]
        #         single_feature_in_ego = warp_affine_simple(single_split_x[b], t_matrix[0, :, :, :], (H, W), mode='nearest', align_corners=False)
        #         single_feature_transed.append(single_feature_in_ego)
        #     vis_dict['code_warp'] = single_feature_transed[0]

        """
        Fusion net
        """
        # Use selected modality's fusion
        if self.fusion_method == 'pyramidfusion':
            fused_feature, occ_outputs = self.backend_models[backend_modality]['fusion'].forward_collab(
                                                heter_feature_2d,
                                                record_len, 
                                                affine_matrix, 
                                                agent_modality_list, 
                                                self.cam_crop_info
                                            )
            fused_feature = self.backend_models[backend_modality]['shrink_conv'](fused_feature)
            output_dict.update({'occ_single_list': occ_outputs})
            output_dict['pyramid'] = 'collab'
        else:
            fused_feature = self.backend_models[backend_modality]['fusion'].forward(heter_feature_2d, record_len, affine_matrix)

        if vis_path!='':
            vis_dict['fusion'] = fused_feature
        
        """
        Detection heads
        """
        # Use selected modality's heads
        cls_preds = self.backend_models[backend_modality]['cls_head'](fused_feature)
        reg_preds = self.backend_models[backend_modality]['reg_head'](fused_feature)
        dir_preds = self.backend_models[backend_modality]['dir_head'](fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        if vis_path!='':
            vis_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
            return vis_dict
        

        return output_dict