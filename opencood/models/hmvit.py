"""
HM-ViT, but use ours heterogeneous modality and model.
"""

# In this heterogeneous version, feature align start before backbone.

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
# from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.fuse_modules.fuse_utils import regroup as Regroup
from opencood.models.fuse_modules.hmvit.hetero_fusion import HeteroFusionBlock
from opencood.models.fuse_modules.hmvit.base_transformer import HeteroFeedForward
from opencood.models.fuse_modules.hmvit.hetero_decoder import HeteroDecoder
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, warp_feature
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
import torchvision


class HeteroFusion(nn.Module):
    def __init__(self, config,train_flag=None):
        super(HeteroFusion, self).__init__()
        self.downsample_rate = config['spatial_transform']['downsample_rate']
        self.discrete_ratio = config['spatial_transform']['voxel_size'][0]
        self.num_types = config.get('num_types', 2)

        self.hetero_fusion_block = HeteroFusionBlock(
            config['hetero_fusion_block'])
        input_dim = config['hetero_fusion_block']['input_dim']

        self.num_iters = config['num_iters']
        self.mlp_head = HeteroFeedForward(input_dim, input_dim, 0, self.num_types)
        self.warpslow = config.get('warpslow', True)

    def forward(self, x, pairwise_t_matrix, mode, record_len, mask):
        temp = mode.detach().clone()
        # pairwise_t_matrix[:, :, :, :, :] = pairwise_t_matrix[:, :, :1, :, :].detach().clone()

        # x = self.fc_input(x.permute(0,1,3,4,2), mode).permute(0,1,4,2,3)
        for _ in range(self.num_iters):
            x = self.hetero_fusion_block(x, pairwise_t_matrix, temp,
                                         record_len, mask, self.warpslow)
        # x = x[:, 0, ...]
        # (B, M, C, H, W) -> (B, C, H, W)
        x = x[:, 0, ...].permute(0, 2, 3, 1)
        x = self.mlp_head(x.unsqueeze(1), temp[:, :1]).squeeze(1).permute(0, 3, 1, 2)
        return x

class HMViT(nn.Module):
    def __init__(self, args,train_flag=None):
        super(HMViT, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            """
            Encoder building
            """
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
            Shrink conv building
            """
            setattr(self, f"shrink_conv_{modality_name}",  DownsampleConv(model_setting['shrink_header']))
            
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))

        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        """
        Fusion, by default multiscale fusion: 
        Note the input of PyramidFusion has downsampled 2x. (SECOND required)
        """
        self.fusion_net = HeteroFusion(args['hmvit'])
        self.decoder = HeteroDecoder(args['hetero_decoder'])

        # check again which module is not fixed.
        check_trainable_module(self)

    def forward(self, data_dict):
        output_dict = {}
        agent_modality_list = data_dict['agent_modality_list'] 
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        record_len = data_dict['record_len'] 

        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.shrink_conv_{modality_name}")(feature)
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
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)

        # in HM-ViT paper, the feature is downsampled 4x, feature channel all 256
        # N, C, H, W -> B,  L, C, H, W

        # build mode for original HM-ViT code
        B, L = pairwise_t_matrix.shape[:2]
        mode = torch.zeros(B, L).to(heter_feature_2d.device).int()

        # if len(self.modality_name_list) <= 2:
        #     mode_flatten = torch.tensor([1 if modality == 'm1' else 0 for modality in agent_modality_list]).to(heter_feature_2d.device).int()

        # elif len(self.modality_name_list) == 3:
        #     mode_flatten = []
        #     for modality in agent_modality_list:
        #         if modality == 'm1':
        #             mode_flatten.append(1)
        #         elif modality == 'm2':
        #             mode_flatten.append(0)
        #         else: # m3 or m4
        #             mode_flatten.append(2)
        #     mode_flatten = torch.tensor(mode_flatten).to(heter_feature_2d.device).int()
        
        # elif len(self.modality_name_list) == 4:
        #     mode_flatten = []
        #     for modality in agent_modality_list:
        #         if modality == 'm1':
        #             mode_flatten.append(1)
        #         elif modality == 'm2':
        #             mode_flatten.append(0)
        #         elif modality == 'm3':
        #             mode_flatten.append(2)
        #         elif modality == 'm4':
        #             mode_flatten.append(3)
        #         else:
        #             raise
        #     mode_flatten = torch.tensor(mode_flatten).to(heter_feature_2d.device).int()
        # import ipdb;ipdb.set_trace()
        mode_flatten = torch.tensor([1 if modality == 'm1' or modality == 'm2' or modality == 'm3' else 0 for modality in agent_modality_list]).to(heter_feature_2d.device).int()
        start_idx = 0
        for i, cav_num in enumerate(record_len):
            mode[i][:cav_num] = mode_flatten[start_idx:start_idx + cav_num]
            start_idx += cav_num

        x, mask = Regroup(heter_feature_2d, record_len, L)

        # print('x.shape',x.shape)
        # print('pairwise_t_matrix.shape',pairwise_t_matrix.shape)
        # print('record_len',record_len)
        # print('L',L)
        # print('mask',mask)
        # print('mode',mode)
        # print(counting_dict)
        # B, L, C, H, W
        x = self.fusion_net(x, pairwise_t_matrix, mode,
                            record_len, mask).squeeze(1)
        
        psm, rm, dm = self.decoder(x.unsqueeze(1), mode, use_upsample=False)

        output_dict.update({
            'cls_preds': psm,
            'reg_preds': rm,
            'dir_preds': dm
        })

        return output_dict
