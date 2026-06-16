""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

# Modifications by Xiangbo Gao <xiangbogaobarry@gmail.com>
# this model for align to codebook(only infer) or train adapter for codebook 

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
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion,PyramidFusion_codebook
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.fuse_modules.adapter import Adapter, Reverter
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
from opencood.models.sub_modules.codebook import UMGMQuantizer
import importlib
import torchvision
from opencood.utils.codebook_utils import gumbelSoftmax

class CollabWC2CInfer(nn.Module):
    def __init__(self, args, train_flag=True):
        super(CollabWC2CInfer, self).__init__()
        self.args = args
        
        ignored_modality = set(args.get("ignored_modality", []))
        modality_name_list = list(args.keys() - ignored_modality)
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()]
        self.modality_name_list = sorted(modality_name_list) # add sorted for ddp 
        print(self.modality_name_list)

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()
        self.fix_modules = ['multi_channel_compressor','pyramid_backbone', 'cls_head', 'reg_head', 'dir_head']
        self.save_codes_dataset = args['save_codes_dataset']
        if self.save_codes_dataset:
            print('this stage is getting codes dataset\n'*10)

        self.cam_crop_info = {} 
        self.save_for_vis = False
        self.only_return_detection = args['only_return_detection']
        self.only_return_class = args['only_return_class']

        self.class_feature_codebook = args.get('class_feature_codebook',True)
        if self.class_feature_codebook:
            print('now is use sparse feature for infer')
        else:
            print('now is use dense feature for infer')
        

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
            print(f"encoder_{modality_name}")
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
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))
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
            if model_setting.get("fix_encoder", False):
                self.fix_modules += [f"encoder_{modality_name}", f"backbone_{modality_name}"]

            """
            Codebook building
            """
            self.multi_channel_compressor_flag = False
            if 'multi_channel_compressor' in model_setting and model_setting['multi_channel_compressor']:
                print('multi_channel_compressor_flag')
                self.multi_channel_compressor_flag = True
            if self.multi_channel_compressor_flag:
                channel = 64
                p_rate = 0.0
                seg_num = model_setting['codebook']['seg_num']
                if model_setting['codebook']['r'] == 1:
                    dict_size = [model_setting['codebook']['dict_size']]
                elif model_setting['codebook']['r'] == 2:
                    dict_size = [model_setting['codebook']['dict_size'], model_setting['codebook']['dict_size']]
                else:
                    dict_size = [model_setting['codebook']['dict_size'], model_setting['codebook']['dict_size'], model_setting['codebook']['dict_size']]
                setattr(self, f"multi_channel_compressor_{modality_name}", UMGMQuantizer(channel, seg_num, dict_size, p_rate,
                                {"latentStageEncoder": lambda: nn.Linear(channel, channel), "quantizationHead": lambda: nn.Linear(channel, channel),
                                "latentHead": lambda: nn.Linear(channel, channel), "restoreHead": lambda: nn.Linear(channel, channel),
                                "dequantizationHead": lambda: nn.Linear(channel, channel), "sideHead": lambda: nn.Linear(channel, channel)}))
                print("codebook:", self.multi_channel_compressor_flag)
                print("seg_num: ", seg_num)        
                print("dict_size: ", model_setting['codebook']['dict_size'])

            """
            Fusion, by default multiscale fusion: 
            Note the input of PyramidFusion has downsampled 2x. (SECOND required)
            """
            setattr(
                self,
                f"pyramid_backbone_{modality_name}",
                PyramidFusion(model_setting["fusion_backbone"]),
            )

            """For feature transformation"""
            self.H = (self.cav_range[4] - self.cav_range[1])
            self.W = (self.cav_range[3] - self.cav_range[0])
            self.fake_voxel_size = 1

            """
            Shrink header
            """
            self.shrink_flag = False
            if 'shrink_header' in model_setting:
                setattr(self, f"shrink_flag_{modality_name}", True)
                setattr(
                    self,
                    f"shrink_conv_{modality_name}",
                    DownsampleConv(model_setting["shrink_header"]),
                )
            """
            code2code
            """
            if model_setting.get("adapter",None) is not None:  # Never equip adapter and reverter for m0
                setattr(self, f"code2code_{modality_name}", Adapter(model_setting["adapter"]))

            """
            no Shared Heads
            """
            setattr(self,f"cls_head_{modality_name}",nn.Conv2d(
                    model_setting["in_head"],
                    model_setting["anchor_number"],
                    kernel_size=1,
                ),
            )
            setattr(
                self,
                f"reg_head_{modality_name}",
                nn.Conv2d(
                    model_setting["in_head"],
                    7 * model_setting["anchor_number"],
                    kernel_size=1,
                ),
            )
            if model_setting.get("dir_args", None):
                setattr(
                    self,
                    f"dir_head_{modality_name}",
                    nn.Conv2d(
                        model_setting["in_head"],
                        model_setting["dir_args"]["num_bins"] * model_setting["anchor_number"],
                        kernel_size=1,
                    ),
                )
        
        # compressor will be only trainable
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'],
                                              args['compressor']['compress_ratio'])

        self.model_train_init()
        # check again which module is not fixed.
        check_trainable_module(self)


    def model_train_init(self):
        # this model only for infer
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)


    def forward(self, data_dict):
        output_dict = {'pyramid': 'collab'}
        
        agent_modality_list = data_dict['agent_modality_list'] 
        record_len = data_dict['record_len'] 
        # Filter out the modality that is not ready to inference
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
        
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}


        ego_modality = agent_modality_list[0]
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name) #test
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)

            """
            Crop/Padd camera feature map.
            """

            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    feature = crop_func(feature)
            """
            Codebook Part
            """            
            N, C, H, W = feature.shape
            if self.multi_channel_compressor_flag:
                feature = feature.permute(0, 2, 3, 1).contiguous().view(-1, C)
                if self.class_feature_codebook or modality_name == ego_modality:
                    quantizeds, codes = eval(f'self.multi_channel_compressor_{modality_name}.quantized')(feature)
                    feature = eval(f'self.multi_channel_compressor_{modality_name}.get_feature')(quantizeds)
                # feature, codes, _, _ = eval(f'self.multi_channel_compressor_{modality_name}')(feature)
                feature = feature.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()
                # feature_vis = feature
                if modality_name != ego_modality:
                    feature_code2code = eval(f'self.code2code_{modality_name}')(feature) # m2 to ego codebook feature
                    feature_code = F.softmax(feature_code2code,dim=1)
                    
                    _, C1, _, _ = feature_code.shape
                    feature_code = feature_code.permute(0, 2, 3, 1).contiguous().view(-1,1,C1)
                    quantizeds_index = torch.argmax(feature_code,dim=2)
                    quantizeds_infer = torch.zeros(quantizeds_index.shape[0], 1, C1).to(quantizeds_index.device)
                    quantizeds_infer.scatter_(dim=2, index=quantizeds_index.unsqueeze(-1), value=1)
                    feature_ego_infer = eval(f'self.multi_channel_compressor_{ego_modality}.get_feature')([quantizeds_infer])
                    feature = feature_ego_infer.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()

            modality_feature_dict[modality_name] = feature

        
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
        
        fused_feature, occ_outputs = eval(f'self.pyramid_backbone_{ego_modality}').forward_collab(
                                                heter_feature_2d,
                                                record_len, 
                                                affine_matrix, 
                                                agent_modality_list, 
                                                self.cam_crop_info
                                            )
        

        fused_feature = eval(f'self.shrink_conv_{ego_modality}')(fused_feature)

        cls_preds = eval(f'self.cls_head_{ego_modality}')(fused_feature)
        reg_preds = eval(f'self.reg_head_{ego_modality}')(fused_feature)
        dir_preds = eval(f'self.dir_head_{ego_modality}')(fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        
        output_dict.update({'occ_single_list': 
                            occ_outputs})
        
        return output_dict
