""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

# Modifications by Xiangbo Gao <xiangbogaobarry@gmail.com>
# New License for modifications: MIT License

import torch
import cv2
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion, PyramidFusion_codebook
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, CoBEVT
from opencood.models.sub_modules.codebook import UMGMQuantizer
import importlib
import torchvision

class HomoCollabcodebook(nn.Module):
    def __init__(self, args, train_flag=True):
        super(HomoCollabcodebook, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list
        self.sensor_type_dict = OrderedDict()
        self.cam_crop_info = {} 

        # setup each modality model
        modality_name = self.modality_name_list[0]
        model_setting = args[modality_name]
        setattr(self, f"cav_range_{modality_name}", model_setting['lidar_range'])
        sensor_name = model_setting['sensor_type']
        self.sensor_type_dict[modality_name] = sensor_name

        # import model
        encoder_filename = "opencood.models.heter_encoders"
        encoder_lib = importlib.import_module(encoder_filename)
        encoder_class = None
        self.multi_sensor = False
        if isinstance(model_setting['core_method'], str):
            self.multi_sensor = False
            target_model_name = model_setting['core_method'].replace('_', '')
            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls
                    
        elif isinstance(model_setting['core_method'], dict):
            self.multi_sensor = True
            target_model_name_camera = model_setting['core_method']['camera'].replace('_', '')
            target_model_name_lidar = model_setting['core_method']['lidar'].replace('_', '')
            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name_camera.lower():
                    encoder_class_camera = cls
                if name.lower() == target_model_name_lidar.lower():
                    encoder_class_lidar = cls

        """
        Encoder building
        """
        
        if self.multi_sensor:
            
            assert model_setting.get('encoder_args_camera', None) and model_setting.get('encoder_args_lidar', None), \
                "for multi_sensor, encoder_args_camera and encoder_args_lidar should be provided"
            setattr(self, f"encoder_{modality_name}_camera", encoder_class_camera(model_setting['encoder_args_camera']))
            setattr(self, f"encoder_{modality_name}_lidar", encoder_class_lidar(model_setting['encoder_args_lidar']))
            if model_setting['encoder_args_camera'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)
                
        else:
            
            assert model_setting.get('encoder_args', None), "encoder_args should be provided"
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)
            

        """
        Backbone building 
        """
        if model_setting.get('backbone_args', None):
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))
        else:
            setattr(self, f"backbone_{modality_name}", lambda x: {'spatial_features_2d': nn.Identity()(x["spatial_features"])})

        """
        Aligner building
        """
        if model_setting.get('aligner_args', None):
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))
        else:
            setattr(self, f"aligner_{modality_name}", AlignNet({"core_method": "identity"}))
        if "camera" in sensor_name:
            camera_mask_args = model_setting['camera_mask_args']
            setattr(self, f"crop_ratio_W_{modality_name}", 
                    (eval(f"self.cav_range_{modality_name}")[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
            setattr(self, f"crop_ratio_H_{modality_name}", 
                    (eval(f"self.cav_range_{modality_name}")[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
            setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
            setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
            self.cam_crop_info[modality_name] = {
                f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
            }

        """
        Codebook building
        """
        self.multi_channel_compressor_flag = False
        if 'multi_channel_compressor' in args and args['multi_channel_compressor']:
            print('multi_channel_compressor_flag')
            self.multi_channel_compressor_flag = True

        channel = 64
        p_rate = 0.0
        seg_num = args['codebook']['seg_num']
        if args['codebook']['r'] == 1:
            dict_size = [args['codebook']['dict_size']]
        elif args['codebook']['r'] == 2:
            dict_size = [args['codebook']['dict_size'], args['codebook']['dict_size']]
        else:
            dict_size = [args['codebook']['dict_size'], args['codebook']['dict_size'], args['codebook']['dict_size']]
        self.multi_channel_compressor = UMGMQuantizer(channel, seg_num, dict_size, p_rate,
                          {"latentStageEncoder": lambda: nn.Linear(channel, channel), "quantizationHead": lambda: nn.Linear(channel, channel),
                           "latentHead": lambda: nn.Linear(channel, channel), "restoreHead": lambda: nn.Linear(channel, channel),
                           "dequantizationHead": lambda: nn.Linear(channel, channel), "sideHead": lambda: nn.Linear(channel, channel)})
        print("codebook:", self.multi_channel_compressor_flag)
        print("seg_num: ", seg_num)        
        print("dict_size: ", args['codebook']['dict_size'])


        """For feature transformation"""
        setattr(self, f"H_{modality_name}", 
                (eval(f"self.cav_range_{modality_name}")[4] - eval(f"self.cav_range_{modality_name}")[1]))
        setattr(self, f"W_{modality_name}",
                (eval(f"self.cav_range_{modality_name}")[3] - eval(f"self.cav_range_{modality_name}")[0]))
        self.fake_voxel_size = 1

        """
        Fusion, by default multiscale fusion: 
        Note the input of PyramidFusion has downsampled 2x. (SECOND required)
        """
        

        if args['fusion_method'] == 'pyramid':
            setattr(self, f"pyramid_backbone", PyramidFusion(args['fusion_backbone']))
        elif args['fusion_method'] == 'pyramid_codebook':
            setattr(self, f"pyramid_backbone", PyramidFusion_codebook(args['fusion_backbone']))
        elif args['fusion_method'] == "max":
            setattr(self, f"pyramid_backbone", MaxFusion())
        elif args['fusion_method'] == "att":
            setattr(self, f"pyramid_backbone", AttFusion(args['att']['feat_dim']))
        elif args['fusion_method'] == 'cobevt':
            setattr(self, f"pyramid_backbone", CoBEVT(args['cobevt']))
        else:
            raise NotImplementedError(f"Method {args['fusion_method']} not implemented.")
            
        if args['fusion_method'] not in ['pyramid','pyramid_codebook']:
            # other method does not have agent_modality_list and cam_crop_info, neither returning occ_single_list
            pyramid_backbone = getattr(self, f"pyramid_backbone")
            pyramid_backbone.forward_collab = lambda *args: (pyramid_backbone.forward(*args[:3]), [])


        """
        Shrink header
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            setattr(self, "shrink_conv", DownsampleConv(args['shrink_header']))

        """
        Shared Heads
        """
        # By default, point pillar pyramid object detection head 
        self.head_method = args.get('head_method', "point_pillar_pyramid_object_detection_head")
        self.downsample_rate = args.get('downsample_rate', 1)
        
        if self.head_method == "point_pillar_pyramid_object_detection_head":
            
            setattr(self, "cls_head", 
                    nn.Conv2d(args['in_head'], args['anchor_number'], 
                            kernel_size=1))
            setattr(self, "reg_head",
                    nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                            kernel_size=1))
            if args.get('dir_args', None):
                setattr(self, "dir_head",
                        nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                kernel_size=1))            
            
        elif self.head_method == "point_pillar_object_detection_head":
            setattr(self, "cls_head", 
                    nn.Conv2d(args['in_head'], 1, 
                            kernel_size=1))
            setattr(self, "reg_head",
                    nn.Conv2d(args['in_head'], 7,
                            kernel_size=1))
            if args.get('dir_args', None):
                setattr(self, "dir_head",
                        nn.Conv2d(args['in_head'], args['dir_args']['num_bins'],
                                kernel_size=1))   
        
            
        else:
            raise NotImplementedError(f"Head method {self.head_method} not implemented.")
        
        
        # compressor will be only trainable
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            setattr(self, f"compressor_{modality_name}", 
                    NaiveCompressor(args['compressor']['input_dim'],
                                    args['compressor']['compress_ratio']))

        self.model_train_init()
        # check again which module is not fixed.
        # check_trainable_module(self)


    def model_train_init(self):
        # if compress, only make compressor trainable
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def forward(self, data_dict, vis_path='', show_bev=False):
        output_dict = {'pyramid': 'collab'}
        vis_dict = {}
        agent_modality_list = data_dict['agent_modality_list'] 
        record_len = data_dict['record_len'] 
        modality_count_dict = Counter(agent_modality_list)
        assert len(modality_count_dict.keys()) == 1, "Cannot have more than one modality in Homo setting"
        modality_feature_dict = {}

        modality_name = self.modality_name_list[0]
        if modality_name in modality_count_dict:
            
            if self.multi_sensor:
                feature_camera = eval(f"self.encoder_{modality_name}_camera")(data_dict, modality_name, self.multi_sensor)
                """
                Crop/Padd camera feature map.
                """
                if "camera" in self.sensor_type_dict[modality_name]:
                    # should be padding. Instead of masking
                    _, _, H, W = feature_camera.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    feature_camera = crop_func(feature_camera)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}_camera").depth_items
                        })
                        
                
                feature_lidar = eval(f"self.encoder_{modality_name}_lidar")(data_dict, modality_name, self.multi_sensor)
                
                feature = feature_camera + feature_lidar
            else:
                feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)
            

            modality_feature_dict[modality_name] = feature
            
            if not self.multi_sensor:
                """
                Crop/Padd camera feature map.
                """
                if "camera" in self.sensor_type_dict[modality_name]:
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
        if vis_path!='':
            vis_dict['encoder'] = heter_feature_2d.clone()
        
        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], 
                                        eval(f"self.H_{modality_name}"), 
                                        eval(f"self.W_{modality_name}"), 
                                        self.fake_voxel_size)
        """
        Codebook Part
        """
        N, C, H, W = heter_feature_2d.shape
        if self.multi_channel_compressor_flag:
            heter_feature_2d = heter_feature_2d.permute(0, 2, 3, 1).contiguous().view(-1, C)
            heter_feature_2d, code, _, codebook_loss = self.multi_channel_compressor(heter_feature_2d)
            heter_feature_2d = heter_feature_2d.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()

            output_dict.update({'codebook_loss': codebook_loss})
            if vis_path!='':
                vis_dict['codebook'] = heter_feature_2d.clone()
                vis_dict['code'] = code[0].view(-1, H, W, 1).permute(0, 3, 1, 2)
                code = code[0].view(-1, H, W, 1).permute(0, 3, 1, 2).float()
                _, C, H, W = code.shape
                B, L = affine_matrix.shape[:2]
                from opencood.models.fuse_modules.fusion_in_one import regroup
                from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
                single_split_x = regroup(code, record_len)
                single_feature_transed = []
                for b in range(B):
                    N = record_len[b]
                    t_matrix = affine_matrix[b][:N, :N, :, :]
                    single_feature_in_ego = warp_affine_simple(single_split_x[b], t_matrix[0, :, :, :], (H, W), mode='nearest', align_corners=False)
                    single_feature_transed.append(single_feature_in_ego)
                vis_dict['code_warp'] = single_feature_transed[0]


        # heter_feature_2d is downsampled 2x
        # add croping information to collaboration module
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], 
                                        eval(f"self.H_{modality_name}"), 
                                        eval(f"self.W_{modality_name}"), 
                                        self.fake_voxel_size)
        
        # feature, occ_outputs = eval(f"self.pyramid_backbone_{modality_name}").forward_collab(
        #                                         data_dict['label_dict']['gt_static'].unsqueeze(1).transpose(-1,-2).flip(-1),
        #                                         record_len, 
        #                                         affine_matrix, 
        #                                         agent_modality_list, 
        #                                         self.cam_crop_info
        #                                     )
        
        # heter_feature_2d_rotated = heter_feature_2d.transpose(-1,-2).flip(-1)
        feature, occ_outputs = self.pyramid_backbone.forward_collab(
                                                heter_feature_2d,
                                                record_len, 
                                                affine_matrix, 
                                                agent_modality_list, 
                                                self.cam_crop_info
                                            )
        # feature = feature.transpose(-1,-2).flip(-1)
        
        if self.shrink_flag:
            feature = eval("self.shrink_conv")(feature)
        
        if vis_path!='':
            vis_dict['fusion'] = feature
            return vis_dict

        if self.head_method == "bev_seg_head":
            output_dict.update(eval(f"self.head_{modality_name}")(feature))
        elif self.head_method == "seg_head":
            output_dict.update(eval(f"self.head_{modality_name}")(feature))
        else:
            cls_preds = eval("self.cls_head")(feature)
            reg_preds = eval("self.reg_head")(feature)
            if hasattr(self, "dir_head"):
                dir_preds = eval("self.dir_head")(feature)
            else:
                dir_preds = None

            output_dict.update({'cls_preds': cls_preds,
                                'reg_preds': reg_preds,
                                'dir_preds': dir_preds})
        
        output_dict.update({'occ_single_list': 
                            occ_outputs})
        # output_dict.update({'codebook_features_trans': feature_transed})
        

        if show_bev:
            return (output_dict, 
                    feature,
                    heter_feature_2d,
                    None,
                    None,
                    None,
                    None)

        return output_dict
