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
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
import torchvision
from opencood.models.sub_modules.codebook import ChannelCompressor
from opencood.models.sub_modules.codebook import UMGMQuantizer
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

import os

class HeterPyramidSharedHead(nn.Module):
    def __init__(self, args, train_flag=True):
        super(HeterPyramidSharedHead, self).__init__()
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

            # Load encoder ckpt
            if self.train_flag:
                model_path = model_setting['model_path']
                if model_path != '':
                    local_rank = torch.distributed.get_rank()
                    checkpoint = torch.load(model_setting['model_path'],map_location='cuda:{}'.format(local_rank))
                    encoder_state_dict = {k.replace(f'encoder_{modality_name}.', ''): v for k, v in checkpoint.items() if f'encoder_{modality_name}' in k}
                    backbone_state_dict = {k.replace(f'backbone_{modality_name}.', ''): v for k, v in checkpoint.items() if f'backbone_{modality_name}' in k}
                    encoder_model = getattr(self, f"encoder_{modality_name}")
                    backbone_model = getattr(self, f"backbone_{modality_name}")
                    encoder_model.load_state_dict(encoder_state_dict, strict=False)
                    backbone_model.load_state_dict(backbone_state_dict, strict=False)
                    print(f"Encoder and Backbone for modality {modality_name} loaded successfully.")

            # """
            # Aligner building
            # """
            # setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))
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
        single supervision
        """
        self.supervise_single = False
        if args.get("supervise_single", False):
            self.supervise_single = True
            in_head_single = args['in_head_single']
            setattr(self, f'cls_head_single', nn.Conv2d(in_head_single, args['anchor_number'], kernel_size=1))
            setattr(self, f'reg_head_single', nn.Conv2d(in_head_single, args['anchor_number'] * 7, kernel_size=1))
            setattr(self, f'dir_head_single', nn.Conv2d(in_head_single, args['anchor_number'] *  args['dir_args']['num_bins'], kernel_size=1))


        """
        Fusion, by default multiscale fusion: 
        Note the input of PyramidFusion has downsampled 2x. (SECOND required)
        """
        self.pyramid_backbone = PyramidFusion(args['fusion_backbone'])

        """
        Codebook
        """
        # self.multi_channel_compressor_flag = False
        # if 'multi_channel_compressor' in args and args['multi_channel_compressor']:
        #     print('multi_channel_compressor_flag')
        #     self.multi_channel_compressor_flag = True

        # channel = 64
        # p_rate = 0.0
        # seg_num = args['codebook']['seg_num']
        # if args['codebook']['r'] == 1:
        #     dict_size = [args['codebook']['dict_size']]
        # elif args['codebook']['r'] == 2:
        #     dict_size = [args['codebook']['dict_size'], args['codebook']['dict_size']]
        # else:
        #     dict_size = [args['codebook']['dict_size'], args['codebook']['dict_size'], args['codebook']['dict_size']]
        # self.multi_channel_compressor = UMGMQuantizer(channel, seg_num, dict_size, p_rate,
        #                   {"latentStageEncoder": lambda: nn.Linear(channel, channel), "quantizationHead": lambda: nn.Linear(channel, channel),
        #                    "latentHead": lambda: nn.Linear(channel, channel), "restoreHead": lambda: nn.Linear(channel, channel),
        #                    "dequantizationHead": lambda: nn.Linear(channel, channel), "sideHead": lambda: nn.Linear(channel, channel)})
        # print("codebook:", self.multi_channel_compressor_flag)
        # print("seg_num: ", seg_num)        
        # print("dict_size: ", args['codebook']['dict_size'])

        """
        Shrink header
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        """
        Shared Heads
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        
        # compressor will be only trainable
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'],
                                              args['compressor']['compress_ratio'])

        self.model_train_init()
        
        if 'stage2' in args and args['stage2']:
            self.stage2()

        # check again which module is not fixed.
        check_trainable_module(self)
        print('----------- Training Parameters -----------')
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)
        print('----------- Training Parameters -----------')


    def regroup(self, x, record_len):
        #print(x)
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
            
    def stage2(self):
        # for p in self.multi_channel_compressor.parameters():
        #     p.requires_grad_(False)
        # for p in self.pyramid_backbone.parameters():
        #     p.requires_grad_(False)
        # for p in self.shrink_conv.parameters():
        #     p.requires_grad_(False)
        # for p in self.cls_head_single.parameters():
        #     p.requires_grad_(False)
        # for p in self.reg_head_single.parameters():
        #     p.requires_grad_(False)
        # for p in self.dir_head_single.parameters():
        #     p.requires_grad_(False)
        # for p in self.cls_head.parameters():
        #     p.requires_grad_(False)
        # for p in self.reg_head.parameters():
        #     p.requires_grad_(False)
        # for p in self.dir_head.parameters():
        #     p.requires_grad_(False)
        for modality_name in self.modality_name_list:
            for p in eval(f"self.encoder_{modality_name}").parameters():
                p.requires_grad_(False)
            for p in eval(f"self.backbone_{modality_name}").parameters():
                p.requires_grad_(False)

    def model_train_init(self):
        # if compress, only make compressor trainable
        if self.compress:
            # freeze all
            # self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def forward(self, data_dict, save_path='', i=0):
        output_dict = {'pyramid': 'collab'}
        agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        print(agent_modality_list)
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            # feature = eval(f"self.aligner_{modality_name}")(feature)
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
        
        if save_path!='':
            os.makedirs(save_path, exist_ok=True)
            for k in range(heter_feature_2d.shape[0]):
                # 对每辆车的特征图进行通道平均
                avg_feature = heter_feature_2d[k].mean(dim=0)  # 沿着通道维度求均值，得到形状 [256, 256]
                # 规范化到 [0, 1] 之间
                avg_feature = (avg_feature - avg_feature.min()) / (avg_feature.max() - avg_feature.min())
                # 将其转为 [0, 255] 的整数范围，符合图像保存的标准
                img = avg_feature.mul(255).byte()
                # 使用 torchvision 保存为图片
                save_image = torchvision.transforms.ToPILImage()(img)  # 转换为 PIL 图片
                save_image.save(f"{save_path}/{i}_{k}_{agent_modality_list[k]}.jpg")  # 保存图片
                print(f"Saved image for car {k} at {save_path}/{i}_{k}_{agent_modality_list[k]}.jpg")
        
        """
        Codebook Part
        """
        # N, C, H, W = heter_feature_2d.shape
        # #print("heter_feature_2d_shape: ", heter_feature_2d.shape)
        # # import pdb
        # # pdb.set_trace()
        # if self.multi_channel_compressor_flag:
        #     #print("------------Codebook information------------")
        #     heter_feature_2d_gt = heter_feature_2d.clone()
        #     heter_feature_2d = heter_feature_2d.permute(0, 2, 3, 1).contiguous().view(-1, C)
        #     heter_feature_2d, _, _, codebook_loss = self.multi_channel_compressor(heter_feature_2d)
        #     heter_feature_2d = heter_feature_2d.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()

        #     if save_path!='':
        #         for k in range(heter_feature_2d.shape[0]):
        #             # 对每辆车的特征图进行通道平均
        #             avg_feature = heter_feature_2d[k].mean(dim=0)  # 沿着通道维度求均值，得到形状 [256, 256]
        #             # 规范化到 [0, 1] 之间
        #             avg_feature = (avg_feature - avg_feature.min()) / (avg_feature.max() - avg_feature.min())
        #             # 将其转为 [0, 255] 的整数范围，符合图像保存的标准
        #             img = avg_feature.mul(255).byte()
        #             # 使用 torchvision 保存为图片
        #             save_image = torchvision.transforms.ToPILImage()(img)  # 转换为 PIL 图片
        #             save_image.save(f"{save_path}/{i}_{k}_{agent_modality_list[k]}_code.jpg")  # 保存图片
        #             print(f"Saved image for car {k} at {save_path}/{i}_{k}_{agent_modality_list[k]}_code.jpg")

        #     heter_feature_2d_gt_split = self.regroup(heter_feature_2d_gt, record_len)
        #     shape_num = 0
        #     #print("record_len: ", record_len)
        #     #print("heter_feature_2d_gt_shape: ", heter_feature_2d_gt.shape)
        #     #print("heter_feature_2d_gt_split: ", len(heter_feature_2d_gt_split))
        #     for index in range(len(heter_feature_2d_gt_split)):
        #         #print("heter_feature_2d_gt_split_shape: ", heter_feature_2d_gt_split[index].shape)
        #         #print(heter_feature_2d_gt_split[index].shape[0])
        #         #print(shape_num)
        #         heter_feature_2d[shape_num] = heter_feature_2d_gt_split[index][0]
        #         shape_num = shape_num + heter_feature_2d_gt_split[index].shape[0]
                
        #     #print("heter_feature_2d_shape: ", heter_feature_2d.shape)
        #     output_dict.update({'codebook_loss': codebook_loss})
        #     #print('codebook_loss', codebook_loss)
        #     #print("------------Codebook information------------")
        
        """
        Single supervision
        """
        if self.supervise_single:
            cls_preds_before_fusion = self.cls_head_single(heter_feature_2d)
            reg_preds_before_fusion = self.reg_head_single(heter_feature_2d)
            dir_preds_before_fusion = self.dir_head_single(heter_feature_2d)
            output_dict.update({'cls_preds_single': cls_preds_before_fusion,
                                'reg_preds_single': reg_preds_before_fusion,
                                'dir_preds_single': dir_preds_before_fusion})

        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

        # heter_feature_2d is downsampled 2x
        # add croping information to collaboration module
        
        fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
                                                heter_feature_2d,
                                                record_len, 
                                                affine_matrix, 
                                                agent_modality_list, 
                                                self.cam_crop_info
                                            )

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        # _, bbox_temp = self.generate_predicted_boxes(cls_preds, reg_preds)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        
        output_dict.update({'occ_single_list': 
                            occ_outputs})

        return output_dict

    def generate_predicted_boxes(self, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        
        batch, H, W, code_size = box_preds.size()   ## code_size 表示的是预测的尺寸
        
        box_preds = box_preds.reshape(batch, H*W, code_size)

        batch_reg = box_preds[..., 0:2]
        # batch_hei = box_preds[..., 2:3] 
        # batch_dim = torch.exp(box_preds[..., 3:6])
        h = box_preds[..., 3:4] * self.out_size_factor * self.voxel_size[0]
        w = box_preds[..., 4:5] * self.out_size_factor * self.voxel_size[1]
        l = box_preds[..., 5:6] * self.out_size_factor * self.voxel_size[2]
        batch_dim = torch.cat([h,w,l], dim=-1)
        batch_hei = box_preds[..., 2:3] * self.out_size_factor * self.voxel_size[2] + self.cav_lidar_range[2]

        batch_rots = box_preds[..., 6:7]
        batch_rotc = box_preds[..., 7:8]

        rot = torch.atan2(batch_rots, batch_rotc)

        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device)
        xs = xs.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device)

        xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
        ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

        xs = xs * self.out_size_factor * self.voxel_size[0] + self.cav_lidar_range[0]   ## 基于feature_map 的size求解真实的坐标
        ys = ys * self.out_size_factor * self.voxel_size[1] + self.cav_lidar_range[1]


        batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, rot], dim=2)
        # batch_box_preds = batch_box_preds.reshape(batch, H, W, batch_box_preds.shape[-1])
        # batch_box_preds = batch_box_preds.permute(0, 3, 1, 2).contiguous()

        # batch_box_preds_temp = torch.cat([xs, ys, batch_hei, batch_dim, rot], dim=1)
        # box_preds = box_preds.permute(0, 3, 1, 2).contiguous()

        # batch_cls_preds = cls_preds.view(batch, H*W, -1)
        return cls_preds, batch_box_preds