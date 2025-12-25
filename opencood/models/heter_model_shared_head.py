# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# In this heterogeneous version, feature align start before backbone.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion#, DiscoFusion, V2VNetFusion, V2XViTFusion, When2commFusion, warp_feature
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
from opencood.models.comm_modules.where2comm import Communication
import torchvision
from opencood.models.sub_modules.codebook import ChannelCompressor
from opencood.models.sub_modules.codebook import UMGMQuantizer
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

class HeterModelSharedhead(nn.Module):
    def __init__(self, args):
        super(HeterModelSharedhead, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list

        self.ego_modality = args['ego_modality']
        self.stage2_added_modality = args.get('stage2_added_modality', None)

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()
        #communication
        self.naive_communication = Communication(args['fusion_args']['communication'])
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
            Aligner building
            """
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))
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
        """
        self.backbone = ResNetBEVBackbone(args['fusion_backbone'])
        self.fusion_net = nn.ModuleList()

        for i in range(len(args['fusion_backbone']['layer_nums'])):
            if args['fusion_method'] == "max":
                self.fusion_net.append(MaxFusion())
            if args['fusion_method'] == "att":
                self.fusion_net.append(AttFusion(args['att']['feat_dim'][i]))

        """
        Codebook
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
        
        self.model_train_init()\

        if 'freeze_for_codebook' in args and args['freeze_for_codebook']:
            self.freeze_for_codebook()            
        if 'freeze_codebook' in args and args['freeze_codebook']:
            print('freezecodebook')
            self.freeze_codebook()

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

    def freeze_codebook(self):
        for p in self.multi_channel_compressor.parameters():
            p.requires_grad_(False)
            
    def freeze_for_codebook(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.shrink_conv.parameters():
            p.requires_grad_(False)
        for p in self.cls_head_single.parameters():
            p.requires_grad_(False)
        for p in self.reg_head_single.parameters():
            p.requires_grad_(False)
        for p in self.dir_head_single.parameters():
            p.requires_grad_(False)
        for p in self.cls_head.parameters():
            p.requires_grad_(False)
        for p in self.reg_head.parameters():
            p.requires_grad_(False)
        for p in self.dir_head.parameters():
            p.requires_grad_(False)        
        for p in self.fusion_net.parameters():
            p.requires_grad_(False)                
        for p in self.naive_communication.parameters():
            p.requires_grad_(False)                      
        for p in self.naive_communication.parameters():
            p.requires_grad_(False)
        for modality_name in self.modality_name_list:
            for p in eval(f"self.encoder_{modality_name}").parameters():
                p.requires_grad_(False)
            for p in eval(f"self.backbone_{modality_name}").parameters():
                p.requires_grad_(False)
            for p in eval(f"self.aligner_{modality_name}").parameters():
                p.requires_grad_(False)

    def model_train_init(self):
        if self.stage2_added_modality is None:
            return
        """
        In stage 2, only ONE modality's aligner is trainable.
        We first fix all modules, and set the aligner trainable.
        """
        # fix all modules
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        
        # unfix aligner module
        for p in eval(f"self.aligner_{self.stage2_added_modality}").parameters():
            p.requires_grad_(True)
        eval(f"self.aligner_{self.stage2_added_modality}").apply(unfix_bn)


    def forward(self, data_dict):
        output_dict = {}
        agent_modality_list = data_dict['agent_modality_list']
        t_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        record_len = data_dict['record_len'] 
        # print(agent_modality_list)

        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature


        """
        Spatial Align
        """
        if len(self.ego_modality) == 2 and eval(f"self.aligner_{self.ego_modality}.spatial_align_flag"):
            """
            e.g. 
                self.ego_modality = 'm4'. The length of string is 2.
                record_len = [2, 3, 3]
                agent_modality_list = [m4, m1, m4, m4, m1, m4, m1, m1].
                ego_idx_in_allcav = [0, 2, 5]
                student_idx_in_allcav = [0, 2, 3, 5]
                ego_idx_in_student = [0, 1, 3]

                in eval, ego can be non-student. only student ego will perform spatial align.
            """
            record_len_list = record_len.detach().cpu().numpy().tolist()
            ego_idx_in_allcav = [0] + np.cumsum(record_len_list)[:-1].tolist()

            student_idx_in_allcav = [i for i, x in enumerate(agent_modality_list) if x == self.ego_modality]
            student_ego_idx_in_allcav = [i for i in ego_idx_in_allcav if i in student_idx_in_allcav]

            student_ego_idx_in_student = [student_idx_in_allcav.index(x) for x in student_ego_idx_in_allcav]
            student_ego_idx_in_ego = [ego_idx_in_allcav.index(x) for x in student_ego_idx_in_allcav]
            spatial_align_sample =  student_ego_idx_in_ego # within a batch, which samples will perform spatial align? only ego is student.

            if(len(spatial_align_sample)):
                student_feature = modality_feature_dict[self.ego_modality][student_ego_idx_in_student] # ego in all student modality

                counting_dict = {modality_name: 0 for modality_name in self.modality_name_list}
                teacher_feature_2d_list = [] # the same shape as 'feature', but replace ego modality feature with all zero.

                ego_aligner = eval(f"self.aligner_{self.ego_modality}")

                for modality_name in agent_modality_list:
                    feat_idx = counting_dict[modality_name]
                    agent_feature = modality_feature_dict[modality_name][feat_idx]
                    if modality_name in ego_aligner.teacher:
                        teacher_feature_2d_list.append(agent_feature)
                    else:
                        teacher_feature_2d_list.append(torch.zeros_like(agent_feature, device=agent_feature.device))
                    counting_dict[modality_name] += 1
                
                # unify the feature shape
                _, _, H, W = modality_feature_dict[self.ego_modality].shape
                target_H = int(H*eval(f"self.crop_ratio_H_{self.ego_modality}"))
                target_W = int(W*eval(f"self.crop_ratio_W_{self.ego_modality}"))
                crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                teacher_feature_2d_list = [crop_func(feat) for feat in teacher_feature_2d_list]

                teacher_feature_full = torch.stack(teacher_feature_2d_list)
                teacher_feature = MaxFusion()(teacher_feature_full, record_len, t_matrix)
                teacher_feature = torchvision.transforms.CenterCrop((H, W))(teacher_feature)
                teacher_feature = teacher_feature[spatial_align_sample]
                
                modality_feature_dict[self.ego_modality][student_ego_idx_in_student] = \
                    ego_aligner.spatail_align(student_feature, teacher_feature,
                            (eval(f"self.xdist_{self.ego_modality}"), eval(f"self.ydist_{self.ego_modality}")))

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


        """
        Codebook Part
        """
        '''get_feature_flag = True
        if get_feature_flag:
            save_path = "/GPFS/rhome/sifeiliu/OpenCOODv2/opencood/logs/feature_folder/"
            print("get feature", num)
            torch.save(heter_feature_2d, os.path.join(save_path,'feature%d.pt' % (num)))
            #torch.save(record_len, os.path.join(save_path,'record_len%d.pt' % (num)))
            #codebook_loss = 0.0
        '''
        
        N, C, H, W = heter_feature_2d.shape
        #print("heter_feature_2d_shape: ", heter_feature_2d.shape)
        # import pdb
        # pdb.set_trace()
        if self.multi_channel_compressor_flag:
            #print("------------Codebook information------------")
            heter_feature_2d_gt = heter_feature_2d.clone()
            heter_feature_2d = heter_feature_2d.permute(0, 2, 3, 1).contiguous().view(-1, C)
            heter_feature_2d, _, _, codebook_loss = self.multi_channel_compressor(heter_feature_2d)
            heter_feature_2d = heter_feature_2d.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()
            heter_feature_2d_gt_split = self.regroup(heter_feature_2d_gt, record_len)
            shape_num = 0
            #print("record_len: ", record_len)
            #print("heter_feature_2d_gt_shape: ", heter_feature_2d_gt.shape)
            #print("heter_feature_2d_gt_split: ", len(heter_feature_2d_gt_split))
            for index in range(len(heter_feature_2d_gt_split)):
                #print("heter_feature_2d_gt_split_shape: ", heter_feature_2d_gt_split[index].shape)
                #print(heter_feature_2d_gt_split[index].shape[0])
                #print(shape_num)
                heter_feature_2d[shape_num] = heter_feature_2d_gt_split[index][0]
                shape_num = shape_num + heter_feature_2d_gt_split[index].shape[0]
                
            #print("heter_feature_2d_shape: ", heter_feature_2d.shape)
            output_dict.update({'codebook_loss': codebook_loss})
            #print('codebook_loss', codebook_loss)
            #print("------------Codebook information------------")

        """
        Feature Fusion (multiscale).

        we omit self.backbone's first layer.
        """

        feature_list = [heter_feature_2d]
        for i in range(1, len(self.fusion_net)):
            heter_feature_2d = self.backbone.get_layer_i_feature(heter_feature_2d, layer_i=i)
            feature_list.append(heter_feature_2d)

        batch_confidence_maps = self.regroup(cls_preds_before_fusion, record_len)

        
        #print('confidencemap:{}'.format(batch_confidence_maps[0].size()))
        #print("fusion2")
        _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
        for i in range(len(feature_list)):
              #print(x.size())
              feature_list[i] = feature_list[i] * communication_masks
              communication_masks = F.max_pool2d(communication_masks, kernel_size=2)
        fused_feature_list = []
        for i, fuse_module in enumerate(self.fusion_net):
            fused_feature_list.append(fuse_module(feature_list[i], record_len, t_matrix))
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list)

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

       

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds,
                            'comm_rates': communication_rates})
        # print("comm_rates",output_dict['comm_rates'])
        return output_dict


    def single_forward(self, data_dict):
        output_dict = {}
        print("single_forward")
        agent_modality_list = data_dict['agent_modality_list']
        t_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        record_len = data_dict['record_len']
        
        cav_id_list = data_dict['cav_id_list']
        # print(agent_modality_list)

        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature


        """
        Spatial Align
        """
        if len(self.ego_modality) == 2 and eval(f"self.aligner_{self.ego_modality}.spatial_align_flag"):
            """
            e.g. 
                self.ego_modality = 'm4'. The length of string is 2.
                record_len = [2, 3, 3]
                agent_modality_list = [m4, m1, m4, m4, m1, m4, m1, m1].
                ego_idx_in_allcav = [0, 2, 5]
                student_idx_in_allcav = [0, 2, 3, 5]
                ego_idx_in_student = [0, 1, 3]

                in eval, ego can be non-student. only student ego will perform spatial align.
            """
            record_len_list = record_len.detach().cpu().numpy().tolist()
            ego_idx_in_allcav = [0] + np.cumsum(record_len_list)[:-1].tolist()

            student_idx_in_allcav = [i for i, x in enumerate(agent_modality_list) if x == self.ego_modality]
            student_ego_idx_in_allcav = [i for i in ego_idx_in_allcav if i in student_idx_in_allcav]

            student_ego_idx_in_student = [student_idx_in_allcav.index(x) for x in student_ego_idx_in_allcav]
            student_ego_idx_in_ego = [ego_idx_in_allcav.index(x) for x in student_ego_idx_in_allcav]
            spatial_align_sample =  student_ego_idx_in_ego # within a batch, which samples will perform spatial align? only ego is student.

            if(len(spatial_align_sample)):
                student_feature = modality_feature_dict[self.ego_modality][student_ego_idx_in_student] # ego in all student modality

                counting_dict = {modality_name: 0 for modality_name in self.modality_name_list}
                teacher_feature_2d_list = [] # the same shape as 'feature', but replace ego modality feature with all zero.

                ego_aligner = eval(f"self.aligner_{self.ego_modality}")

                for modality_name in agent_modality_list:
                    feat_idx = counting_dict[modality_name]
                    agent_feature = modality_feature_dict[modality_name][feat_idx]
                    if modality_name in ego_aligner.teacher:
                        teacher_feature_2d_list.append(agent_feature)
                    else:
                        teacher_feature_2d_list.append(torch.zeros_like(agent_feature, device=agent_feature.device))
                    counting_dict[modality_name] += 1
                
                # unify the feature shape
                _, _, H, W = modality_feature_dict[self.ego_modality].shape
                target_H = int(H*eval(f"self.crop_ratio_H_{self.ego_modality}"))
                target_W = int(W*eval(f"self.crop_ratio_W_{self.ego_modality}"))
                crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                teacher_feature_2d_list = [crop_func(feat) for feat in teacher_feature_2d_list]

                teacher_feature_full = torch.stack(teacher_feature_2d_list)
                teacher_feature = MaxFusion()(teacher_feature_full, record_len, t_matrix)
                teacher_feature = torchvision.transforms.CenterCrop((H, W))(teacher_feature)
                teacher_feature = teacher_feature[spatial_align_sample]
                
                modality_feature_dict[self.ego_modality][student_ego_idx_in_student] = \
                    ego_aligner.spatail_align(student_feature, teacher_feature,
                            (eval(f"self.xdist_{self.ego_modality}"), eval(f"self.ydist_{self.ego_modality}")))

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

        N, _, H, W = heter_feature_2d.shape

        heter_feature_2d = warp_affine_simple(heter_feature_2d,t_matrix[0,0,:N,:,:],(H,W))

        feature_list = [heter_feature_2d]
        for i in range(1, len(self.fusion_net)):
            heter_feature_2d = self.backbone.get_layer_i_feature(heter_feature_2d, layer_i=i)
            feature_list.append(heter_feature_2d)
        
        fused_feature = self.backbone.decode_multiscale_feature(feature_list)
        

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)
        
        """
        Single supervision
        """
        
        # cls_preds_before_fusion = self.cls_head_single(heter_feature_2d)
        # reg_preds_before_fusion = self.reg_head_single(heter_feature_2d)
        # dir_preds_before_fusion = self.dir_head_single(heter_feature_2d)
        cls_preds_before_fusion = self.cls_head(fused_feature)
        reg_preds_before_fusion = self.reg_head(fused_feature)
        dir_preds_before_fusion = self.dir_head(fused_feature)


        for i in range(len(cav_id_list)):
            cav_dict={}
            cav_id = cav_id_list[i]
            if i==0:
                cav_id='ego'
            cav_dict.update({'cls_preds': cls_preds_before_fusion[i].unsqueeze(0),
                            'reg_preds': reg_preds_before_fusion[i].unsqueeze(0),
                            'dir_preds': dir_preds_before_fusion[i].unsqueeze(0)})
            output_dict.update({cav_id: cav_dict})
        
        return output_dict