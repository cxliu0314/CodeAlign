from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class NaiveDecoder(nn.Module):
    """
    A Naive decoder implementation

    Parameters
    ----------
    params: dict

    Attributes
    ----------
    num_ch_dec : list
        The decoder layer channel numbers.

    num_layer : int
        The number of decoder layers.

    input_dim : int
        The channel number of the input to
    """
    def __init__(self, params):
        super(NaiveDecoder, self).__init__()

        self.num_ch_dec = params['num_ch_dec']
        self.num_layer = params['num_layer']
        self.input_dim = params['input_dim']

        assert len(self.num_ch_dec) == self.num_layer

        # decoder
        self.convs = OrderedDict()
        for i in range(self.num_layer-1, -1, -1):
            # upconv_0
            num_ch_in = self.input_dim if i == self.num_layer-1\
                else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]

            self.convs[("upconv", i, 0)] = nn.Conv2d(
                num_ch_in, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 0)] = nn.BatchNorm2d(num_ch_out)
            self.convs[("relu", i, 0)] = nn.ReLU(True)

            # upconv_1
            self.convs[("upconv", i, 1)] = nn.Conv2d(
                num_ch_out, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 1)] = nn.BatchNorm2d(num_ch_out)
            self.convs[("relu", i, 1)] = nn.ReLU(True)
        self.decoder = nn.ModuleList(list(self.convs.values()))

    @staticmethod
    def upsample(x):
        """Upsample input tensor by a factor of 2
        """
        # why nearest
        return F.interpolate(x, scale_factor=2, mode="nearest")

    def forward(self, x, use_upsample=True):
        """
        Upsample to

        Parameters
        ----------
        x : torch.tensor
            The bev bottleneck feature, shape: (B, L, C1, H, W)

        Returns
        -------
        Output features with (B, L, C2, H, W)
        """
        b, l, c, h, w = x.shape
        x = rearrange(x, 'b l c h w -> (b l) c h w')

        for i in range(self.num_layer-1, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = self.convs[("norm", i, 0)](x)
            x = self.convs[("relu", i, 0)](x)
            if use_upsample:
                x = self.upsample(x)

            x = self.convs[("upconv", i, 1)](x)
            x = self.convs[("norm", i, 1)](x)
            x = self.convs[("relu", i, 1)](x)

        x = rearrange(x, '(b l) c h w -> b l c h w',
                      b=b, l=l)
        return x

class SharedDecoder(nn.Module):
    def __init__(self, params):
        super(SharedDecoder, self).__init__()
        input_dim = params['num_ch_dec'][0]
        self.decoder = NaiveDecoder(params)

        self.cls_head = nn.Conv2d(input_dim, params['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(input_dim, 7 * params['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(input_dim, 2 * params['anchor_number'],
                                  kernel_size=1)
    
    def forward(self, x, agent_modality_list, use_upsample=True):
        feature = self.decoder(x, use_upsample=use_upsample).squeeze(1)
        psm = self.cls_head(feature)
        rm = self.reg_head(feature)
        dm = self.dir_head(feature)
        return psm, rm, dm

class HeteroDecoder(nn.Module):
    """
    A Naive decoder implementation

    Parameters
    ----------
    params: dict

    Attributes
    ----------
    num_ch_dec : list
        The decoder layer channel numbers.

    num_layer : int
        The number of decoder layers.

    input_dim : int
        The channel number of the input to
    """

    def __init__(self, params):
        super(HeteroDecoder, self).__init__()
        input_dim = params['num_ch_dec'][0]
        self.camera_decoder = NaiveDecoder(params)
        self.lidar_decoder = NaiveDecoder(params)


        self.third_ = False
        self.fourth_ = False
        if params.get('num_types', 2) >= 3:
            self.third_ = True
            self.third_decoder = NaiveDecoder(params)
            self.third_cls_head = nn.Conv2d(input_dim, params['anchor_number'],
                                  kernel_size=1)
            self.third_reg_head = nn.Conv2d(input_dim, 7 * params['anchor_number'],
                                        kernel_size=1)
            self.third_dir_head = nn.Conv2d(input_dim, 2 * params['anchor_number'],
                                        kernel_size=1)
        
        if params.get('num_types', 2) >= 4:
            self.third_ = False
            self.fourth_ = True
            self.fourth_decoder = NaiveDecoder(params)
            self.fourth_cls_head = nn.Conv2d(input_dim, params['anchor_number'],
                                  kernel_size=1)
            self.fourth_reg_head = nn.Conv2d(input_dim, 7 * params['anchor_number'],
                                        kernel_size=1)
            self.fourth_dir_head = nn.Conv2d(input_dim, 2 * params['anchor_number'],
                                        kernel_size=1)

        self.camera_cls_head = nn.Conv2d(input_dim, params['anchor_number'],
                                  kernel_size=1)
        self.camera_reg_head = nn.Conv2d(input_dim, 7 * params['anchor_number'],
                                  kernel_size=1)
        self.camera_dir_head = nn.Conv2d(input_dim, 2 * params['anchor_number'],
                                  kernel_size=1)
        self.lidar_cls_head = nn.Conv2d(input_dim, params['anchor_number'],
                                  kernel_size=1)
        self.lidar_reg_head = nn.Conv2d(input_dim, 7 * params['anchor_number'],
                                  kernel_size=1)
        self.lidar_dir_head = nn.Conv2d(input_dim, 2 * params['anchor_number'],
                                  kernel_size=1)

    def forward(self, x, mode, use_upsample=True):
        """
        Upsample to

        Parameters
        ----------
        x : torch.tensor
            The bev bottleneck feature, shape: (B, L, C1, H, W)
        mode : torch.tensor
            B, L, but only care about ego.
        Returns
        -------
        Output features with (B, L, C2, H, W)
        """
        temp = mode[:, 0]
        ego_mode = torch.clone(temp)
        # ego_mode[:] = 0
        camera_psm, camera_rm, lidar_psm, lidar_rm,  = None, None, None, None
        camera_dm, lidar_dm = None, None 

        third_psm, third_rm, third_dm = None, None, None 
        fourth_psm, fourth_rm, fourth_dm = None, None, None 

        # If there is at least one camera
        if torch.any(ego_mode == 0):
            camera_feature = x[ego_mode == 0, ...]
            camera_feature = self.camera_decoder(camera_feature, use_upsample=use_upsample).squeeze(1)
            camera_psm = self.camera_cls_head(camera_feature)
            camera_rm = self.camera_reg_head(camera_feature)
            camera_dm = self.camera_dir_head(camera_feature)
        # If there is at least one lidar
        if torch.any(ego_mode == 1):
            lidar_feature = x[ego_mode == 1, ...]
            lidar_feature = self.lidar_decoder(lidar_feature, use_upsample=use_upsample).squeeze(1)
            lidar_psm = self.lidar_cls_head(lidar_feature)
            lidar_rm = self.lidar_reg_head(lidar_feature)
            lidar_dm = self.lidar_dir_head(lidar_feature)

        if torch.any(ego_mode == 2):
            third_feature = x[ego_mode == 2, ...]
            third_feature = self.third_decoder(third_feature, use_upsample=use_upsample).squeeze(1)
            third_psm = self.third_cls_head(third_feature)
            third_rm = self.third_reg_head(third_feature)
            third_dm = self.third_dir_head(third_feature)

        if torch.any(ego_mode == 3):
            fourth_feature = x[ego_mode == 3, ...]
            fourth_feature = self.fourth_decoder(fourth_feature, use_upsample=use_upsample).squeeze(1)
            fourth_psm = self.fourth_cls_head(fourth_feature)
            fourth_rm = self.fourth_reg_head(fourth_feature)
            fourth_dm = self.fourth_dir_head(fourth_feature)

        if self.third_:
            psm = self.combine_3_features(camera_psm, lidar_psm, third_psm, ego_mode)
            rm = self.combine_3_features(camera_rm, lidar_rm, third_rm, ego_mode)
            dm = self.combine_3_features(camera_dm, lidar_dm, third_dm, ego_mode)

            return psm, rm, dm

        if self.fourth_:
            psm = self.combine_4_features(camera_psm, lidar_psm, third_psm, fourth_psm, ego_mode)
            rm = self.combine_4_features(camera_rm, lidar_rm, third_rm, fourth_rm, ego_mode)
            dm = self.combine_4_features(camera_dm, lidar_dm, third_dm, fourth_dm, ego_mode)

            return psm, rm, dm

        psm = self.combine_features(camera_psm, lidar_psm, ego_mode)
        rm = self.combine_features(camera_rm, lidar_rm, ego_mode)
        dm = self.combine_features(camera_dm, lidar_dm, ego_mode)

        return psm, rm, dm
    

    def combine_features(self, camera, lidar, ego_mode):
        combined_features = []
        camera_count = 0
        lidar_count = 0
        for i in range(len(ego_mode)):
            if ego_mode[i] == 0:
                combined_features.append(camera[camera_count, ...])
                camera_count += 1
            elif ego_mode[i] == 1:
                combined_features.append(lidar[lidar_count, ...])
                lidar_count += 1
            else:
                raise ValueError(f"Mode but be either 1 or 0 but received "
                                 f"{ego_mode[i]}")
        combined_features = torch.stack(combined_features, dim=0)
        return combined_features
    

    def combine_3_features(self, camera, lidar, third, ego_mode):
        combined_features = []
        camera_count = 0
        lidar_count = 0
        third_count = 0
        for i in range(len(ego_mode)):
            if ego_mode[i] == 0:
                combined_features.append(camera[camera_count, ...])
                camera_count += 1
            elif ego_mode[i] == 1:
                combined_features.append(lidar[lidar_count, ...])
                lidar_count += 1
            elif ego_mode[i] == 2:
                combined_features.append(third[third_count, ...])
                third_count += 1
            else:
                raise ValueError(f"Mode but 0,1,2 received "
                                 f"{ego_mode[i]}")
        combined_features = torch.stack(combined_features, dim=0)
        return combined_features
    
    def combine_4_features(self, camera, lidar, third, fourth, ego_mode):
        combined_features = []
        camera_count = 0
        lidar_count = 0
        third_count = 0
        fourth_count = 0
        for i in range(len(ego_mode)):
            if ego_mode[i] == 0:
                combined_features.append(camera[camera_count, ...])
                camera_count += 1
            elif ego_mode[i] == 1:
                combined_features.append(lidar[lidar_count, ...])
                lidar_count += 1
            elif ego_mode[i] == 2:
                combined_features.append(third[third_count, ...])
                third_count += 1
            elif ego_mode[i] == 3:
                combined_features.append(fourth[third_count, ...])
                fourth_count += 1
            else:
                raise ValueError(f"Mode but 0,1,2 received "
                                 f"{ego_mode[i]}")
        combined_features = torch.stack(combined_features, dim=0)
        return combined_features