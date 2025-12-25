# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from opencood.models.sub_modules.feature_alignnet_modules import (
    SCAligner,
    Res1x1Aligner,
    Res3x3Aligner,
    Res3x3Aligner,
    CBAM,
    ConvNeXt,
    AttBlock,
    FANet,
    SDTAAgliner,
)
# from opencood.models.sub_modules.deformable_attention import (
#     deformable_attn_pytorch,
#     LearnedPositionalEncoding,
#     constant_init,
#     xavier_init,
# )
# from opencood.models.sub_modules.deformable_attention import (
#     compute_mixed_cis,
#     compute_axial_cis,
#     init_2d_freqs,
#     init_t_xy,
#     apply_rotary_emb,
#     init_random_2d_freqs,
# )
# from positional_encodings.torch_encodings import PositionalEncoding2D, PositionalEncodingPermute2D, Summer
from opencood.models.sub_modules.resblock import ResNetModified,Bottleneck
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DoubleConv
import warnings
import numpy as np


class BaseAdapter(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        in_cav_lidar_range,
        out_cav_lidar_range,
        in_feature_shape,
        out_feature_shape,
        **kwargs,
    ):
        # TODO: For now, we ignore the z axis, not sure if we need to consider it.
        # We also assume that the agent is always at the center of the lidar range
        super(BaseAdapter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_cav_lidar_range = in_cav_lidar_range    #输入agent的感知range
        self.out_cav_lidar_range = out_cav_lidar_range  #输出agent的感知range
        self.in_feature_shape = in_feature_shape        #输入的feature map size
        self.out_feature_shape = out_feature_shape      #输出的feature map size

        in_range_lidar = np.array(
            [in_cav_lidar_range[3] - in_cav_lidar_range[0], in_cav_lidar_range[4] - in_cav_lidar_range[1]]
        ) # 输入感知range的长度
        out_range_lidar = np.array(
            [out_cav_lidar_range[3] - out_cav_lidar_range[0], out_cav_lidar_range[4] - out_cav_lidar_range[1]]
        ) # 输出感知range的长度
        self.ratio = out_range_lidar / in_range_lidar #感知range比

        in_range_feature = np.array([in_feature_shape[1], in_feature_shape[0]])
        out_range_feature = np.array([out_feature_shape[1], out_feature_shape[0]])

        in_ratio = in_range_feature / in_range_lidar    #  input 1/voxel_size_in
        out_ratio = out_range_feature / out_range_lidar #  output 1/voxel_size_out
        self.feat_ratio = out_ratio / in_ratio          #  voxel_size_in / voxel_size_out

        left_new = in_cav_lidar_range[0] * in_ratio[0] * self.feat_ratio[0]     #-> 计算如果in_range 在out_voxel_size下的feature map coord
        right_new = in_cav_lidar_range[3] * in_ratio[0] * self.feat_ratio[0]
        top_new = in_cav_lidar_range[1] * in_ratio[1] * self.feat_ratio[1]
        bottom_new = in_cav_lidar_range[4] * in_ratio[1] * self.feat_ratio[1]

        left_target = out_cav_lidar_range[0] * out_ratio[0]
        right_target = out_cav_lidar_range[3] * out_ratio[0]
        top_target = out_cav_lidar_range[1] * out_ratio[1]
        bottom_target = out_cav_lidar_range[4] * out_ratio[1]

        left_diff = left_new - left_target #### -170.66667 
        right_diff = right_target - right_new
        top_diff = top_new - top_target
        bottom_diff = bottom_target - bottom_new

        self.pad = nn.ZeroPad2d((round(left_diff), round(right_diff), round(top_diff), round(bottom_diff)))

        self.init_adapter()

    def init_adapter(self):
        raise NotImplementedError

    def forward(self, ego_feature, protocol_feature):
        raise NotImplementedError


class AdapterIdentity(BaseAdapter):
    def __init__(self, **kwargs):
        super(AdapterIdentity, self).__init__(**kwargs)

    def init_adapter(self):
        self.resize = nn.Upsample(
            scale_factor=(self.out_channels / self.in_channels, self.feat_ratio[0], self.feat_ratio[1]),
            mode="trilinear",
        )

    def forward(self, ego_feature):
        ego_feature = self.resize(ego_feature.unsqueeze(1))
        ego_feature = ego_feature.squeeze(1)
        return ego_feature
class AdapterIdentity_resize(BaseAdapter):
    def __init__(self, **kwargs):
        super(AdapterIdentity_resize, self).__init__(**kwargs)

    def init_adapter(self):
        self.resize = nn.Upsample(
            scale_factor=(self.out_channels / self.in_channels, self.feat_ratio[0], self.feat_ratio[1]),
            mode="trilinear",
        )

    def forward(self, ego_feature):
        ego_feature = self.resize(ego_feature.unsqueeze(1))
        ego_feature = ego_feature.squeeze(1)
        ego_feature = self.pad(ego_feature)
        print(self.pad)
        # add for 0.3m and 100.8 range 
        if tuple(ego_feature.shape[2:]) != tuple(self.out_feature_shape):
            print('need to interpolate to match beacause pad must be int',ego_feature.shape[2:],self.out_feature_shape)
            ego_feature = F.interpolate(ego_feature, size=self.out_feature_shape, mode='bilinear', align_corners=False)

        return ego_feature

from opencood.models.sub_modules.feature_alignnet_modules import ConvNeXtBlock
from typing import List, Dict, Optional

class FiLMHead(nn.Module):
    """
    为了兼容不同层的通道数，MLP 的输出长度固定为 2 * max_channels，具体层用到多少就 slice 多少。
    """
    def __init__(self, emb_dim: int, max_channels: int, use_tanh: bool = True, mlp_hidden_mult: int = 4):
        super().__init__()
        hidden = max(emb_dim * mlp_hidden_mult, emb_dim)
        self.max_channels = max_channels
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2 * max_channels)
        self.act_out = nn.Tanh() if use_tanh else nn.Identity()
        
        # 初始化最后一层：gamma 部分接近 1，beta 部分接近 0
        with torch.no_grad():
            # 假设输出前半是 gamma，后半是 beta
            self.fc2.weight.data.fill_(0.0)
            self.fc2.bias.data.fill_(0.0)
            # 让 gamma 初始为 1
            self.fc2.bias.data[:max_channels] = 1.0  # gamma 初始化为 1
            # beta 初始化为 0（默认）

    def forward(self, x: torch.Tensor, emb: torch.Tensor, channels: int):
        gb_full = self.fc2(F.gelu(self.fc1(emb)))
        gb_full = self.act_out(gb_full)

        gamma_full, beta_full = gb_full.chunk(2, dim=0)
        gamma = gamma_full[:channels].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        beta  = beta_full[:channels].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # # 在 forward 中加 debug
        # print(f"gamma: mean={gamma.mean().item():.3f}, std={gamma.std().item():.3f}")
        # print(f"beta:  mean={beta.mean().item():.3f}, std={beta.std().item():.3f}")
        return gamma * x + beta

class ConvNeXtFiLM(nn.Module):
    def __init__(self, dim: int, num_of_blocks: int, kernel_size: int = 7, deform: bool = False):
        super().__init__()
        self.dim = dim
        self.num_of_blocks = num_of_blocks
        blocks = []
        for _ in range(num_of_blocks):
            blocks.append(ConvNeXtBlock(dim, kernel_size=kernel_size, deform=deform))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor,
                film_heads: List[FiLMHead],
                emb_list_for_blocks: List[torch.Tensor]):
        assert len(emb_list_for_blocks) == self.num_of_blocks, \
            f"需要 {self.num_of_blocks} 个 block 对应的 embedding，实际 {len(emb_list_for_blocks)}"
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            x = film_heads[i](x, emb_list_for_blocks[i], channels=self.dim)
        return x

class AdapterConvNextFiLM(BaseAdapter):
    """
    submodule_args:
      - dim: int（ConvNeXt/hidden 通道）
      - num_of_blocks: int
      - film_args: {
          'emb_dim': int,
          'modality_map': List[str],      # 例如 ['m2','m3']
        }
    """
    def __init__(self, submodule_args, **kwargs):
        self.submodule_args = submodule_args
        self.film_args = submodule_args.get("film_args", None)
        super(AdapterConvNextFiLM, self).__init__(**kwargs)

    def init_adapter(self):
        # 上采样模块
        self.resize = nn.Upsample(
            scale_factor=(self.feat_ratio[0], self.feat_ratio[1]),
            mode="bilinear"
        )

        # 通道设定
        hidden_channel: int = self.submodule_args.get("dim", 64)
        kernel_size: int = self.submodule_args.get("kernel_size", 7)
        num_of_blocks: int = self.submodule_args["num_of_blocks"]
        deform: bool = self.submodule_args.get("deform", False)

        # 主干与通道变换
        self.channel_convert1 = nn.Conv2d(self.in_channels, hidden_channel, kernel_size=1)
        self.conv = ConvNeXtFiLM(dim=hidden_channel,
                                 num_of_blocks=num_of_blocks,
                                 kernel_size=kernel_size,
                                 deform=deform)
        self.channel_convert2 = nn.Conv2d(hidden_channel, self.out_channels, kernel_size=1)
        self.smoothing = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

        # ===== FiLM：一个共享的 MLP，所有层位置共用 =====
        emb_dim: int = self.film_args['emb_dim']
        use_tanh: bool = self.film_args.get('use_tanh', True)
        mlp_hidden_mult: int = self.film_args.get('mlp_hidden_mult', 4)

        # 计算 FiLM 涉及的最大通道数（用于共享 MLP 输出上限）
        self.num_film_sites = 1 + num_of_blocks + 1
        film_max_channels = max(hidden_channel, self.out_channels)
        self.film_heads = nn.ModuleList([
            FiLMHead(
                emb_dim=emb_dim,
                max_channels=film_max_channels,
                use_tanh=use_tanh,
                mlp_hidden_mult=mlp_hidden_mult
            )
            for _ in range(self.num_film_sites)
        ])

        # ===== 分层 embedding 池（模态 × 位置(层) × emb_dim）=====
        modality_map: List[str] = self.film_args.get('modality_map', None)
        if modality_map is None or len(modality_map) == 0:
            raise ValueError("film_args['modality_map'] 必须提供且非空，例如 ['m2','m3']")
        self.modality_map: List[str] = modality_map
        self.modality2idx: Dict[str, int] = {m: i for i, m in enumerate(modality_map)}

        # 参数形状：[M, L, D]
        M = len(self.modality_map)
        L = self.num_film_sites
        D = emb_dim
        # 更随机一些的初始化
        self.emb_table = nn.Parameter(torch.empty(M, L, D))
        nn.init.normal_(self.emb_table, mean=0.0, std=0.02)

    def _gather_layer_embeddings(self, agent_modality_list: List[str]) -> List[torch.Tensor]:
        """
        根据 batch 的模态列表，取出该 batch 在每个 FiLM 位置的 embedding。
        返回长度为 num_film_sites 的列表，每个元素形状 [B, emb_dim]
        """
        device = self.emb_table.device
        try:
            modality_idx = torch.tensor([self.modality2idx[m] for m in agent_modality_list],
                                        device=device, dtype=torch.long)  # [B]
        except KeyError as e:
            raise ValueError(f"未知模态 {e.args[0]}，可选：{self.modality_map}")

        # 选出该 batch 的所有位置 embedding => [B, L, D]
        emb_batch_all = self.emb_table.index_select(dim=0, index=modality_idx)
        # 拆分为 L 个 [B, D]
        return [emb_batch_all[:, s, :] for s in range(self.num_film_sites)]

    def forward(self, ego_feature: torch.Tensor, mod_name: str):
        """
        ego_feature: [B, C_in, H, W]
        agent_modality_list: List[str]，长度 = B，比如 ['m2','m3','m2']
        """
        x = ego_feature * self.submodule_args.get("early_scale", 1.0)
        if not self.submodule_args.get("late_upsample", False):
            x = self.resize(x)

        print(f"using emb {self.modality2idx[mod_name]}")
        layer_embeds = self.emb_table[self.modality2idx[mod_name]]

        # === pre-FiLM：通道映射到 hidden 后，立刻调制 ===
        x = self.channel_convert1(x)                                   # [B, hidden, H, W]
        x = self.film_heads[0](x, layer_embeds[0], channels=x.size(1))

        # === ConvNeXt 主干 + 每层 FiLM ===
        conv_block_embs = layer_embeds[1:-1]                           # 长度 = num_of_blocks
        x = self.conv(x, film_heads=self.film_heads[1:-1], emb_list_for_blocks=conv_block_embs)

        # === 输出映射 + 平滑 + post-FiLM ===
        x = self.channel_convert2(x)                                   # [B, C_out, H, W]
        x = self.smoothing(x)
        x = self.film_heads[-1](x, layer_embeds[-1], channels=x.size(1))

        if self.submodule_args.get("late_upsample", False):
            x = self.resize(x)

        return x


class AdapterConvNext(BaseAdapter):
    def __init__(self, submodule_args, **kwargs):
        self.submodule_args = submodule_args
        super(AdapterConvNext, self).__init__(**kwargs)

    def init_adapter(self):
        self.resize = nn.Upsample(scale_factor=(self.feat_ratio[0], self.feat_ratio[1]), mode="bilinear") #  voxel_size_in / voxel_size_out  -> in_lidar_range/voxel_size_out 
        hiddle_channel = self.submodule_args.get("dim", 64)
        self.channel_convert1 = nn.Conv2d(self.in_channels, hiddle_channel, kernel_size=1)
        self.conv = ConvNeXt(self.submodule_args)
        self.channel_convert2 = nn.Conv2d(hiddle_channel, self.out_channels, kernel_size=1)
        self.smoothing = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, ego_feature):
        ego_feature = ego_feature * self.submodule_args.get("early_scale", 1.0)
        if not self.submodule_args.get("late_upsample", False):
            ego_feature = self.resize(ego_feature)
        ego_feature = self.channel_convert1(ego_feature)
        protocol_feature = self.conv(ego_feature)
        protocol_feature = self.channel_convert2(protocol_feature)
        if self.submodule_args.get("late_upsample", False):
            protocol_feature = self.resize(protocol_feature)
        # protocol_feature = self.pad(protocol_feature)

        return protocol_feature
    
class AdapterConvNext_resize(BaseAdapter):
    def __init__(self, submodule_args, **kwargs):
        self.submodule_args = submodule_args
        super(AdapterConvNext_resize, self).__init__(**kwargs)

    def pad_feature_map(self,feature_map, target_size):

        # 获取当前特征图的大小
        import ipdb;ipdb.set_trace()
        current_size = feature_map.shape[2]
        # in voxel size:
        in_range_lidar = np.array(
            [self.in_cav_lidar_range[3] - self.in_cav_lidar_range[0],
              self.in_cav_lidar_range[4] - self.in_cav_lidar_range[1]]
        ) 
        in_voxel_size = current_size / in_range_lidar
          # 假设 feature_map 是 (C, H, W) 形式，H = W
        
        # 计算每边需要填充的像素数
        pad_size = (target_size - current_size) // 2
        extra_pad = (target_size - current_size) % 2  # 如果有多余的像素，处理不均匀的部分
        
        # 填充上下左右的零，torch.nn.functional.pad 使用 (left, right, top, bottom) 的顺序
        padded_map = F.pad(feature_map, 
                        (pad_size, pad_size + extra_pad, pad_size, pad_size + extra_pad), 
                        mode='constant', value=0)
        
        return padded_map

    def init_adapter(self):
        
        self.resize = nn.Upsample(scale_factor=(self.feat_ratio[0], self.feat_ratio[1]), mode="bilinear") #  voxel_size_in / voxel_size_out
        hiddle_channel = self.submodule_args.get("dim", 64)
        self.channel_convert1 = nn.Conv2d(self.in_channels, hiddle_channel, kernel_size=1)
        self.conv = ConvNeXt(self.submodule_args)
        self.channel_convert2 = nn.Conv2d(hiddle_channel, self.out_channels, kernel_size=1)
        self.smoothing = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, ego_feature):
        # from matplotlib import pyplot as plt
        # fig,ax = plt.subplots(2,figsize=(5,5*2))
        # import ipdb;ipdb.set_trace()
        protocol_feature = ego_feature * self.submodule_args.get("early_scale", 1.0)
        if not self.submodule_args.get("late_upsample", False):
            protocol_feature = self.resize(protocol_feature)
        protocol_feature = self.channel_convert1(protocol_feature)
        protocol_feature = self.conv(protocol_feature)
        protocol_feature = self.channel_convert2(protocol_feature)
        if self.submodule_args.get("late_upsample", False):
            protocol_feature = self.resize(protocol_feature)
        protocol_feature = self.pad(protocol_feature)
        print(self.pad)
        # add for 0.3m and 100.8 range 
        if tuple(protocol_feature.shape[2:]) != tuple(self.out_feature_shape):
            print('need to interpolate to match beacause pad must be int',protocol_feature.shape[2:],self.out_feature_shape)
            protocol_feature = F.interpolate(protocol_feature, size=self.out_feature_shape, mode='bilinear', align_corners=False)

        return protocol_feature


class AdapterAtt(BaseAdapter):
    def __init__(self, submodule_args, **kwargs):
        self.submodule_args = submodule_args
        super(AdapterAtt, self).__init__(**kwargs)

    def init_adapter(self):
        
        self.resize = nn.Upsample(scale_factor=(self.feat_ratio[0], self.feat_ratio[1]), mode="bilinear")
        hiddle_channel = self.submodule_args.get("dim", 64)
        self.channel_convert1 = nn.Conv2d(self.in_channels, hiddle_channel, kernel_size=1)
        self.channel_convert2 = nn.Conv2d(hiddle_channel, self.out_channels, kernel_size=1)
        self.smoothing = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        
        self.patch_size = self.submodule_args.get("patch_size", 16)
        stride = self.patch_size
        if self.submodule_args.get("late_upsample", False):
            H = self.in_feature_shape[0]
            W = self.in_feature_shape[1]
        else:
            H = self.out_feature_shape[0]
            W = self.out_feature_shape[1]
        num_heads = self.submodule_args.get("num_heads", 4)
        depth = self.submodule_args.get("depth", 3)
        
        self.patch_embed = nn.Conv2d(hiddle_channel, hiddle_channel, kernel_size=self.patch_size, stride=stride)
        self.pos_embed = nn.Parameter(torch.zeros(1, H // self.patch_size * W // self.patch_size, hiddle_channel))
        self.blocks = nn.ModuleList([
            AttBlock(hiddle_channel, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hiddle_channel)
        

    def forward(self, ego_feature):
        ego_feature = ego_feature * self.submodule_args.get("early_scale", 1.0)
        if not self.submodule_args.get("late_upsample", False):
            ego_feature = self.resize(ego_feature)
        ego_feature = self.channel_convert1(ego_feature)
        # protocol_feature = self.conv(ego_feature)
        
        
        B, C, H, W = ego_feature.shape
        # Patch embedding
        x = self.patch_embed(ego_feature)  # shape: (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # shape: (B, num_patches, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for blk in self.blocks: x = blk(x)
        
        x = self.norm(x)
        
        # Reshape back to image-like tensor
        # import pdb; pdb.set_trace()
        x = x.transpose(1, 2).reshape(B, C, H // self.patch_size, W // self.patch_size)
        protocol_feature = nn.functional.interpolate(x, scale_factor=self.patch_size, mode='bilinear', align_corners=False)
        
        protocol_feature = self.channel_convert2(protocol_feature)
        if self.submodule_args.get("late_upsample", False):
            protocol_feature = self.resize(protocol_feature)
        # protocol_feature = apply_gaussian_smoothing(protocol_feature, 5, 1.0)
        # protocol_feature = self.pad(protocol_feature)

        return protocol_feature

class AdapterConv(BaseAdapter):
    """
    Conv Adapter
    Upsample the protocol feature to the same size of ego feature with a convolutional layer.

    """

    def __init__(self, **kwargs):
        super(AdapterConv, self).__init__(**kwargs)

    def init_adapter(self):
        self.resize = nn.Upsample(scale_factor=(self.feat_ratio[0], self.feat_ratio[1]), mode="bilinear")
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, ego_feature):

        ego_feature = self.resize(ego_feature)
        protocol_feature = self.conv(ego_feature)

        protocol_feature = self.pad(protocol_feature)

        return protocol_feature


class AdapterFC(BaseAdapter):
    """
    Conv Adapter
    Upsample the protocol feature to the same size of ego feature with a convolutional layer.

    eg:
    in_H_voxel = 128
    in_W_voxel = 128
    in_H = 512
    in_W = 512
    out_H_voxel = 192
    out_W_voxel = 192
    out_H = 1536
    out_W = 1536

    in_factor_H = 512 / 128 = 4
    (out_H_voxel - in_H_voxel) / 2 = (192 - 128) / 2 = 32

    """

    def __init__(self, **kwargs):
        super(AdapterFC, self).__init__(**kwargs)

    def init_adapter(self):
        self.resize = nn.Upsample(scale_factor=(self.feat_ratio[0], self.feat_ratio[1]), mode="bilinear")

        self.weights = nn.Parameter(
            torch.Tensor(self.in_feature_shape[0], self.in_feature_shape[1], self.in_channels, self.out_channels)
        )
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        self.biases = nn.Parameter(torch.zeros(self.in_feature_shape[0], self.in_feature_shape[1], self.out_channels))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.biases, -bound, bound)

    def forward(self, ego_feature):
        ego_feature = self.resize(ego_feature)
        B, C, H, W = ego_feature.shape
        ego_feature = ego_feature.reshape(B, C, H, W)

        # Perform pixel-wise fully connected operation
        protocol_feature = torch.einsum("bchw,hwco->bhwo", ego_feature, self.weights) + self.biases.view(
            H, W, self.out_channels
        )
        protocol_feature = protocol_feature.permute(0, 3, 1, 2)

        protocol_feature = self.pad(protocol_feature)
        # TODO: make sure this is on the correct dimension
        # protocol_feature = protocol_feature[:, :, self.crop_top:self.crop_bottom, self.crop_left:self.crop_right]

        return protocol_feature


class DeformableSpatialAttentionLayer(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        num_heads=8,
        num_points=4,
        dropout=0.1,
        scale_ratio=1.0,
    ):
        super(DeformableSpatialAttentionLayer, self).__init__()
        if out_channel % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {out_channel} and {num_heads}")
        self.dim_per_head = out_channel // num_heads

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        # however, CUDA is not available in this implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(self.dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
                "However, CUDA is not available in this implementation."
            )

        assert self.dim_per_head % 2 == 0, "embed_dims must be divisible by 2"

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_heads = num_heads
        self.num_points = num_points
        self.dropout = nn.Dropout(dropout)
        self.sampling_offsets = nn.Linear(self.in_channel, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(self.in_channel, num_heads * num_points)
        self.value_proj = nn.Linear(self.in_channel, self.out_channel)
        self.output_proj = nn.Linear(self.out_channel, self.out_channel)
        self.scale_ratio = scale_ratio
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, 1, self.num_points, 1)
        )

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        # TODO: Remove the hard coded half precision
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        identity=None,
        device="cuda",
        dtype=torch.half,
        spatial_shapes=None,
    ):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                (bs, num_query, embed_dims).
            value (Tensor): The value tensor with shape
                (bs, num_query, embed_dims).
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            spatial_shapes (tuple): Spatial shape of features (h, w).

        Returns:
             Tensor: forwarded results with shape [bs, num_query, embed_dims].
        """

        bs, num_query, embed_dims = query.shape
        h, w = spatial_shapes

        if identity is None:
            identity = query

        if query_pos is not None:
            query = query + query_pos
        value = self.value_proj(value)
        # if key_padding_mask is not None:
        #     value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.reshape(
            bs, num_query, self.num_heads, self.dim_per_head
        )  # bs, num_query, num_heads, embed_dims//num_heads
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_points, 2
        )  # bs, num_query, num_heads, num_points, 2
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_points
        )  # bs, num_query, num_heads, num_points
        attention_weights = attention_weights.softmax(-1).to(
            dtype
        )  # TODO: attention_weights.softmax(-1) changed attention_weights from half to float

        reference_points = self.get_reference_points(
            h, w, bs=bs, scale_ratio=self.scale_ratio, device=device, dtype=dtype
        )  # bs, num_query, 2
        offset_normalizer = torch.Tensor([w, h]).to(device).to(dtype)
        sampling_locations = reference_points[:, :, None, None, :] + sampling_offsets / offset_normalizer

        output = self.output_proj(deformable_attn_pytorch(value, (h, w), sampling_locations, attention_weights))
        # return self.dropout(output) + identity
        return self.dropout(output) + identity

    def get_reference_points(self, H, W, bs=1, scale_ratio=1.0, device="cuda", dtype=torch.half):
        if type(scale_ratio) != tuple:
            scale_ratio = (scale_ratio, scale_ratio)

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
        )
        # TODO: make sure the x y dimension is correct
        # import ipdb;ipdb.set_trace()
        ref_y = ref_y.reshape(-1)[None] / H * torch.tensor(scale_ratio[0][0]).to(device)
        ref_x = ref_x.reshape(-1)[None] / W * torch.tensor(scale_ratio[1][0]).to(device)
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1)
        return ref_2d

class AdapterDSA(BaseAdapter):
    """Deformable Spatial Attention."""

    def __init__(
        self,
        in_channels,
        out_channels,
        class_channels,
        in_cav_lidar_range,
        out_cav_lidar_range,
        in_feature_shape,
        out_feature_shape,
        submodule_args,
        **kwargs,
    ):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.class_channels = class_channels
        self.H, self.W = in_feature_shape
        self.outH = out_feature_shape[0]
        self.outW = out_feature_shape[1]
        self.n_layers = submodule_args.get("n_layers", 8)
        self.num_heads = submodule_args.get("num_heads", 8)
        self.num_points = submodule_args.get("num_points", 4)
        self.dropout = submodule_args.get("dropout", 0.1)
        self.rope_mixed = submodule_args.get("rope_mixed", True)  # TODO: False does not work
        self.rope_theta = submodule_args.get("rope_theta", 10.0)
        self.dim_per_head = out_channels // self.num_heads

        super(AdapterDSA, self).__init__(
            in_channels,
            out_channels,
            in_cav_lidar_range,
            out_cav_lidar_range,
            in_feature_shape,
            out_feature_shape,
            **kwargs,
        )

    def init_adapter(self):
        # self.bev_embedding = nn.Embedding(self.outH * self.outW, self.out_channels)
        self.resize = nn.Upsample(scale_factor=(self.feat_ratio[0], self.feat_ratio[1]), mode="bilinear")
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(self.in_channels, self.class_channels, kernel_size=1)
        # self.positional_encoding = LearnedPositionalEncoding(self.out_channels // 2, self.outH, self.outW)
        self.in_pos_embed_sinusoidal = PositionalEncodingPermute2D(self.in_channels)
        self.in_pos_scale_factor = nn.Parameter(torch.ones(1) / 30.0)
        self.out_pos_embed_sinusoidal = PositionalEncoding2D(self.out_channels)
        self.out_pos_scale_factor = nn.Parameter(torch.ones(1) / 30.0)

        self.attention_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attention_layers.append(
                DeformableSpatialAttentionLayer(
                    self.in_channels, self.out_channels, self.num_heads, self.num_points, self.dropout, self.ratio
                )
            )

    def forward(self, ego_feature):
        B, C, H, W = ego_feature.shape
        # ego_feature = self.in_pos_embed_sinusoidal(ego_feature)
        ego_feature_embed = self.in_pos_embed_sinusoidal(ego_feature)
        ego_feature = ego_feature_embed * self.in_pos_scale_factor + ego_feature
        key = ego_feature.view(B, C, H * W).transpose(1, 2)  # B, H*W, C
        device, dtype = key.device, key.dtype
        # query = self.bev_embedding.weight.to(device).reshape(1, self.outH*self.outW, self.out_channels).repeat(B, 1, 1) # B, outH*outW, outC
        # query = self.resize(ego_feature).view(B, C, self.outH*self.outW).transpose(1, 2) # B, H*W, C
        query = (
            self.conv(self.resize(ego_feature)).view(B, self.out_channels, self.outH * self.outW).transpose(1, 2)
        )  # B, H*W, C
        # pos_mask = torch.zeros((B, self.outH, self.outW), device=device).to(dtype)
        # query_pos = self.positional_encoding(pos_mask).to(dtype).flatten(2).transpose(1,2) # B, H*W, C

        output = query

        # print(self.in_pos_scale_factor, self.out_pos_scale_factor)

        for layer in range(self.n_layers):

            output = output.reshape(B, self.outH, self.outW, self.out_channels)
            output_embed = self.out_pos_embed_sinusoidal(output)
            output = output_embed * self.out_pos_scale_factor + output
            output = output.reshape(B, self.outH * self.outW, self.out_channels)

            output = self.attention_layers[layer](
                query=output,
                key=None,  # key is not used
                value=key,
                identity=output,
                # query_pos=query_pos, # query_pos is not used for rotary position embedding
                device=device,
                dtype=dtype,
                spatial_shapes=(self.outH, self.outW),
            )

        output = output.transpose(1, 2).reshape(B, self.out_channels, self.outH, self.outW)
        output = self.conv_out(output)
        protocol_feature = self.pad(output)
        return protocol_feature
    
class Adapterres(ResNetBEVBackbone):
    def __init__(self,model_cfg):
        super().__init__(model_cfg)
        self.model_cfg = model_cfg
        self.resnet = ResNetModified(Bottleneck, 
                                        self.model_cfg['layer_nums'],
                                        self.model_cfg['layer_strides'],
                                        self.model_cfg['num_filters'],
                                        inplanes = model_cfg.get('inplanes', 64),
                                        groups=32,
                                        width_per_group=4)
        self.shrink = DoubleConv(384,16,kernel_size=3,stride=1,padding=1)
    def forward(self, ego_feature):
        feature_list = self.get_multiscale_feature(ego_feature)
        feature = self.decode_multiscale_feature(feature_list)
        feature = self.shrink(feature)
        return feature

class Adapter(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_name = args["core_method"]
        self.model_name = model_name

        if model_name == "adapterfc":
            self.adapter = AdapterFC(**args["args"])
        elif model_name == "adapterconv":
            self.adapter = AdapterConv(**args["args"])
        elif model_name == "adapterconvnext":
            self.adapter = AdapterConvNext(**args["args"])
        elif model_name == "adapterconvnextfilm":
            self.adapter = AdapterConvNextFiLM(**args["args"])
        elif model_name == "adapterconvnext_resize":
            self.adapter = AdapterConvNext_resize(**args["args"])
        elif model_name == "adapterdsa":
            self.adapter = AdapterDSA(**args["args"])
        elif model_name == "identity":
            self.adapter = AdapterIdentity(**args["args"])
        elif model_name == "identity_resize":
            self.adapter = AdapterIdentity_resize(**args["args"])
        elif model_name == "adapteratt":
            self.adapter = AdapterAtt(**args["args"])
        elif model_name == "adapterresnet":
            self.adapter = Adapterres(args)
        else:
            raise NotImplementedError(f"Adapter {model_name} not implemented")

    def forward(self, x, mod_name='m0'):
        if self.model_name == "adapterconvnextfilm":
            return self.adapter(x, mod_name)
        else:
            return self.adapter(x)


class Reverter(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_name = args["core_method"]

        if model_name == "adapterfc":
            self.reverter = AdapterFC(**args["args"])
        elif model_name == "adapterconv":
            self.reverter = AdapterConv(**args["args"])
        elif model_name == "adapterconvnext":
            self.reverter = AdapterConvNext(**args["args"])
        elif model_name == "adapterconvnext_resize":
            self.reverter = AdapterConvNext_resize(**args["args"])
        elif model_name == "adapterdsa":
            self.reverter = AdapterDSA(**args["args"])
        elif model_name == "identity":
            self.reverter = AdapterIdentity(**args["args"])
        elif model_name == "identity_resize":
            self.reverter = AdapterIdentity_resize(**args["args"])
        elif model_name == "adapteratt":
            self.reverter = AdapterAtt(**args["args"])
        else:
            raise NotImplementedError(f"Reverter {model_name} not implemented")

    def forward(self, x):
        return self.reverter(x)
