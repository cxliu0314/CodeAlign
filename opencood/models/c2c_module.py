import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.sub_modules.resblock import ResNetModified, Bottleneck
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
import numpy as np
from opencood.models.sub_modules.feature_alignnet_modules import ConvNeXtBlock
from typing import List, Dict, Optional


class Translator(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_name = args["core_method"]
        self.model_name = model_name

        if model_name == "convnext":
            self.translator = TranslatorConvNext(**args["args"])
        elif model_name == "convnext_film":
            self.translator = TranslatorConvNextFilm(**args["args"])
        elif model_name == "convnext_emb_cat":
            self.translator = TranslatorConvNextEmbCat(**args["args"])
        elif model_name == "convnext_emb_mlp_cat":
            self.translator = TranslatorConvNextEmbMLPCat(**args["args"])
        elif model_name == "convnext_with_modal_heads":
            self.translator = TranslatorConvNextWithModalHeads(**args["args"])
        elif model_name == "convnext_emb_gate":
            self.translator = TranslatorConvNextEmbGate(**args["args"])
        elif model_name == "convnext_emb_gate_multilayer":
            self.translator = TranslatorConvNextEmbGateMultilayer(**args["args"])
        else:
            raise NotImplementedError(f"Adapter {model_name} not implemented")

    def forward(self, x, mod_name='m0'):
        if self.model_name == "convnext":
            return self.translator(x)
        else:
            return self.translator(x, mod_name)
        
class ConvNeXt(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args['dim']
        kernel_size = args.get("kernel_size", 7)
        num_of_blocks = args['num_of_blocks']
        deform = args.get('deform', False)
        model_list = nn.ModuleList()
        for _ in range(num_of_blocks):
            model_list.append(ConvNeXtBlock(dim, kernel_size=kernel_size, deform=deform))
 
        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)

class TranslatorConvNext(nn.Module):
    def __init__(self, submodule_args, in_channels, out_channels, in_cav_lidar_range, out_cav_lidar_range, in_feature_shape, out_feature_shape):
        super(TranslatorConvNext, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.submodule_args = submodule_args
        self.film_args = submodule_args.get("film_args", None)
        
        in_range_feature = np.array([in_feature_shape[1], in_feature_shape[0]])
        out_range_feature = np.array([out_feature_shape[1], out_feature_shape[0]])
        in_range_lidar = np.array([in_cav_lidar_range[3] - in_cav_lidar_range[0], in_cav_lidar_range[4] - in_cav_lidar_range[1]])
        out_range_lidar = np.array([out_cav_lidar_range[3] - out_cav_lidar_range[0], out_cav_lidar_range[4] - out_cav_lidar_range[1]])
        in_ratio = in_range_feature / in_range_lidar    #  input 1/voxel_size_in
        out_ratio = out_range_feature / out_range_lidar #  output 1/voxel_size_out
        self.feat_ratio = out_ratio / in_ratio          #  voxel_size_in / voxel_size_out
        
        self.init_adapter()

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
        return protocol_feature
    
class TranslatorConvNextWithModalHeads(nn.Module):
    def __init__(self, submodule_args, in_channels, out_channels, in_cav_lidar_range, out_cav_lidar_range, in_feature_shape, out_feature_shape):
        super(TranslatorConvNextWithModalHeads, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.submodule_args = submodule_args
        
        in_range_feature = np.array([in_feature_shape[1], in_feature_shape[0]])
        out_range_feature = np.array([out_feature_shape[1], out_feature_shape[0]])
        in_range_lidar = np.array([in_cav_lidar_range[3] - in_cav_lidar_range[0], in_cav_lidar_range[4] - in_cav_lidar_range[1]])
        out_range_lidar = np.array([out_cav_lidar_range[3] - out_cav_lidar_range[0], out_cav_lidar_range[4] - out_cav_lidar_range[1]])
        in_ratio = in_range_feature / in_range_lidar    #  input 1/voxel_size_in
        out_ratio = out_range_feature / out_range_lidar #  output 1/voxel_size_out
        self.feat_ratio = out_ratio / in_ratio          #  voxel_size_in / voxel_size_out
        
        self.init_modality_mapping()
        self.init_adapter()
        self.init_modal_heads()
    
    def init_modality_mapping(self):
        modality_map = self.submodule_args.get('modality_map', None)
        if modality_map is None or len(modality_map) == 0:
            raise ValueError("submodule_args['modality_map'] is needed, e.g., ['m2','m3']")
        self.modality_map = modality_map
        self.modality2idx = {m: i for i, m in enumerate(modality_map)}
    
    def init_adapter(self):
        self.resize = nn.Upsample(scale_factor=(self.feat_ratio[0], self.feat_ratio[1]), mode="bilinear") #  voxel_size_in / voxel_size_out  -> in_lidar_range/voxel_size_out 
        hiddle_channel = self.submodule_args.get("dim", 64)
        self.channel_convert1 = nn.Conv2d(self.in_channels, hiddle_channel, kernel_size=1)
        self.conv = ConvNeXt(self.submodule_args)
        self.smoothing = nn.Conv2d(hiddle_channel, hiddle_channel, kernel_size=3, padding=1)
    
    def init_modal_heads(self):
        hidden_dim = self.submodule_args.get("dim", 64)
        mlp_hidden_dim = self.submodule_args.get("mlp_hidden_dim", 64)
        
        self.modal_heads = nn.ModuleList()
        for _ in range(len(self.modality_map)):
            mlp = nn.Sequential(
                nn.Conv2d(hidden_dim, mlp_hidden_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mlp_hidden_dim, self.out_channels, kernel_size=1)
            )
            self.modal_heads.append(mlp)
    
    def forward(self, ego_feature, mod_name: str):
        if mod_name not in self.modality2idx:
            raise ValueError(f"unknown {mod_name}, available: {self.modality_map}")
        
        ego_feature = ego_feature * self.submodule_args.get("early_scale", 1.0)
        if not self.submodule_args.get("late_upsample", False):
            ego_feature = self.resize(ego_feature)
        
        ego_feature = self.channel_convert1(ego_feature)
        protocol_feature = self.conv(ego_feature)
        protocol_feature = self.smoothing(protocol_feature)
        
        modal_idx = self.modality2idx[mod_name]
        protocol_feature = self.modal_heads[modal_idx](protocol_feature)
        
        if self.submodule_args.get("late_upsample", False):
            protocol_feature = self.resize(protocol_feature)
        
        return protocol_feature
    

class FiLMHead(nn.Module):
    """
    To accommodate the number of channels in different layers, the output length of the MLP is fixed at 2 * max_channels, and only as much as needed for a specific layer is sliced.
    """
    def __init__(self, emb_dim: int, max_channels: int, use_tanh: bool = True, mlp_hidden_mult: int = 4):
        super().__init__()
        hidden = max(emb_dim * mlp_hidden_mult, emb_dim)
        self.max_channels = max_channels
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2 * max_channels)
        self.act_out = nn.Tanh() if use_tanh else nn.Identity()
        
        with torch.no_grad():
            self.fc2.weight.data.fill_(0.0)
            self.fc2.bias.data.fill_(0.0)
            self.fc2.bias.data[:max_channels] = 1.0

    def forward(self, x: torch.Tensor, emb: torch.Tensor, channels: int):
        gb_full = self.fc2(F.gelu(self.fc1(emb)))
        gb_full = self.act_out(gb_full)

        gamma_full, beta_full = gb_full.chunk(2, dim=0)
        gamma = gamma_full[:channels].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        beta = beta_full[:channels].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
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
            f"need {self.num_of_blocks} embedding, now {len(emb_list_for_blocks)}"
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            x = film_heads[i](x, emb_list_for_blocks[i], channels=self.dim)
        return x

class TranslatorConvNextFilm(nn.Module):
    """
    submodule_args:
      - dim: int(ConvNeXt/hidden channel)
      - num_of_blocks: int
      - film_args: {
          'emb_dim': int,
          'modality_map': List[str],
        }
    """
    def __init__(self, submodule_args, in_channels, out_channels, in_cav_lidar_range, out_cav_lidar_range, in_feature_shape, out_feature_shape):
        super(TranslatorConvNextFilm, self).__init__()
        # 必要的输入参数保存
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.submodule_args = submodule_args
        self.film_args = submodule_args.get("film_args", None)
        
        # 计算特征图缩放比例（仅保留必要的计算）
        in_range_feature = np.array([in_feature_shape[1], in_feature_shape[0]])
        out_range_feature = np.array([out_feature_shape[1], out_feature_shape[0]])
        in_range_lidar = np.array([in_cav_lidar_range[3] - in_cav_lidar_range[0], in_cav_lidar_range[4] - in_cav_lidar_range[1]])
        out_range_lidar = np.array([out_cav_lidar_range[3] - out_cav_lidar_range[0], out_cav_lidar_range[4] - out_cav_lidar_range[1]])
        
        in_ratio = in_range_feature / in_range_lidar    #  input 1/voxel_size_in
        out_ratio = out_range_feature / out_range_lidar #  output 1/voxel_size_out
        self.feat_ratio = out_ratio / in_ratio          #  voxel_size_in / voxel_size_out
        
        # 初始化适配器结构
        self.init_adapter()

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

        # FiLM：共享的 MLP
        emb_dim: int = self.film_args['emb_dim']
        use_tanh: bool = self.film_args.get('use_tanh', True)
        mlp_hidden_mult: int = self.film_args.get('mlp_hidden_mult', 4)

        # 计算 FiLM 涉及的最大通道数
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

        # 分层 embedding 池
        modality_map: List[str] = self.film_args.get('modality_map', None)
        if modality_map is None or len(modality_map) == 0:
            raise ValueError("film_args['modality_map'] 必须提供且非空，例如 ['m2','m3']")
        self.modality_map: List[str] = modality_map
        self.modality2idx: Dict[str, int] = {m: i for i, m in enumerate(modality_map)}

        # 参数初始化
        M = len(self.modality_map)
        L = self.num_film_sites
        D = emb_dim
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

        # pre-FiLM：通道映射到 hidden 后，立刻调制
        x = self.channel_convert1(x)                                   # [B, hidden, H, W]
        x = self.film_heads[0](x, layer_embeds[0], channels=x.size(1))

        # ConvNeXt 主干 + 每层 FiLM
        conv_block_embs = layer_embeds[1:-1]                           # 长度 = num_of_blocks
        x = self.conv(x, film_heads=self.film_heads[1:-1], emb_list_for_blocks=conv_block_embs)

        # 输出映射 + 平滑 + post-FiLM
        x = self.channel_convert2(x)                                   # [B, C_out, H, W]
        x = self.smoothing(x)
        x = self.film_heads[-1](x, layer_embeds[-1], channels=x.size(1))

        if self.submodule_args.get("late_upsample", False):
            x = self.resize(x)

        return x

class TranslatorConvNextEmbCat(nn.Module):
    def __init__(self, submodule_args, in_channels, out_channels, in_cav_lidar_range, out_cav_lidar_range, in_feature_shape, out_feature_shape):
        super(TranslatorConvNextEmbCat, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.submodule_args = submodule_args
        self.embedding_dim = submodule_args.get("embedding_dim", 64)  # 模态embedding的维度
        
        if in_feature_shape != out_feature_shape:
            print(f"警告: 输入特征形状 {in_feature_shape} 和输出特征形状 {out_feature_shape} 不匹配，将使用输出特征形状")
        
        self.init_modality_embedding()
        self.init_translator()

    def init_modality_embedding(self):
        modality_map = self.submodule_args.get('modality_map', None)
        if modality_map is None or len(modality_map) == 0:
            raise ValueError("submodule_args['modality_map'] 必须提供且非空，例如 ['m2','m3']")
        self.modality_map = modality_map
        self.modality2idx = {m: i for i, m in enumerate(modality_map)}
        
        M = len(self.modality_map)  # 模态数量
        D = self.embedding_dim     # embedding维度
        self.emb_table = nn.Parameter(torch.empty(M, D))
        nn.init.normal_(self.emb_table, mean=0.0, std=0.02)

    def init_translator(self):
        hidden_channel = self.submodule_args.get("dim", 64)
        self.channel_convert1 = nn.Conv2d(self.in_channels + self.embedding_dim, hidden_channel, kernel_size=1)
        self.layer_norm = nn.LayerNorm(hidden_channel)  # 添加LayerNorm以提高训练稳定性
        self.conv = ConvNeXt(self.submodule_args)
        self.channel_convert2 = nn.Conv2d(hidden_channel, self.out_channels, kernel_size=1)
        self.smoothing = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, ego_feature: torch.Tensor, mod_name: str):
        if mod_name not in self.modality2idx:
            raise ValueError(f"未知模态 {mod_name}，可选：{self.modality_map}")
        
        modality_embedding = self.emb_table[self.modality2idx[mod_name]]
        B, _, H, W = ego_feature.shape
        modality_embedding = modality_embedding.view(1, self.embedding_dim, 1, 1)
        modality_embedding = modality_embedding.expand(B, -1, H, W)
        
        concatenated_feature = torch.cat([ego_feature, modality_embedding], dim=1)
        
        x = self.channel_convert1(concatenated_feature)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        x = self.conv(x)
        
        x = self.channel_convert2(x)
        x = self.smoothing(x)
        
        return x


class TranslatorConvNextEmbMLPCat(nn.Module):
    def __init__(self, submodule_args, in_channels, out_channels, in_cav_lidar_range, out_cav_lidar_range, in_feature_shape, out_feature_shape):
        super(TranslatorConvNextEmbMLPCat, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.submodule_args = submodule_args
        self.embedding_dim = submodule_args.get("embedding_dim", 64)  # 模态embedding的维度
        self.hidden_mlp_dim = submodule_args.get("hidden_mlp_dim", 128)  # MLP隐藏层维度
        
        if in_feature_shape != out_feature_shape:
            print(f"警告: 输入特征形状 {in_feature_shape} 和输出特征形状 {out_feature_shape} 不匹配，将使用输出特征形状")
        
        self.init_modality_embedding()
        self.init_translator()

    def init_modality_embedding(self):
        modality_map = self.submodule_args.get('modality_map', None)
        if modality_map is None or len(modality_map) == 0:
            raise ValueError("submodule_args['modality_map'] 必须提供且非空，例如 ['m2','m3']")
        self.modality_map = modality_map
        self.modality2idx = {m: i for i, m in enumerate(modality_map)}
        
        M = len(self.modality_map)  # 模态数量
        D = self.embedding_dim     # embedding维度
        self.emb_table = nn.Parameter(torch.empty(M, D))
        nn.init.normal_(self.emb_table, mean=0.0, std=0.02)
        
        self.emb_mlp = nn.Sequential(
            nn.Linear(D, self.hidden_mlp_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_mlp_dim, D)
        )

    def init_translator(self):
        hidden_channel = self.submodule_args.get("dim", 64)
        self.channel_convert = nn.Conv2d(self.in_channels + self.embedding_dim, hidden_channel, kernel_size=1)
        self.layer_norm = nn.LayerNorm(hidden_channel)  # 添加LayerNorm以提高训练稳定性
        self.conv = ConvNeXt(self.submodule_args)
        
        # 先平滑
        self.smoothing = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, padding=1)
        
        # 然后用MLP做维度映射
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channel, self.out_channels, kernel_size=1)
        )

    def forward(self, ego_feature: torch.Tensor, mod_name: str):
        if mod_name not in self.modality2idx:
            raise ValueError(f"未知模态 {mod_name}，可选：{self.modality_map}")
        
        modality_embedding = self.emb_table[self.modality2idx[mod_name]]
        processed_embedding = self.emb_mlp(modality_embedding)
        B, _, H, W = ego_feature.shape
        processed_embedding = processed_embedding.view(1, self.embedding_dim, 1, 1)
        processed_embedding = processed_embedding.expand(B, -1, H, W)
        
        concatenated_feature = torch.cat([ego_feature, processed_embedding], dim=1)
        
        x = self.channel_convert(concatenated_feature)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        x = self.conv(x)
        
        x = self.smoothing(x)
        x = self.channel_mlp(x)
        
        return x


class GatedModulation(nn.Module):
    def __init__(self, num_channels, emb_dim):
        super().__init__()
        # 使用模态嵌入来生成每个通道的权重（门控信号）
        self.gate_net = nn.Sequential(
            nn.Linear(emb_dim, num_channels // 4), # 先降维
            nn.ReLU(),
            nn.Linear(num_channels // 4, num_channels), # 再升回通道数
            nn.Sigmoid() # 用Sigmoid将输出压缩到[0, 1]作为权重
        )

    def forward(self, x, emb):
        gate_weights = self.gate_net(emb) # [B, C]
        gate_weights = gate_weights.unsqueeze(-1).unsqueeze(-1).contiguous() # [B, C, 1, 1]
        return x * gate_weights # 对特征图的每个通道进行加权
    
class TranslatorConvNextEmbGate(nn.Module):
    def __init__(self, submodule_args, in_channels, out_channels, in_cav_lidar_range, out_cav_lidar_range, in_feature_shape, out_feature_shape):
        super(TranslatorConvNextEmbGate, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.submodule_args = submodule_args
        self.embedding_dim = submodule_args.get("embedding_dim", 64)  # 模态embedding的维度
        
        if in_feature_shape != out_feature_shape:
            print(f"警告: 输入特征形状 {in_feature_shape} 和输出特征形状 {out_feature_shape} 不匹配，将使用输出特征形状")
        
        self.init_modality_embedding()
        self.init_translator()

    def init_modality_embedding(self):
        modality_map = self.submodule_args.get('modality_map', None)
        if modality_map is None or len(modality_map) == 0:
            raise ValueError("submodule_args['modality_map'] 必须提供且非空，例如 ['m2','m3']")
        self.modality_map = modality_map
        self.modality2idx = {m: i for i, m in enumerate(modality_map)}
        
        M = len(self.modality_map)  # 模态数量
        D = self.embedding_dim     # embedding维度
        self.emb_table = nn.Parameter(torch.empty(M, D))
        nn.init.normal_(self.emb_table, mean=0.0, std=0.02)

    def init_translator(self):
        hidden_channel = self.submodule_args.get("dim", 64)
        self.channel_convert1 = nn.Conv2d(self.in_channels, hidden_channel, kernel_size=1)
        self.layer_norm = nn.LayerNorm(hidden_channel)  # 添加LayerNorm以提高训练稳定性
        self.conv = ConvNeXt(self.submodule_args)
        self.channel_convert2 = nn.Conv2d(hidden_channel, self.out_channels, kernel_size=1)
        self.smoothing = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

        self.gated_modulation = GatedModulation(hidden_channel, self.embedding_dim)

    def forward(self, ego_feature: torch.Tensor, mod_name: str):
        if mod_name not in self.modality2idx:
            raise ValueError(f"未知模态 {mod_name}，可选：{self.modality_map}")
        
        modality_embedding = self.emb_table[self.modality2idx[mod_name]]
        B, _, H, W = ego_feature.shape
        modality_embedding = modality_embedding.expand(B, -1)
        
        x = self.channel_convert1(ego_feature)
        x = self.gated_modulation(x, modality_embedding)
        
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        x = self.conv(x)
        
        x = self.channel_convert2(x)
        x = self.smoothing(x)
        
        return x


class ConvNeXtGateMod(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args['dim']
        kernel_size = args.get("kernel_size", 7)
        num_of_blocks = args['num_of_blocks']
        deform = args.get('deform', False)
        emb_dim = args.get('embedding_dim', 64)
        
        self.model_blocks = nn.ModuleList()
        self.gated_modulations = nn.ModuleList()
        
        for _ in range(num_of_blocks):
            self.gated_modulations.append(GatedModulation(dim, emb_dim))
            self.model_blocks.append(ConvNeXtBlock(dim, kernel_size=kernel_size, deform=deform))

    def forward(self, x, emb):
        for i, (gate_mod, block) in enumerate(zip(self.gated_modulations, self.model_blocks)):
            x = gate_mod(x, emb)
            x = block(x)
        return x

class TranslatorConvNextEmbGateMultilayer(nn.Module):
    def __init__(self, submodule_args, in_channels, out_channels, in_cav_lidar_range, 
                 out_cav_lidar_range, in_feature_shape, out_feature_shape):
        super(TranslatorConvNextEmbGateMultilayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.submodule_args = submodule_args
        self.embedding_dim = submodule_args.get("embedding_dim", 64)  # 模态embedding的维度
        
        if in_feature_shape != out_feature_shape:
            print(f"警告: 输入特征形状 {in_feature_shape} 和输出特征形状 {out_feature_shape} 不匹配，将使用输出特征形状")
        
        self.init_modality_embedding()
        self.init_translator()

    def init_modality_embedding(self):
        modality_map = self.submodule_args.get('modality_map', None)
        if modality_map is None or len(modality_map) == 0:
            raise ValueError("submodule_args['modality_map'] 必须提供且非空，例如 ['m2','m3']")
        self.modality_map = modality_map
        self.modality2idx = {m: i for i, m in enumerate(modality_map)}
        
        M = len(self.modality_map)  # 模态数量
        D = self.embedding_dim     # embedding维度
        self.emb_table = nn.Parameter(torch.empty(M, D))
        nn.init.normal_(self.emb_table, mean=0.0, std=0.02)

    def init_translator(self):
        hidden_channel = self.submodule_args.get("dim", 64)
        mlp_hidden_dim = self.submodule_args.get("mlp_hidden_dim", 64)
        self.channel_convert1 = nn.Conv2d(self.in_channels, hidden_channel, kernel_size=1)
        self.layer_norm = nn.LayerNorm(hidden_channel)
        
        convnext_args = self.submodule_args.copy()
        convnext_args['embedding_dim'] = self.embedding_dim
        self.conv = ConvNeXtGateMod(convnext_args)
        
        self.smoothing = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, padding=1)
        self.mlp = nn.Sequential(
                nn.Conv2d(hidden_channel, mlp_hidden_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mlp_hidden_dim, self.out_channels, kernel_size=1)
            )

    def forward(self, ego_feature: torch.Tensor, mod_name: str):
        if mod_name not in self.modality2idx:
            raise ValueError(f"未知模态 {mod_name}，可选：{self.modality_map}")
        
        modality_embedding = self.emb_table[self.modality2idx[mod_name]]
        B, _, H, W = ego_feature.shape
        modality_embedding = modality_embedding.expand(B, -1)
        
        x = self.channel_convert1(ego_feature)
        x = self.layer_norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        
        x = self.conv(x, modality_embedding)
        
        x = self.smoothing(x)
        x = self.mlp(x)
        
        return x
