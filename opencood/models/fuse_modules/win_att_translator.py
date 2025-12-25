import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.fuse_modules.base_fuse_module import BaseFusionModule

class WinAttTranslator(BaseFusionModule):
    def __init__(self, in_channels=256, d_model=256, nhead=8, num_layers=3,
                 local_window_size=(16,16), global_window_size=(32,32), codebook_size=16,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.nhead = nhead
        
        # 输入映射与归一化
        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        
        # 窗口注意力层
        self.local_attn = WindowAttention(d_model, nhead, local_window_size, dropout)
        self.global_attn = WindowAttention(d_model, nhead, global_window_size, dropout)
        
        # 解码器（复用att_base_translator的交叉注意力结构）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_proj = nn.Linear(d_model, codebook_size)

    def forward(self, x, codebook):
        # x: (B, C, H, W) = (B, 256, 256, 256)
        # codebook: (16, d_model)
        B, C, H, W = x.shape
        
        # 输入处理
        x = self.input_proj(x)  # (B, d_model, 256, 256)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, d_model) = (B, 65536, 256)
        x = self.norm(x)
        
        # 位置编码（展平为序列后添加）
        pos_enc = self._generate_positional_encoding(H, W, x.device)
        x = x + pos_enc.unsqueeze(0)
        
        # 窗口注意力（局部+全局）
        x_local = self.local_attn(x, H, W)  # (B, 65536, d_model)
        x_global = self.global_attn(x, H, W)  # (B, 65536, d_model)
        x = x_local + x_global  # 特征融合
        
        # 与codebook交叉注意力
        codebook = codebook.unsqueeze(0).repeat(B, 1, 1)  # (B, 16, d_model)
        output = self.decoder(x, codebook)  # (B, 65536, d_model)
        
        # 输出处理
        logits = self.output_proj(output)  # (B, 65536, codebook_size)
        indices = logits.argmax(dim=-1)  # (B, 65536)
        
        return logits, indices

    def _generate_positional_encoding(self, H, W, device):
        # 生成2D相对位置编码
        position = torch.arange(H*W, device=device).reshape(H, W)
        i, j = position // W, position % W
        relative_i = i[:, None] - i[None, :]
        relative_j = j[:, None] - j[None, :]
        relative_pos = torch.stack([relative_i, relative_j], dim=-1)
        return relative_pos.flatten(0, 1).float()


class WindowAttention(nn.Module):
    def __init__(self, d_model, nhead, window_size, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.window_size = window_size
        self.dim_head = d_model // nhead
        
        # QKV投影
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        
        # 相对位置编码表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), nhead)
        )
        
        # 生成相对位置索引
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.relative_position_index = relative_coords.sum(-1)
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, H, W):
        # x: (B, H*W, d_model)
        B, L, C = x.shape
        window_size_h, window_size_w = self.window_size
        
        # 划分窗口
        x = x.view(B, H, W, C)
        x = x.unfold(1, window_size_h, window_size_h).unfold(2, window_size_w, window_size_w)
        x = x.reshape(B, -1, window_size_h*window_size_w, C)  # (B, num_windows, N, C)
        
        # QKV投影
        qkv = self.qkv_proj(x).reshape(B, -1, window_size_h*window_size_w, 3, self.nhead, self.dim_head)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)  # (3, B, num_windows, nhead, N, dim_head)
        q, k, v = qkv.unbind(0)
        
        # 计算注意力
        q = q * self.dim_head ** -0.5
        attn = (q @ k.transpose(-2, -1))  # (B, num_windows, nhead, N, N)
        
        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(window_size_h*window_size_w, window_size_h*window_size_w, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (nhead, N, N)
        attn = attn + relative_position_bias.unsqueeze(0).unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 输出
        x = (attn @ v).transpose(3, 4).reshape(B, -1, window_size_h*window_size_w, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # 窗口合并
        x = x.reshape(B, H//window_size_h, W//window_size_w, window_size_h, window_size_w, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        x = x.flatten(1, 2)  # (B, H*W, C)
        
        return x