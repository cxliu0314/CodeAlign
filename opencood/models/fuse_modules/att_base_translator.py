import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from typing import Optional, Dict


class CodebookCrossTranslator(nn.Module):
    def __init__(self, args: Dict):
        super().__init__()

        # 从args配置加载参数
        self.H, self.W = args['in_feature_shape']
        self.D = args['in_channels']
        self.Dc = args['code_dims']
        self.N = self.H * self.W
        self.h = args['num_heads']
        self.eps = 1e-6
        self.use_posenc = args.get('use_posenc', True)
        self.return_probs = args.get('return_probs', False)
        self.return_y_feat = args.get('return_y_feat', True)

        # 1. 输入归一化（启用权重和偏置）
        self.ln_in = nn.LayerNorm(self.D, elementwise_affine=True)

        # 2. 位置编码（可学习参数）
        if self.use_posenc:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.N, self.D))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
        else:
            self.pos_emb = None

        # 3. Transformer配置参数
        self.num_layers = args.get('num_layers', 2)
        self.dim_feedforward = args.get('dim_feedforward', self.D * 4)
        self.dropout = args.get('dropout', 0.1)

        # 4. 创建Transformer解码器层（支持交叉注意力）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D,
            nhead=self.h,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN结构更稳定
        )

        # 5. 堆叠多层Transformer
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )

        # 6. 创建Transformer解码器层（支持交叉注意力）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.D,
            nhead=self.h,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN结构更稳定
        )

        # 7. 堆叠多层Transformer解码器
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_layers
        )

        # 6. 最终输出投影（带偏置）
        self.final_proj = nn.Linear(self.D, self.D, bias=True)

    @staticmethod
    def build_2d_sincos_posenc(H: int, W: int, C: int) -> torch.Tensor:
        assert C % 2 == 0, "d_model 需为偶数以构建2D正弦PE"
        y, x = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                              torch.arange(W, dtype=torch.float32), indexing="ij")
        omega_y = torch.arange(C // 4, dtype=torch.float32) / (C // 4)
        omega_x = torch.arange(C // 4, dtype=torch.float32) / (C // 4)
        omega_y = 1.0 / (10000 ** omega_y)
        omega_x = 1.0 / (10000 ** omega_x)

        out_y = torch.einsum('hw,d->hwd', y, omega_y)
        out_x = torch.einsum('hw,d->hwd', x, omega_x)
        pe = torch.cat([torch.sin(out_y), torch.cos(out_y),
                        torch.sin(out_x), torch.cos(out_x)], dim=-1)
        pe = pe.view(H * W, C)
        return pe

    def _prep_inputs(self, feature: torch.Tensor) -> torch.Tensor:
        B, D, H, W = feature.shape
        assert H == self.H and W == self.W and D == self.D, \
            f"期望输入为 [B,{self.H},{self.W},{self.D}]，但得到 {feature.shape}"
        X = feature.view(B, self.N, D)
        X = self.ln_in(X)
        if self.use_posenc and self.pos_emb is not None:
            X = X + self.pos_emb
        return X

    def _prep_codebook(self, E_t: torch.Tensor, device) -> torch.Tensor:
        if E_t.dim() == 2:
            V, Dc = E_t.shape
            E_t = E_t.unsqueeze(0)
            E_t = E_t.to(device)
        elif E_t.dim() == 3:
            pass
        else:
            raise ValueError("E_t 形状必须是 [V, Dc] 或 [B, V, Dc]")
        return E_t

    def forward(self, feature: torch.Tensor, E_t: torch.Tensor, temperature: Optional[float] = None, return_dict: bool = False) -> Dict[str, torch.Tensor]:
        B, D, H, W = feature.shape
        device = feature.device

        # 1) 预处理 BEV token
        X = self._prep_inputs(feature)  # [B,N,D]

        # 2) 码本整理（维度已对齐，无需投影）
        E_t = self._prep_codebook(E_t, device)  # [B or 1, V, Dc]
        if E_t.size(0) == 1 and B > 1:
            E_t = E_t.expand(B, -1, -1).contiguous()

        # 3) 将码本作为memory传入Transformer解码器
        # 交叉注意力机制: X作为query, E_t作为key/value
        X = self.transformer_decoder(tgt=X, memory=E_t)

        # 4) 最终投影与残差连接
        X = X + self.final_proj(X)  # 增加残差连接

        # 5) 形状恢复
        y_feat = X.view(B, H, W, D).permute(0, 3, 1, 2)  # [B,D,H,W]

        # 6) 分类logits计算
        logits_map = torch.einsum('b d h w, b v d -> b v h w', y_feat, E_t.permute(0, 2, 1))
        if temperature is not None:
            logits_map = logits_map / max(temperature, self.eps)

        # 7) 软/硬输出
        indices = torch.argmax(logits_map, dim=1)
        probs = F.softmax(logits_map, dim=1) if self.return_probs else None

        if return_dict:
            out = {
                "logits": logits_map,
                "indices": indices,
            }
            if probs is not None:
                out["probs"] = probs
            if y_feat is not None:
                out["y_feat"] = y_feat
            return out
        else:
            return logits_map


if __name__ == '__main__':
    def load_config(config_path: str):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config


    # 示例配置：`config.yaml`
    config_yaml = """
    args:
        core_method: adapterconvnext
        args:
            in_channels: 64
            code_dims: 64
            in_feature_shape: [256, 256]
            out_feature_shape: [256, 256]
            num_heads: 4
            posenc_learnable: false
            use_posenc: true
            return_probs: true
            return_y_feat: true
    """
    with open("config.yaml", "w") as f:
        f.write(config_yaml)


    # 加载配置并实例化模型
    config = load_config("config.yaml")
    model_args = config['args']['args']
    model = CodebookCrossTranslator(model_args)

    # 测试前向
    F_in = torch.randn(2, 64, 256, 256)
    E_t = torch.randn(16, 64)
    output = model(F_in, E_t)
    print(output["logits"].shape)  # 应该输出 [2,16,256,256]
    print(output["indices"].shape)  # 应该输出 [2,256,256]