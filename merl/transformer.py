from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchvision import models


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class KhmerOCRTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        backbone_name: str = "resnet18",
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        resnet, feature_dim = self._load_backbone(backbone_name)
        self.backbone = nn.Sequential(*(list(resnet.children())[:-2]))
        self.conv_proj = nn.Conv2d(feature_dim, d_model, kernel_size=1)

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src_img: torch.Tensor, tgt_text_idx: torch.Tensor) -> torch.Tensor:
        features = self.backbone(src_img)
        features = self.conv_proj(features)

        src = features.flatten(2).permute(2, 0, 1)
        src = self.pos_encoder(src)

        tgt = self.embedding(tgt_text_idx).permute(1, 0, 2)
        tgt = self.pos_encoder(tgt)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
        tgt_padding_mask = tgt_text_idx == 0

        output = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        return self.fc_out(output.permute(1, 0, 2))

    def _load_backbone(self, backbone_name: str):
        name = backbone_name.lower()
        weight_map = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet34": models.ResNet34_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
        }
        if not hasattr(models, name):
            raise ValueError(f"Unsupported backbone '{backbone_name}'")

        weights = weight_map.get(name)
        builder = getattr(models, name)
        try:
            resnet = builder(weights=weights)
        except Exception:  # pragma: no cover - offline fallback
            resnet = builder(weights=None)
        return resnet, resnet.fc.in_features


__all__ = ["KhmerOCRTransformer", "PositionalEncoding"]
