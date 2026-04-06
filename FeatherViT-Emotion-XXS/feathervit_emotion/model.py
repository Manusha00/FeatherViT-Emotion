from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        use_act: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if use_act else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
    ) -> None:
        super().__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: List[nn.Module] = []
        if hidden_dim != in_channels:
            layers.append(ConvBNAct(in_channels, hidden_dim, kernel_size=1))

        layers.append(
            ConvBNAct(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                groups=hidden_dim,
            )
        )
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        y = self.block(x)
        if self.use_residual:
            return x + y
        return y


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        q = self.norm1(x)
        attn_out, _ = self.attn(q, q, q, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class FeatherGlobalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: int,
        patch_h: int = 2,
        patch_w: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w

        self.local_rep = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3, stride=1),
            ConvBNAct(in_channels, transformer_dim, kernel_size=1, stride=1),
        )
        self.transformers = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=transformer_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(n_transformer_blocks)
            ]
        )
        self.norm = nn.LayerNorm(transformer_dim)
        self.proj = ConvBNAct(transformer_dim, in_channels, kernel_size=1, stride=1)
        self.fuse = ConvBNAct(2 * in_channels, in_channels, kernel_size=3, stride=1)

    def _resize_input_if_needed(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        new_h = ((h + self.patch_h - 1) // self.patch_h) * self.patch_h
        new_w = ((w + self.patch_w - 1) // self.patch_w) * self.patch_w
        if new_h != h or new_w != w:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return x

    def _unfold(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        b, c, h, w = x.shape
        patch_area = self.patch_h * self.patch_w
        patches = F.unfold(
            x,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(b, c, patch_area, -1)
        return patches, (h, w)

    def _fold(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        b, c, patch_area, n = patches.shape
        patches = patches.reshape(b, c * patch_area, n)
        return F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._resize_input_if_needed(x)
        res = x

        fm = self.local_rep(x)
        patches, output_size = self._unfold(fm)
        b, c, p, n = patches.shape

        tokens = patches.permute(0, 2, 3, 1).reshape(b * p, n, c)
        tokens = self.transformers(tokens)
        tokens = self.norm(tokens)

        patches = tokens.reshape(b, p, n, c).permute(0, 3, 1, 2)
        fm = self._fold(patches, output_size=output_size)
        fm = self.proj(fm)
        return self.fuse(torch.cat([res, fm], dim=1))


@dataclass(frozen=True)
class FeatherViTEmotionXXSConfig:
    stem_out: int = 16
    layer1_out: int = 16
    layer2_out: int = 24
    layer3_out: int = 48
    layer4_out: int = 64
    layer5_out: int = 80
    layer3_transformer_dim: int = 64
    layer4_transformer_dim: int = 80
    layer5_transformer_dim: int = 96
    layer3_ffn_dim: int = 128
    layer4_ffn_dim: int = 160
    layer5_ffn_dim: int = 192
    expand_ratio: float = 2.0
    last_expansion_factor: int = 4


class FeatherViTEmotionXXS(nn.Module):
    """
    Lightweight FeatherViT-Emotion XXS backbone (~1.3M params target).
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.0) -> None:
        super().__init__()
        cfg = FeatherViTEmotionXXSConfig()
        self.cfg = cfg

        self.conv1 = ConvBNAct(3, cfg.stem_out, kernel_size=3, stride=2)

        self.layer1 = self._make_inverted_stage(
            in_channels=cfg.stem_out,
            out_channels=cfg.layer1_out,
            num_blocks=1,
            stride=1,
            expand_ratio=cfg.expand_ratio,
        )
        self.layer2 = self._make_inverted_stage(
            in_channels=cfg.layer1_out,
            out_channels=cfg.layer2_out,
            num_blocks=3,
            stride=2,
            expand_ratio=cfg.expand_ratio,
        )
        self.layer3 = self._make_feather_stage(
            in_channels=cfg.layer2_out,
            out_channels=cfg.layer3_out,
            transformer_dim=cfg.layer3_transformer_dim,
            ffn_dim=cfg.layer3_ffn_dim,
            n_transformer_blocks=2,
            stride=2,
            expand_ratio=cfg.expand_ratio,
        )
        self.layer4 = self._make_feather_stage(
            in_channels=cfg.layer3_out,
            out_channels=cfg.layer4_out,
            transformer_dim=cfg.layer4_transformer_dim,
            ffn_dim=cfg.layer4_ffn_dim,
            n_transformer_blocks=4,
            stride=2,
            expand_ratio=cfg.expand_ratio,
        )
        self.layer5 = self._make_feather_stage(
            in_channels=cfg.layer4_out,
            out_channels=cfg.layer5_out,
            transformer_dim=cfg.layer5_transformer_dim,
            ffn_dim=cfg.layer5_ffn_dim,
            n_transformer_blocks=3,
            stride=2,
            expand_ratio=cfg.expand_ratio,
        )

        exp_channels = min(cfg.last_expansion_factor * cfg.layer5_out, 960)
        self.expansion = ConvBNAct(cfg.layer5_out, exp_channels, kernel_size=1, stride=1)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(exp_channels, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_inverted_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        expand_ratio: float,
    ) -> nn.Sequential:
        blocks: List[nn.Module] = []
        for i in range(num_blocks):
            blk_stride = stride if i == 0 else 1
            blocks.append(
                InvertedResidual(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=blk_stride,
                    expand_ratio=expand_ratio,
                )
            )
            in_channels = out_channels
        return nn.Sequential(*blocks)

    def _make_feather_stage(
        self,
        in_channels: int,
        out_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: int,
        stride: int,
        expand_ratio: float,
    ) -> nn.Sequential:
        blocks: List[nn.Module] = []
        if stride == 2:
            blocks.append(
                InvertedResidual(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2,
                    expand_ratio=expand_ratio,
                )
            )
            in_channels = out_channels
        blocks.append(
            FeatherGlobalBlock(
                in_channels=in_channels,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                n_transformer_blocks=n_transformer_blocks,
                patch_h=2,
                patch_w=2,
                num_heads=4,
                dropout=0.0,
            )
        )
        return nn.Sequential(*blocks)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.expansion(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        return self.classifier(x)


def build_feathervit_emotion(num_classes: int, dropout: float = 0.0) -> FeatherViTEmotionXXS:
    return FeatherViTEmotionXXS(num_classes=num_classes, dropout=dropout)
