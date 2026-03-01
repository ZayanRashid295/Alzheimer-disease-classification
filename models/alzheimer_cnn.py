"""
Custom CNN for Alzheimer MRI: 4 conv blocks, optional SE attention, GAP, FC head.
Input channels and num_classes are dynamic; He initialization supported.
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block: global pool -> FC -> ReLU -> FC -> sigmoid -> scale."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool. Padding to preserve spatial size before pool."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.pool(x)


class AlzheimerCNN(nn.Module):
    """
    Custom CNN: 4 conv blocks (32->64->128->256), optional SE on last block, GAP, FC 256->128->64->num_classes.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        conv_filters: List[int] = (32, 64, 128, 256),
        fc_sizes: List[int] = (256, 128, 64),
        dropout: tuple = (0.5, 0.3),
        use_se: bool = True,
        se_reduction: int = 16,
        he_init: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_filters = list(conv_filters)
        self.fc_sizes = list(fc_sizes)
        self.use_se = use_se
        self._build_conv_layers(se_reduction)
        self._build_fc_layers(dropout)
        if he_init:
            self._init_weights()

    def _build_conv_layers(self, se_reduction: int) -> None:
        layers = []
        in_ch = self.in_channels
        for i, out_ch in enumerate(self.conv_filters):
            layers.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.conv_blocks = nn.Sequential(*layers)
        if self.use_se:
            self.se = SqueezeExcitation(self.conv_filters[-1], reduction=se_reduction)
        else:
            self.se = None

    def _build_fc_layers(self, dropout: tuple) -> None:
        # After 4 blocks of 2x2 pool: 224 -> 112 -> 56 -> 28 -> 14
        self.gap = nn.AdaptiveAvgPool2d(1)
        in_features = self.conv_filters[-1]
        drop0, drop1 = dropout[0], dropout[1]
        fc_list = []
        for i, h in enumerate(self.fc_sizes):
            fc_list.append(nn.Linear(in_features, h))
            fc_list.append(nn.ReLU(inplace=True))
            fc_list.append(nn.Dropout(drop0 if i == 0 else drop1))
            in_features = h
        fc_list.append(nn.Linear(in_features, self.num_classes))
        self.fc = nn.Sequential(*fc_list)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        if self.se is not None:
            x = self.se(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def get_conv_output(self, x: torch.Tensor) -> torch.Tensor:
        """Return last conv feature map (for Grad-CAM)."""
        x = self.conv_blocks(x)
        if self.se is not None:
            x = self.se(x)
        return x
