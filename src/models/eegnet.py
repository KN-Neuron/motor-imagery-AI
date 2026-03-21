"""
EEGNet — compact CNN for EEG classification.

Reference: Lawhern et al., 2018, "EEGNet: A Compact Convolutional Neural
Network for EEG-based Brain-Computer Interfaces"

Adapted from: https://github.com/amrzhd/EEGNet/blob/main/EEGNet.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet architecture.

    Parameters
    ----------
    chans : int
        Number of EEG channels (e.g. 64 or 21).
    classes : int
        Number of output classes.
    time_points : int
        Number of time samples per epoch.
    temp_kernel : int
        Temporal convolution kernel size.
    f1 : int
        Number of temporal filters.
    f2 : int
        Number of pointwise filters (typically ``f1 * d``).
    d : int
        Depth multiplier for depthwise convolution.
    pk1, pk2 : int
        Pooling kernel sizes for blocks 2 and 3.
    dropout_rate : float
        Dropout probability.
    """

    def __init__(
        self,
        chans: int = 64,
        classes: int = 3,
        time_points: int = 641,
        temp_kernel: int = 80,
        f1: int = 8,
        f2: int = 16,
        d: int = 2,
        pk1: int = 4,
        pk2: int = 8,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        linear_size = (time_points // (pk1 * pk2)) * f2

        # Block 1: Temporal filtering
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, temp_kernel), padding="same", bias=False),
            nn.BatchNorm2d(f1),
        )

        # Block 2: Spatial filtering (depthwise)
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1)),
            nn.Dropout(dropout_rate),
        )

        # Block 3: Separable convolution
        self.block3 = nn.Sequential(
            nn.Conv2d(
                d * f1, d * f1, (1, 16), groups=d * f1, padding="same", bias=False
            ),
            nn.Conv2d(d * f1, f2, 1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2)),
            nn.Dropout(dropout_rate),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(linear_size, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, C, T) → (B, 1, C, T)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
