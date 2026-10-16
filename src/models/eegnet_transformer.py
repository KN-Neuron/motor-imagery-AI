"""
EEGNet front end (temporal conv + depthwise spatial conv) followed by a Transformer encoder
over time tokens. Architecture proposed by the project author (2026-10-08).

Changes against the original sketch (behaviour otherwise identical):
  - accepts (B, C, T) as well as (B, 1, C, T);
  - the token count is taken from a dummy forward pass: with kernel 64 and padding 32 the
    temporal conv returns T+1 samples, so ``samples // 8`` was wrong whenever (T+1) % 8 == 0
    (e.g. T=639) and the classifier crashed on a size mismatch.
"""

import torch
import torch.nn as nn


class SpatialEEGNetTransformer(nn.Module):
    def __init__(self, n_classes=4, channels=22, samples=1000, F1=8, D=2,
                 d_model=32, nhead=4, num_layers=2, dropout=0.5):
        super().__init__()
        F2 = F1 * D

        # Block 1: temporal (frequency) filtering
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2: depthwise spatial filtering
        self.depthwise = nn.Conv2d(F1, F2, kernel_size=(channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.act = nn.ELU()

        # shorter sequence before the Transformer (attention is O(T^2))
        self.pool = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop = nn.Dropout(dropout)

        # F2 -> d_model token projection
        self.proj = nn.Linear(F2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        with torch.no_grad():
            seq_len = self._tokens(torch.zeros(1, 1, channels, samples)).shape[1]
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * d_model, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def _tokens(self, x):
        x = self.bn1(self.conv1(x))
        x = self.drop(self.act(self.bn2(self.depthwise(x))))
        x = self.pool(x)                       # [B, F2, 1, T']
        return x.squeeze(2).permute(0, 2, 1)   # [B, T', F2]

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)                 # (B, C, T) -> (B, 1, C, T)
        x = self._tokens(x)
        x = self.proj(x) + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.classifier(x)
