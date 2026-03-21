"""
PyTorch Dataset for EEG epochs.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """
    EEG dataset wrapping numpy arrays as torch tensors.

    Parameters
    ----------
    X : np.ndarray, shape (n_epochs, n_channels, n_timepoints)
    y : np.ndarray, shape (n_epochs,)
    augment : bool
        If True, apply Gaussian noise + random time shift.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augment:
            x = x + torch.randn_like(x) * 0.1
            shift = torch.randint(-20, 21, (1,)).item()
            if shift > 0:
                x = torch.cat([torch.zeros_like(x[:, :shift]), x[:, :-shift]], dim=-1)
            elif shift < 0:
                x = torch.cat([x[:, -shift:], torch.zeros_like(x[:, :(-shift)])], dim=-1)

        return x, y