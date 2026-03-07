"""
PyTorch Dataset for EEG epochs.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """
    Wraps numpy arrays (n_samples, n_channels, n_timepoints) into a
    PyTorch Dataset with optional Gaussian noise + time-shift augmentation.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_channels, n_timepoints)
    y : np.ndarray, shape (n_samples,)
    augment : bool
        If True, add Gaussian noise (σ=0.1) and random time shift (±20 samples).
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
            noise = torch.randn_like(x) * 0.1
            x = x + noise
            shift = np.random.randint(-20, 21)
            x = torch.roll(x, shifts=shift, dims=-1)

        return x, y
