"""
Subject-based data splitting and DataLoader creation.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset


def subject_split(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Split data by subject ID (no subject leaks between sets).

    Parameters
    ----------
    X, y, subjects : np.ndarray
        Full dataset arrays.
    train_ratio, val_ratio : float
        Proportions (test = 1 - train - val).
    seed : int
        Random seed for shuffling subjects.

    Returns
    -------
    dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test,
        train_mask, val_mask, test_mask,
        train_subjects, val_subjects, test_subjects
    """
    rng = np.random.RandomState(seed)
    unique_subjects = np.unique(subjects)
    rng.shuffle(unique_subjects)

    n = len(unique_subjects)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_subs = set(unique_subjects[:n_train])
    val_subs = set(unique_subjects[n_train : n_train + n_val])
    test_subs = set(unique_subjects[n_train + n_val :])

    train_mask = np.isin(subjects, list(train_subs))
    val_mask = np.isin(subjects, list(val_subs))
    test_mask = np.isin(subjects, list(test_subs))

    return {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "X_val": X[val_mask],
        "y_val": y[val_mask],
        "X_test": X[test_mask],
        "y_test": y[test_mask],
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "train_subjects": train_subs,
        "val_subjects": val_subs,
        "test_subjects": test_subs,
    }


def make_dataloaders(
    split_data: dict,
    batch_size: int = 64,
    num_workers: int = 0,
    augment_train: bool = False,
) -> dict:
    """
    Create PyTorch DataLoaders from a split dict.

    Returns
    -------
    dict with keys: train_loader, val_loader, test_loader
    """
    train_ds = EEGDataset(split_data["X_train"], split_data["y_train"], augment=augment_train)
    val_ds = EEGDataset(split_data["X_val"], split_data["y_val"])
    test_ds = EEGDataset(split_data["X_test"], split_data["y_test"])

    return {
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val_loader": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test_loader": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
