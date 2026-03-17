"""
Subject-based splitting and DataLoader creation.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset


def _balance_array(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, ...]:
    """Downsample to smallest class within a single split."""
    if len(X) == 0:
        return (X, y, subjects) if subjects is not None else (X, y)

    n_classes = len(np.unique(y))
    counts = np.bincount(y, minlength=n_classes)
    if 0 in counts:
        return (X, y, subjects) if subjects is not None else (X, y)

    rng = np.random.RandomState(seed)
    min_count = min(counts)
    idx = []
    for cls in range(n_classes):
        cls_idx = np.where(y == cls)[0]
        chosen = rng.choice(cls_idx, size=min_count, replace=False)
        idx.append(chosen)
    idx = np.concatenate(idx)
    rng.shuffle(idx)

    if subjects is not None:
        return X[idx], y[idx], subjects[idx]
    return X[idx], y[idx]


def subject_split(
    X_all: np.ndarray,
    y_all: np.ndarray,
    subjects_all: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    balance: bool = True,
) -> dict[str, np.ndarray]:
    """
    Split data by subject into train/val/test with per-split balancing.

    Parameters
    ----------
    train_ratio, val_ratio : float
        Test ratio is inferred as 1 - train - val.
    balance : bool
        If True, downsample each split independently to smallest class.

    Returns
    -------
    dict with keys: X_train, y_train, subjects_train, X_val, y_val, ...,
        X_test, y_test, subjects_test, train_subjects (set), val_subjects,
        test_subjects, train_mask, val_mask, test_mask.
    """
    rng = np.random.RandomState(seed)
    unique_subjects = np.unique(subjects_all)
    rng.shuffle(unique_subjects)

    n = len(unique_subjects)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_subjects = set(unique_subjects[:n_train])
    val_subjects = set(unique_subjects[n_train:n_train + n_val])
    test_subjects = set(unique_subjects[n_train + n_val:])

    split: dict[str, object] = {
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
    }

    for key, subj_set in [("train", train_subjects), ("val", val_subjects), ("test", test_subjects)]:
        mask = np.isin(subjects_all, list(subj_set))
        split[f"{key}_mask"] = mask

        X, y, s = X_all[mask], y_all[mask], subjects_all[mask]
        if balance and len(X) > 0:
            X, y, s = _balance_array(X, y, s, seed=seed)
        split[f"X_{key}"] = X
        split[f"y_{key}"] = y
        split[f"subjects_{key}"] = s

    return split


def make_dataloaders(
    split: dict[str, np.ndarray],
    batch_size: int = 64,
    augment_train: bool = False,
) -> dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders from a split dict.
    """
    train_ds = EEGDataset(split["X_train"], split["y_train"], augment=augment_train)
    val_ds = EEGDataset(split["X_val"], split["y_val"])
    test_ds = EEGDataset(split["X_test"], split["y_test"])

    return {
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val_loader": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test_loader": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }