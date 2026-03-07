"""
Preprocessing — epoching, bandpass filtering, normalization, class balancing.

This module contains the core preprocessing logic that was previously
copy-pasted across 5+ notebook cells.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
from tqdm import tqdm


def _normalize_epochs(X: np.ndarray) -> np.ndarray:
    """Per-subject, per-channel z-score normalization (in-place)."""
    for ch in range(X.shape[1]):
        mean = X[:, ch, :].mean()
        std = X[:, ch, :].std()
        if std > 0:
            X[:, ch, :] = (X[:, ch, :] - mean) / std
    return X


def _balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    n_classes: int,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample to the smallest class."""
    if rng is None:
        rng = np.random.RandomState()

    min_count = min(np.bincount(y))
    balanced_idx = []
    for cls in range(n_classes):
        cls_idx = np.where(y == cls)[0]
        chosen = rng.choice(cls_idx, size=min_count, replace=False)
        balanced_idx.append(chosen)
    balanced_idx = np.concatenate(balanced_idx)
    rng.shuffle(balanced_idx)

    return X[balanced_idx], y[balanced_idx], subjects[balanced_idx]


def epoch_subjects(
    raw_data: dict[str, mne.io.Raw],
    event_id: dict[str, int],
    channels: list[str] | None = None,
    low_freq: float = 7.0,
    high_freq: float = 30.0,
    tmin: float = 0.0,
    tmax: float = 4.0,
    baseline: tuple | None = None,
    normalize: bool = True,
    balance: bool = True,
    label_offset: int = 1,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Epoch all subjects with given preprocessing parameters.

    Parameters
    ----------
    raw_data : dict[str, mne.io.Raw]
        Raw EEG data per subject.
    event_id : dict[str, int]
        MNE-style event mapping, e.g. ``{"left_hand": 2, "right_hand": 3}``.
    channels : list of str, optional
        Channel subset to pick. ``None`` → all channels.
    low_freq, high_freq : float
        Bandpass filter bounds.
    tmin, tmax : float
        Epoch time window.
    baseline : tuple or None
        Baseline correction parameter for ``mne.Epochs``.
    normalize : bool
        Apply per-subject channel-wise z-score.
    balance : bool
        Downsample to smallest class.
    label_offset : int
        Subtract this from MNE event codes to get 0-indexed labels.
        Default 1 for ternary (rest=1→0), use 2 for binary (left=2→0).
    seed : int
        Random seed for balancing.
    verbose : bool
        Show progress bar.

    Returns
    -------
    X_all : np.ndarray, shape (n_epochs, n_channels, n_timepoints)
    y_all : np.ndarray, shape (n_epochs,)
    subjects_all : np.ndarray, shape (n_epochs,)
    skipped : list of str
        Subject IDs that failed processing.
    """
    rng = np.random.RandomState(seed)
    n_classes = len(event_id)

    all_X, all_y, all_subjects = [], [], []
    skipped = []

    iterator = tqdm(raw_data, desc="Epoching") if verbose else raw_data
    for subject in iterator:
        try:
            raw = raw_data[subject].copy()
            if channels is not None:
                raw.pick(channels)
            raw.filter(
                low_freq, high_freq,
                fir_design="firwin",
                skip_by_annotation="edge",
                verbose=False,
            )

            events, _ = mne.events_from_annotations(
                raw, event_id="auto", verbose=False
            )

            epochs = mne.Epochs(
                raw, events, event_id,
                tmin=tmin, tmax=tmax,
                baseline=baseline,
                preload=True, verbose=False,
            )

            X = epochs.get_data().astype(np.float32)
            y = epochs.events[:, -1] - label_offset

            if normalize:
                X = _normalize_epochs(X)

            all_X.append(X)
            all_y.append(y)
            all_subjects.append(np.full(len(y), int(subject)))

        except Exception as e:
            skipped.append(subject)
            if verbose:
                print(f"⚠️  Subject {subject} failed: {e}")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    subjects_all = np.concatenate(all_subjects, axis=0)

    if balance:
        X_all, y_all, subjects_all = _balance_classes(
            X_all, y_all, subjects_all, n_classes, rng
        )

    if verbose:
        print(f"Total epochs: {X_all.shape[0]} | Shape: {X_all.shape}")
        print(f"Class distribution: {np.bincount(y_all)}")
        print(f"Subjects: {len(np.unique(subjects_all))}")
        if skipped:
            print(f"Skipped: {skipped}")

    return X_all, y_all, subjects_all, skipped


def epoch_with_params(
    raw_data: dict[str, mne.io.Raw],
    low_freq: float,
    high_freq: float,
    tmin: float,
    tmax: float,
    baseline: tuple | None,
    channels: list[str] | None = None,
    task_mode: str = "binary",
    seed: int = 42,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper for preprocessing grid search.

    Automatically selects event_id and label_offset based on *task_mode*.
    Returns (X, y, subjects) — no skipped list.
    """
    if task_mode == "binary":
        event_id = {"left_hand": 2, "right_hand": 3}
        label_offset = 2
    else:
        event_id = {"rest": 1, "left_hand": 2, "right_hand": 3}
        label_offset = 1

    # For baseline correction with tmin >= 0, shift tmin back
    actual_tmin = tmin
    if baseline == (None, 0) and tmin >= 0:
        actual_tmin = -0.5

    X, y, subjects, _ = epoch_subjects(
        raw_data,
        event_id=event_id,
        channels=channels,
        low_freq=low_freq,
        high_freq=high_freq,
        tmin=actual_tmin,
        tmax=tmax,
        baseline=baseline,
        label_offset=label_offset,
        seed=seed,
        verbose=verbose,
    )

    return X, y, subjects


def cache_epochs(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    path: str | Path,
) -> None:
    """Save processed epochs to compressed .npz for fast reload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y, subjects=subjects)
    print(f"Cached epochs to {path} ({X.shape[0]} epochs)")


def load_cached_epochs(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load cached epochs. Returns None if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return None
    data = np.load(path)
    print(f"Loaded cached epochs from {path}")
    return data["X"], data["y"], data["subjects"]
