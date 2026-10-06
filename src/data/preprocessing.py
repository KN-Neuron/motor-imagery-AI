"""
EEG preprocessing — filtering, epoching, per-subject normalization.
"""

from __future__ import annotations

import numpy as np
import mne

from src.data.normalization import NORMALIZATIONS, exp_moving_standardize, normalize_subject

# Explicit annotation mapping — no fragile event_id='auto'
ANNOTATION_MAPPING = {"T0": 1, "T1": 2, "T2": 3}


def epoch_subjects(
    raw_data: dict[str, mne.io.Raw],
    event_id: dict[str, int],
    channels: list[str] | None = None,
    low_freq: float = 7.0,
    high_freq: float = 30.0,
    tmin: float = 0.0,
    tmax: float = 4.0,
    baseline: tuple | None = None,
    normalization: str = "none",
    balance: bool = False,
    label_offset: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Epoch all subjects with given preprocessing parameters.

    Returns UNBALANCED data by default. Balance per-split via subject_split().

    Parameters
    ----------
    raw_data : dict of {subject_id: mne.io.Raw}
    event_id : dict, e.g. {'left_hand': 2, 'right_hand': 3}
    channels : list or None (all 64 channels)
    normalization : one of NORMALIZATIONS. Always computed per subject, without
        labels (zscore_subject_channel, euclidean_alignment on epochs;
        exp_moving_standardization on the continuous filtered recording).
    balance : bool — if True, downsample globally (NOT recommended, use per-split)
    label_offset : int or None — subtracted from event codes to make 0-indexed.
        If None, auto-detected as min(event_id.values()).

    Returns
    -------
    X_all, y_all, subjects_all, skipped
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"Unknown normalization '{normalization}', choose from {NORMALIZATIONS}")
    if label_offset is None:
        label_offset = min(event_id.values())

    all_X, all_y, all_subjects = [], [], []
    skipped: list[str] = []

    for subject in raw_data:
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

            if normalization == "exp_moving_standardization":
                raw._data = exp_moving_standardize(raw.get_data()).astype(np.float64)

            events, _ = mne.events_from_annotations(
                raw, event_id=ANNOTATION_MAPPING, verbose=False
            )

            epochs = mne.Epochs(
                raw, events, event_id,
                tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True, verbose=False,
            )

            X = epochs.get_data().astype(np.float32)
            y = epochs.events[:, -1] - label_offset

            if normalization in ("zscore_subject_channel", "euclidean_alignment"):
                X = normalize_subject(X, normalization)

            all_X.append(X)
            all_y.append(y)
            all_subjects.append(np.full(len(y), int(subject)))

        except Exception as e:
            print(f"[epoch_subjects] skipped {subject}: {type(e).__name__}: {e}")
            skipped.append(subject)

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    subjects_all = np.concatenate(all_subjects, axis=0)

    if skipped:
        print(f"Skipped {len(skipped)} subjects: {skipped[:5]}...")

    if balance:
        rng = np.random.RandomState(seed)
        n_classes = len(np.unique(y_all))
        min_count = min(np.bincount(y_all))
        idx = []
        for cls in range(n_classes):
            cls_idx = np.where(y_all == cls)[0]
            chosen = rng.choice(cls_idx, size=min_count, replace=False)
            idx.append(chosen)
        idx = np.concatenate(idx)
        rng.shuffle(idx)
        X_all, y_all, subjects_all = X_all[idx], y_all[idx], subjects_all[idx]

    return X_all, y_all, subjects_all, skipped


def epoch_with_params(
    raw_data: dict[str, mne.io.Raw],
    low_freq: float,
    high_freq: float,
    tmin: float,
    tmax: float,
    baseline: tuple | None = None,
    channels: list[str] | None = None,
    task_mode: str = "binary",
    normalization: str = "none",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper for epoch_subjects with task mode selection.

    Used by preprocessing grid search and final retrain.
    """
    if task_mode == "binary":
        event_id = {"left_hand": 2, "right_hand": 3}
        label_offset = 2
    else:
        event_id = {"rest": 1, "left_hand": 2, "right_hand": 3}
        label_offset = 1

    X, y, subjects, _ = epoch_subjects(
        raw_data, event_id,
        channels=channels,
        low_freq=low_freq, high_freq=high_freq,
        tmin=tmin, tmax=tmax,
        baseline=baseline,
        normalization=normalization,
        balance=False,
        label_offset=label_offset,
        seed=seed,
    )
    return X, y, subjects