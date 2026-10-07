"""Build PhysioNet L/R epoch arrays from a local EDF directory (or kagglehub)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.data.loader import find_edf_files
from src.data.loading import download_dataset, load_raw_subjects
from src.data.normalization import PreprocMeta, normalize_subject
from src.data.preprocessing import epoch_subjects
from src.data.subjects import filter_subjects

EVENT_ID = {"left_hand": 2, "right_hand": 3}


def load_raw(data_cfg: dict) -> dict:
    """
    Raw per subject after exclusion. Exclusion comes ONLY from ``data.exclude``
    (preset or id list) plus the real sfreq check; every dropped subject is logged
    with its reason (nothing is excluded silently by the downloader).
    """
    runs = data_cfg.get("runs", ["R04", "R08", "R12"])
    sfreq = data_cfg.get("sfreq", 160.0)
    files = (find_edf_files(data_cfg["data_dir"], runs) if data_cfg.get("data_dir")
             else download_dataset(desired_runs=runs, exclude=set()))
    raw = load_raw_subjects(files, sfreq=sfreq)
    for sid in sorted(set(files) - set(raw)):
        print(f"[subjects] dropped {sid}: no file with sfreq == {sfreq} (checked on the actual EDF header)")
    kept, _ = filter_subjects(raw, data_cfg.get("exclude", "koellod2023"), sfreq)
    return kept


def regress_out(X: np.ndarray, subjects: np.ndarray, ch_names: list[str], refs: list[str]):
    """EOG-style regression: per subject (label-free), remove from every non-reference
    channel its least-squares projection on the reference channels, then drop the
    references. Returns (X_clean, kept_channel_names)."""
    ri = [ch_names.index(r) for r in refs]
    ki = [i for i in range(len(ch_names)) if i not in ri]
    out = np.empty((len(X), len(ki), X.shape[2]), dtype=X.dtype)
    for sid in np.unique(subjects):
        m = subjects == sid
        R = X[m][:, ri].transpose(1, 0, 2).reshape(len(ri), -1)
        K = X[m][:, ki].transpose(1, 0, 2).reshape(len(ki), -1)
        R0 = R - R.mean(1, keepdims=True)
        K0 = K - K.mean(1, keepdims=True)
        B = np.linalg.lstsq(R0.T, K0.T, rcond=None)[0]  # (refs, kept)
        clean = K0 - B.T @ R0
        out[m] = clean.reshape(len(ki), m.sum(), -1).transpose(1, 0, 2)
    return out, [ch_names[i] for i in ki]


def build_epochs(raw: dict, band, tmin, tmax, normalization="none", channels=None,
                 cache_dir: str | None = None, eog_regress: list[str] | None = None):
    """Returns X (n, C, T), y in {0,1}, subjects (int), ch_names, PreprocMeta.
    ``eog_regress``: reference channels regressed out per subject and then dropped. The
    regression runs on unnormalized epochs and the normalization is applied afterwards
    (EA mixes channels, so regressing after it would no longer remove the references)."""
    final_norm = normalization
    if eog_regress:
        if normalization == "exp_moving_standardization":
            raise ValueError("eog_regress is not supported with exp_moving_standardization")
        normalization = "none"
    key = hashlib.md5(json.dumps([sorted(raw), band, tmin, tmax, normalization, channels]).encode()).hexdigest()[:10]
    f = Path(cache_dir) / f"epochs_{key}.npz" if cache_dir else None
    first = next(iter(raw.values()))
    ch_names = list(first.ch_names) if channels is None else list(channels)  # MNE pick() keeps the given order
    missing = [c for c in ch_names if c not in first.ch_names]
    if missing:
        raise KeyError(f"channels not in recording: {missing}")
    if f is not None and f.exists():
        z = np.load(f)
        X, y, s = z["X"], z["y"], z["s"]
    else:
        X, y, s, _ = epoch_subjects(raw, EVENT_ID, channels=channels, low_freq=band[0], high_freq=band[1],
                                    tmin=tmin, tmax=tmax, normalization=normalization, label_offset=2)
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez(f, X=X, y=y, s=s)
    if eog_regress:
        X, ch_names = regress_out(X, s, ch_names, eog_regress)
        if final_norm != "none":
            X = X.copy()
            for sid in np.unique(s):
                X[s == sid] = normalize_subject(X[s == sid], final_norm)
        normalization = final_norm
    meta = PreprocMeta.from_training_data(
        X, bandpass=tuple(band), tmin=tmin, tmax=tmax, sfreq=float(first.info["sfreq"]),
        channels=ch_names, normalization=normalization)
    return X, y, s, ch_names, meta
