"""Build PhysioNet L/R epoch arrays from a local EDF directory (or kagglehub)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.data.loader import find_edf_files
from src.data.loading import download_dataset, load_raw_subjects
from src.data.normalization import PreprocMeta
from src.data.preprocessing import epoch_subjects
from src.data.subjects import filter_subjects

EVENT_ID = {"left_hand": 2, "right_hand": 3}


def load_raw(data_cfg: dict) -> dict:
    """Raw per subject after exclusion (preset + real sfreq check, all logged)."""
    runs = data_cfg.get("runs", ["R04", "R08", "R12"])
    files = (find_edf_files(data_cfg["data_dir"], runs) if data_cfg.get("data_dir")
             else download_dataset(desired_runs=runs))
    raw = load_raw_subjects(files, sfreq=data_cfg.get("sfreq", 160.0), cache_dir=data_cfg.get("cache_dir"))
    kept, _ = filter_subjects(raw, data_cfg.get("exclude", "koellod2023"), data_cfg.get("sfreq", 160.0))
    return kept


def build_epochs(raw: dict, band, tmin, tmax, normalization="none", channels=None,
                 cache_dir: str | None = None):
    """Returns X (n, C, T), y in {0,1}, subjects (int), ch_names, PreprocMeta."""
    key = hashlib.md5(json.dumps([sorted(raw), band, tmin, tmax, normalization, channels]).encode()).hexdigest()[:10]
    f = Path(cache_dir) / f"epochs_{key}.npz" if cache_dir else None
    first = next(iter(raw.values()))
    ch_names = [c for c in first.ch_names if channels is None or c in channels]
    if f is not None and f.exists():
        z = np.load(f)
        X, y, s = z["X"], z["y"], z["s"]
    else:
        X, y, s, _ = epoch_subjects(raw, EVENT_ID, channels=channels, low_freq=band[0], high_freq=band[1],
                                    tmin=tmin, tmax=tmax, normalization=normalization, label_offset=2)
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez(f, X=X, y=y, s=s)
    meta = PreprocMeta.from_training_data(
        X, bandpass=tuple(band), tmin=tmin, tmax=tmax, sfreq=float(first.info["sfreq"]),
        channels=ch_names, normalization=normalization)
    return X, y, s, ch_names, meta
