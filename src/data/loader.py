"""
Data loading — download PhysioNet dataset and read raw EDF files.

Supports optional caching of raw MNE objects to avoid re-reading
hundreds of EDF files on every run.
"""

from __future__ import annotations

import os
import re
import pickle
from pathlib import Path
from typing import Any

import mne
import numpy as np
from tqdm import tqdm


def download_dataset(
    dataset_id: str = "brianleung2020/eeg-motor-movementimagery-dataset",
    desired_runs: list[str] | None = None,
) -> dict[str, list[str]]:
    """
    Download dataset via ``kagglehub`` and return a mapping of
    subject ID → list of EDF file paths for the desired runs.

    Parameters
    ----------
    dataset_id : str
        Kaggle dataset identifier.
    desired_runs : list of str, optional
        Run codes to keep (e.g. ``["R04", "R08", "R12"]``).

    Returns
    -------
    dict[str, list[str]]
        ``{ "001": ["/path/S001R04.edf", ...], ... }``
    """
    import kagglehub

    if desired_runs is None:
        desired_runs = ["R04", "R08", "R12"]

    path = kagglehub.dataset_download(dataset_id)
    print(f"Dataset path: {path}")

    pat = re.compile(r"^S\d{3}.*\.edf$", re.IGNORECASE)
    subjects_data: dict[str, list[str]] = {}

    for dirname, _, filenames in os.walk(os.path.join(path, "files")):
        for filename in filenames:
            if pat.match(filename) and filename[4:-4] in desired_runs:
                subject = filename[1:4]
                subjects_data.setdefault(subject, []).append(
                    os.path.join(dirname, filename)
                )

    print(f"Found {len(subjects_data)} subjects")
    return subjects_data


def load_raw_subjects(
    subjects_data: dict[str, list[str]],
    sfreq: float = 160.0,
    n_subjects: int | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, mne.io.Raw]:
    """
    Read raw EDF files per subject, concatenating multi-run files.

    Parameters
    ----------
    subjects_data : dict
        Mapping from ``download_dataset()``.
    sfreq : float
        Expected sampling frequency (files with different sfreq are skipped).
    n_subjects : int, optional
        Limit to first *n* subjects (for quick debugging).
    cache_dir : str or Path, optional
        If provided, pickled raw data is saved/loaded from this directory.

    Returns
    -------
    dict[str, mne.io.Raw]
        ``{ "001": <Raw object>, ... }``
    """
    if cache_dir is not None:
        cache_path = Path(cache_dir) / "raw_data.pkl"
        if cache_path.exists():
            print(f"Loading cached raw data from {cache_path}")
            with open(cache_path, "rb") as f:
                raw_data = pickle.load(f)
            if n_subjects:
                keys = sorted(raw_data.keys())[:n_subjects]
                raw_data = {k: raw_data[k] for k in keys}
            print(f"Loaded {len(raw_data)} subjects from cache")
            return raw_data

    subjects = sorted(subjects_data.keys())
    if n_subjects:
        subjects = subjects[:n_subjects]

    raw_data: dict[str, mne.io.Raw] = {}
    for subject in tqdm(subjects, desc="Loading subjects"):
        raws = []
        for f in subjects_data.get(subject, []):
            raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
            if raw.info["sfreq"] == sfreq:
                raws.append(raw)
        if len(raws) == 0:
            print(f"⚠️  Subject {subject}: no valid files, skipping")
            continue
        elif len(raws) == 1:
            raw_data[subject] = raws[0]
        else:
            raw_data[subject] = mne.io.concatenate_raws(raws)

    print(f"Loaded {len(raw_data)} subjects")

    if cache_dir is not None:
        cache_path = Path(cache_dir) / "raw_data.pkl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Caching raw data to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(raw_data, f)

    return raw_data
