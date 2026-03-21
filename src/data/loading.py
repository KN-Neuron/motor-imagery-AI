"""
Data downloading and raw EDF loading.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import mne

# Subjects with known bad annotations (Shuqfa et al. 2024, Chowdhury et al. 2023)
BAD_SUBJECTS = {"038", "082", "089", "104"}


def download_dataset(
    dataset: str = "brianleung2020/eeg-motor-movementimagery-dataset",
    desired_runs: list[str] | None = None,
) -> dict[str, list[str]]:
    """
    Download dataset via kagglehub and return {subject_id: [file_paths]}.

    Parameters
    ----------
    dataset : str
        Kaggle dataset identifier.
    desired_runs : list of str
        Run codes to include, e.g. ["R04", "R08", "R12"].
    """
    import kagglehub

    if desired_runs is None:
        desired_runs = ["R04", "R08", "R12"]

    path = kagglehub.dataset_download(dataset)
    print(f"Dataset path: {path}")

    pat = re.compile(r"^S\d{3}.*\.edf$", re.IGNORECASE)
    subjects_data: dict[str, list[str]] = {}

    for dirname, _, filenames in os.walk(os.path.join(path, "files")):
        for filename in filenames:
            if pat.match(filename) and filename[4:-4] in desired_runs:
                subject = filename[1:4]
                if subject not in BAD_SUBJECTS:
                    subjects_data.setdefault(subject, []).append(
                        os.path.join(dirname, filename)
                    )

    print(f"Found {len(subjects_data)} subjects (excluded {len(BAD_SUBJECTS)} bad)")
    return subjects_data


def load_raw_subjects(
    subjects_data: dict[str, list[str]],
    sfreq: float = 160.0,
    n_subjects: int | None = None,
    cache_dir: str | None = None,
) -> dict[str, mne.io.Raw]:
    """
    Load raw EDF files per subject, concatenating multiple runs.

    Parameters
    ----------
    subjects_data : dict from download_dataset()
    sfreq : float
        Expected sampling frequency; files with different sfreq are skipped.
    n_subjects : int, optional
        Limit number of subjects (for debugging).
    cache_dir : str, optional
        Not used yet — placeholder for future caching.

    Returns
    -------
    dict of {subject_id: mne.io.Raw}
    """
    from tqdm import tqdm

    raw_data: dict[str, mne.io.Raw] = {}
    subjects = list(subjects_data.keys())
    if n_subjects is not None:
        subjects = subjects[:n_subjects]

    for subject in tqdm(subjects, desc="Loading subjects"):
        raws = []
        for f in subjects_data[subject]:
            raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
            if raw.info["sfreq"] == sfreq:
                raws.append(raw)

        if len(raws) == 0:
            print(f"⚠️ Subject {subject}: no valid files, skipping")
            continue
        elif len(raws) == 1:
            raw_data[subject] = raws[0]
        else:
            raw_data[subject] = mne.io.concatenate_raws(raws)

    print(f"Loaded {len(raw_data)} subjects")
    return raw_data