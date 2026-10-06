"""
Channel subsets and name mapping for simulating BrainAccess montages on PhysioNet.

ASSUMPTION (do potwierdzenia z dokumentacją producenta): the real BrainAccess
MIDI (16 ch) / MAXI (32 ch) electrode positions are NOT known from this repo.
The lists below are plausible 10-10 subsets that all exist in PhysioNet and are
used only to SIMULATE reduced montages. Replace with the real layout before
drawing any BrainAccess conclusion.
"""

from __future__ import annotations

import numpy as np

MIDI16 = ["F3", "Fz", "F4", "FC3", "FC4", "C3", "Cz", "C4",
          "CP3", "CP4", "P3", "Pz", "P4", "O1", "Oz", "O2"]
MAXI32 = MIDI16 + ["Fp1", "Fp2", "F7", "F8", "FC1", "FC2", "FC5", "FC6",
                   "C1", "C2", "CP1", "CP2", "CP5", "CP6", "T7", "T8"]
MONTAGES = {"midi16": MIDI16, "maxi32": MAXI32}


def norm_name(name: str) -> str:
    """PhysioNet 'Fc3.' / 'C3..' -> 'FC3'; BrainAccess-style 'FC3' unchanged."""
    return name.replace(".", "").upper()


def subset_indices(ch_names: list[str], wanted: list[str]) -> list[int]:
    """Indices of ``wanted`` in ``ch_names`` (name-normalized). Raises if one is missing:
    a common subset must be physically present, never zero-filled."""
    lookup = {norm_name(c): i for i, c in enumerate(ch_names)}
    missing = [w for w in wanted if norm_name(w) not in lookup]
    if missing:
        raise KeyError(f"channels not available: {missing}")
    return [lookup[norm_name(w)] for w in wanted]


def pick_channels(X: np.ndarray, ch_names: list[str], wanted: list[str]) -> np.ndarray:
    return X[:, subset_indices(ch_names, wanted)]
