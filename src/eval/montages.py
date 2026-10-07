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

# EEGMMIDB channel labels as MNE reads them from the EDF headers (dots pad to 4 chars).
PHYSIONET64 = (
    "Fc5. Fc3. Fc1. Fcz. Fc2. Fc4. Fc6. C5.. C3.. C1.. Cz.. C2.. C4.. C6.. "
    "Cp5. Cp3. Cp1. Cpz. Cp2. Cp4. Cp6. Fp1. Fpz. Fp2. Af7. Af3. Afz. Af4. Af8. "
    "F7.. F5.. F3.. F1.. Fz.. F2.. F4.. F6.. F8.. Ft7. Ft8. T7.. T8.. T9.. T10. "
    "Tp7. Tp8. P7.. P5.. P3.. P1.. Pz.. P2.. P4.. P6.. P8.. Po7. Po3. Poz. Po4. Po8. "
    "O1.. Oz.. O2.. Iz.."
).split()


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
