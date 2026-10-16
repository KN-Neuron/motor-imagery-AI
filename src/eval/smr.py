"""
SMR-predictor from resting state (RQ3), approximating Blankertz et al. 2010
(NeuroImage 51:1303, doi:10.1016/j.neuroimage.2010.03.022): Laplacian-filtered
C3/C4, PSD, 1/f-type noise floor fitted outside the peak range, predictor =
peak of PSD above that floor in the SMR band, averaged over C3 and C4.
This is an APPROXIMATION of the published procedure (exact fit ranges and
channel choice not reproduced; do weryfikacji before claiming replication).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

# PhysioNet channel names (MNE keeps the dots)
LAPLACE = {"C3": ("C3..", ["Fc3.", "Cp3.", "C1..", "C5.."]),
           "C4": ("C4..", ["Fc4.", "Cp4.", "C2..", "C6.."])}


def laplacian(data: np.ndarray, ch_names: list[str], centre: str, neighbours: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(ch_names)}
    return data[idx[centre]] - np.mean([data[idx[n]] for n in neighbours], axis=0)


def smr_peak(sig: np.ndarray, sfreq: float, band=(8.0, 15.0), fit_ranges=((3.0, 6.0), (20.0, 35.0))) -> float:
    f, p = welch(sig, fs=sfreq, nperseg=int(sfreq * 4))
    logp = np.log10(p + 1e-30)
    fit = np.zeros_like(f, bool)
    for lo, hi in fit_ranges:
        fit |= (f >= lo) & (f <= hi)
    coef = np.polyfit(np.log10(f[fit]), logp[fit], 1)  # power-law floor
    resid = logp - np.polyval(coef, np.log10(np.maximum(f, 1e-3)))
    m = (f >= band[0]) & (f <= band[1])
    return float(resid[m].max())


def smr_predictor(data: np.ndarray, ch_names: list[str], sfreq: float) -> float:
    return float(np.mean([smr_peak(laplacian(data, ch_names, c, n), sfreq) for c, n in LAPLACE.values()]))
