"""
Filter Bank Common Spatial Patterns (FBCSP) — STUB.

This module is a placeholder for the FBCSP pipeline (section 20 of the
original notebook). The Butterworth bandpass filter is implemented;
the full FBCSP pipeline (filter bank → CSP per band → feature selection
→ classifier) is TODO.

Reference frequency bands (4 Hz bandwidth, 4–40 Hz):
  4–8, 8–12, 12–16, 16–20, 20–24, 24–28, 28–32, 32–36, 36–40 Hz
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter


# ── Default FBCSP filter bank ────────────────────────────────
FBCSP_BANDS = [
    (4, 8), (8, 12), (12, 16), (16, 20), (20, 24),
    (24, 28), (28, 32), (32, 36), (36, 40),
]


def butter_bandpass_filter(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 5,
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (filtering applied along last axis).
    lowcut, highcut : float
        Passband edges in Hz.
    fs : float
        Sampling frequency.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray — filtered signal.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, signal, axis=-1)


def apply_filter_bank(
    X: np.ndarray,
    fs: float = 160.0,
    bands: list[tuple[float, float]] | None = None,
    order: int = 5,
) -> list[np.ndarray]:
    """
    Apply a bank of bandpass filters to EEG epochs.

    Parameters
    ----------
    X : (n_epochs, n_channels, n_timepoints)
    fs : float
    bands : list of (low, high) tuples

    Returns
    -------
    list of np.ndarray — one filtered copy of X per band.
    """
    if bands is None:
        bands = FBCSP_BANDS

    return [butter_bandpass_filter(X, low, high, fs, order) for low, high in bands]


# TODO: Implement full FBCSP pipeline:
#   1. apply_filter_bank()
#   2. CSP per band → extract features
#   3. Mutual Information Based Best Individual Feature (MIBIF) selection
#   4. Classifier (SVM / LDA)
#
# See: Ang et al., 2012, "Filter Bank Common Spatial Pattern Algorithm
#      on BCI Competition IV Datasets 2a and 2b"
