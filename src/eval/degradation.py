"""
Pseudo-BrainAccess: controlled degradations of held-out PhysioNet epochs.

All functions take RAW (volts, un-normalized) epochs (n, C, T) and return
degraded epochs of the SAME shape; the model's normalization is applied
afterwards, per subject, exactly as it would be at inference.
"""

from __future__ import annotations

import numpy as np

from src.eval.montages import norm_name, subset_indices


def scale_to_microvolts(X, **_):
    return X * 1e6


def zero_channels(X, ch_names, keep, **_):
    """Channels outside ``keep`` become zeros (what the old BA scripts did)."""
    out = np.zeros_like(X)
    idx = subset_indices(ch_names, keep)
    out[:, idx] = X[:, idx]
    return out


def zero_pad_window(X, frac: float = 0.5, **_):
    """Keep the first ``frac`` of the window, zero-pad the rest."""
    out = np.zeros_like(X)
    k = int(X.shape[-1] * frac)
    out[..., :k] = X[..., :k]
    return out


def add_noise_and_drift(X, sfreq, snr_db: float = 0.0, drift_hz: float = 0.1,
                        drift_amp: float = 5.0, seed: int = 0, **_):
    """White noise at ``snr_db`` and a slow sinusoidal drift (amplitude in channel stds)."""
    rng = np.random.RandomState(seed)
    std = X.std(axis=-1, keepdims=True)
    noise = rng.randn(*X.shape) * std * 10 ** (-snr_db / 20)
    t = np.arange(X.shape[-1]) / sfreq
    phase = rng.uniform(0, 2 * np.pi, size=(X.shape[0], X.shape[1], 1))
    drift = drift_amp * std * np.sin(2 * np.pi * drift_hz * t + phase)
    return (X + noise + drift).astype(X.dtype)


def make_degradations(ch_names: list[str], sfreq: float, montages: dict[str, list[str]]) -> dict:
    """
    name -> dict(fn=callable(X)->X', kind=...). ``kind`` tells the study how to
    treat it: 'array' (apply fn), 'band' (use pre-filtered variant),
    'norm_mismatch' (test normalized differently), 'retrain_subset' (model
    retrained on the subset, evaluated on it).
    """
    deg = {
        "baseline": dict(kind="array", fn=lambda X: X),
        "scale_uV": dict(kind="array", fn=lambda X: scale_to_microvolts(X)),
        "norm_mismatch_zscore": dict(kind="norm_mismatch", fn=lambda X: X, test_norm="zscore_subject_channel"),
        "band_0.5-45Hz": dict(kind="band", fn=lambda X: X, band=(0.5, 45.0)),
        "zero_pad_50pct": dict(kind="array", fn=lambda X: zero_pad_window(X, 0.5)),
        "noise_drift": dict(kind="array", fn=lambda X: add_noise_and_drift(X, sfreq)),
    }
    for name, keep in montages.items():
        deg[f"subset_{name}_retrained"] = dict(kind="retrain_subset", fn=lambda X: X, keep=keep)
        deg[f"zeros_{name}"] = dict(kind="array", fn=lambda X, keep=keep: zero_channels(X, ch_names, keep))
    return deg
