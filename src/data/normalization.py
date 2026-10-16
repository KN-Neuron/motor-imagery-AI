"""
Label-free, per-subject (or per-session) normalization and preprocessing
metadata with compatibility checks.

All normalizers use ONLY the data of one subject/session, never labels and
never other subjects, so no test information leaks into training statistics.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.signal import lfilter, welch

NORMALIZATIONS = (
    "none",
    "zscore_subject_channel",
    "exp_moving_standardization",
    "euclidean_alignment",
)


class PreprocessingMismatch(ValueError):
    """Input data does not match the preprocessing the model was trained with."""


# ── Normalizers ────────────────────────────────────────────────────

def zscore_subject_channel(X: np.ndarray) -> np.ndarray:
    """Per-channel z-score over all epochs and time points of ONE subject."""
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True)
    return ((X - mean) / np.where(std > 0, std, 1.0)).astype(np.float32)


def exp_moving_standardize(
    data: np.ndarray, factor_new: float = 0.001, init_block_size: int = 1000,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Electrode-wise exponential moving standardization, Schirrmeister et al. 2017
    (Hum Brain Mapp 38:5391, doi:10.1002/hbm.23730); factor_new=0.001 is their
    0.999 decay. Causal, so usable online. ``data`` is (channels, time) and
    should be CONTINUOUS (apply before epoching). The first ``init_block_size``
    samples are standardized with the statistics of that block.
    """
    data = np.asarray(data, dtype=np.float64)
    n_t = data.shape[-1]
    b, a = [factor_new], [1.0, -(1.0 - factor_new)]
    mean = lfilter(b, a, data, axis=-1, zi=None)
    demeaned = data - mean
    var = lfilter(b, a, demeaned ** 2, axis=-1)
    std = np.sqrt(var)
    out = demeaned / np.maximum(std, eps)
    k = min(init_block_size, n_t)
    blk = data[..., :k]
    out[..., :k] = (blk - blk.mean(-1, keepdims=True)) / np.maximum(
        blk.std(-1, keepdims=True), eps
    )
    return out.astype(np.float32)


def euclidean_alignment(X: np.ndarray, ref_inv_sqrt: np.ndarray | None = None):
    """
    Euclidean Alignment, He & Wu 2020 (IEEE TBME 67(2):399, doi:10.1109/TBME.2019.2913914).
    X (n, C, T) of one subject; returns R^{-1/2} X with R the mean covariance.
    If ``ref_inv_sqrt`` is given it is reused (e.g. from calibration trials).
    """
    if ref_inv_sqrt is None:
        ref_inv_sqrt = ea_reference(X)
    return np.einsum("ij,njt->nit", ref_inv_sqrt, X.astype(np.float64)).astype(np.float32)


def ea_reference(X: np.ndarray, reg: float = 1e-10) -> np.ndarray:
    X = X.astype(np.float64)
    cov = np.einsum("nct,ndt->cd", X, X) / (X.shape[0] * X.shape[2])
    cov += reg * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    return np.real(fractional_matrix_power(cov, -0.5))


def normalize_subject(X: np.ndarray, method: str, **kw) -> np.ndarray:
    """
    Normalize epochs (n, C, T) of one subject. For ``exp_moving_standardization``
    prefer applying it to the continuous recording (see preprocessing.py); on
    epochs it is applied per epoch with a 100-sample init block.
    """
    if method == "none":
        return X
    if method == "zscore_subject_channel":
        return zscore_subject_channel(X)
    if method == "exp_moving_standardization":
        return np.stack([
            exp_moving_standardize(e, init_block_size=kw.get("init_block_size", 100))
            for e in X
        ])
    if method == "euclidean_alignment":
        return euclidean_alignment(X, kw.get("ref_inv_sqrt"))
    raise ValueError(f"Unknown normalization '{method}', choose from {NORMALIZATIONS}")


# ── Preprocessing metadata + compatibility check ───────────────────

def _out_of_band_fraction(X: np.ndarray, sfreq: float, low: float, high: float) -> float:
    nper = min(X.shape[-1], 256)
    f, p = welch(X, fs=sfreq, nperseg=nper, axis=-1)
    p = p.mean(axis=tuple(range(p.ndim - 1)))
    keep = (f >= low * 0.5) & (f <= high * 1.25)
    return float(p[~keep].sum() / max(p.sum(), 1e-30))


@dataclass
class PreprocMeta:
    """Everything an inference/transfer step must reproduce. Saved in model meta JSON."""
    bandpass: tuple[float, float]
    tmin: float
    tmax: float
    sfreq: float
    channels: list[str]
    normalization: str
    n_times: int
    units: str = "V"
    # Reference statistics from the TRAINING data (after normalization):
    ref_channel_std: float | None = None
    ref_out_of_band: float | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(f"Unknown normalization '{self.normalization}'")
        self.bandpass = tuple(self.bandpass)

    @classmethod
    def from_training_data(cls, X: np.ndarray, **kw) -> "PreprocMeta":
        m = cls(n_times=X.shape[-1], **kw)
        m.ref_channel_std = float(np.median(X.std(axis=-1)))
        m.ref_out_of_band = _out_of_band_fraction(X, m.sfreq, *m.bandpass)
        return m

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bandpass"] = list(self.bandpass)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocMeta":
        return cls(**d)


def check_compatibility(
    meta: PreprocMeta, X: np.ndarray, channels: list[str], sfreq: float,
    scale_tol: float = 10.0, band_tol: float = 0.15, strict: bool = True,
) -> list[str]:
    """
    Compare incoming (already normalized, epoched) data with training metadata.
    Hard mismatches (channels, sfreq, window length, scale) raise
    PreprocessingMismatch when ``strict``; otherwise they are returned as
    warnings. Band mismatch is estimated spectrally and always warns.
    """
    problems: list[str] = []
    if list(channels) != list(meta.channels):
        problems.append(f"channels differ: got {list(channels)[:6]}..., expected {meta.channels[:6]}...")
    if float(sfreq) != float(meta.sfreq):
        problems.append(f"sfreq {sfreq} != {meta.sfreq}")
    if X.shape[-1] != meta.n_times:
        problems.append(f"window length {X.shape[-1]} != {meta.n_times} (no zero-padding allowed)")
    if X.shape[1] != len(meta.channels):
        problems.append(f"{X.shape[1]} channels != {len(meta.channels)}")
    if meta.ref_channel_std:
        ratio = float(np.median(X.std(axis=-1))) / meta.ref_channel_std
        if not (1 / scale_tol <= ratio <= scale_tol):
            problems.append(
                f"scale mismatch: median channel std is {ratio:.3g}x training "
                f"(units/normalization '{meta.normalization}')"
            )
    if meta.ref_out_of_band is not None and X.shape[-1] >= 32:
        oob = _out_of_band_fraction(X, sfreq, *meta.bandpass)
        if oob > meta.ref_out_of_band + band_tol:
            problems.append(
                f"band mismatch: out-of-band power {oob:.2f} vs training "
                f"{meta.ref_out_of_band:.2f} (expected band {meta.bandpass})"
            )
    if problems and strict:
        raise PreprocessingMismatch("; ".join(problems))
    for p in problems:
        warnings.warn(p)
    return problems
