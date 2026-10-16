"""
EEG data augmentation utilities.

Three shape-preserving augmentations (ported from brain_board_augmentation.ipynb):
  - Gaussian noise scaled per channel std
  - Time warp via cubic spline (subtle stretch / compress)
  - Sliding window epoch extraction (offline, changes epoch length)

Public API
----------
add_gaussian_noise(X, noise_std_scale, rng)
time_warp_epochs(X, scale_range, rng)
sliding_window_augment(X, y, sfreq, window_sec, stride_sec)
build_augmented_dataset(X, y, sfreq, cfg)
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


# ─────────────────────────────────────────────────────────────────────────────
# Core transforms
# ─────────────────────────────────────────────────────────────────────────────

def add_gaussian_noise(
    X: np.ndarray,
    noise_std_scale: float = 0.06,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add Gaussian noise scaled to each channel's own std.

    Parameters
    ----------
    X : (n_epochs, n_channels, n_timepoints)
    noise_std_scale : float
        Noise magnitude relative to per-channel std (default 0.06).
    rng : numpy Generator, optional

    Returns
    -------
    np.ndarray, same shape and dtype as X.
    """
    rng = rng or np.random.default_rng()
    out = np.empty_like(X, dtype=np.float64)
    for i in range(X.shape[0]):
        x = X[i].astype(np.float64)
        ch_std = np.std(x, axis=1, keepdims=True)
        ch_std = np.where(ch_std < 1e-12, 1.0, ch_std)
        out[i] = x + noise_std_scale * ch_std * rng.standard_normal(x.shape)
    return out.astype(X.dtype, copy=False)


def time_warp_epochs(
    X: np.ndarray,
    scale_range: tuple[float, float] = (0.92, 1.08),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Lightly stretch or compress the time axis via cubic spline resampling.

    Each epoch gets an independent random scale factor drawn uniformly from
    *scale_range*.  The output is resampled back to the original length so
    the array shape is preserved.

    Parameters
    ----------
    X : (n_epochs, n_channels, n_timepoints)
    scale_range : (low, high)  e.g. (0.92, 1.08)
    rng : numpy Generator, optional

    Returns
    -------
    np.ndarray, same shape and dtype as X.
    """
    rng = rng or np.random.default_rng()
    n_ep, n_ch, n_t = X.shape
    t_orig = np.linspace(0.0, 1.0, n_t)
    out = np.empty_like(X, dtype=np.float64)

    for i in range(n_ep):
        scale = float(rng.uniform(*scale_range))
        n_new = max(2, int(round(n_t * scale)))
        t_warped = np.linspace(0.0, 1.0, n_new)
        t_out = np.linspace(0.0, 1.0, n_t)
        for c in range(n_ch):
            spline = CubicSpline(t_orig, X[i, c].astype(np.float64))
            warped_ch = spline(t_warped)
            out[i, c] = np.interp(t_out, t_warped, warped_ch)

    return out.astype(X.dtype, copy=False)


def sliding_window_augment(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    window_sec: float = 1.0,
    stride_sec: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract overlapping windows from each epoch (offline augmentation).

    This changes the time-point dimension from the original epoch length to
    *window_sec × sfreq* samples.  Use only on the training split; apply the
    same window length to validation / eval data by slicing the first window.

    Parameters
    ----------
    X : (n_epochs, n_channels, n_timepoints)
    y : (n_epochs,)
    sfreq : float
    window_sec, stride_sec : float

    Returns
    -------
    X_out : (n_windows, n_channels, window_samples)
    y_out : (n_windows,)
    """
    n_ep, n_ch, n_t = X.shape
    win = max(1, int(round(window_sec * sfreq)))
    step = max(1, int(round(stride_sec * sfreq)))
    if win > n_t:
        raise ValueError(
            f"sliding_window: window ({win} samples) > epoch length ({n_t} samples). "
            f"Reduce window_sec or increase tmax."
        )
    windows, labels = [], []
    for i in range(n_ep):
        for start in range(0, n_t - win + 1, step):
            windows.append(X[i, :, start : start + win])
            labels.append(y[i])
    return np.stack(windows, axis=0), np.array(labels, dtype=y.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# High-level builder
# ─────────────────────────────────────────────────────────────────────────────

def build_augmented_dataset(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    cfg: dict,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an augmented training dataset from *X* / *y* according to *cfg*.

    Expected *cfg* keys (all optional, shown with defaults):
    ::

        augmentation:
          enabled: false
          gaussian_noise: true
          n_noise_copies: 2
          noise_std_scale: 0.06
          time_warp: true
          time_warp_prob: 0.5
          time_warp_scale_range: [0.92, 1.08]
          sliding_window: false
          sliding_window_sec: 1.0
          sliding_stride_sec: 0.25

    The function always returns the original epochs plus any augmented copies
    concatenated.  Sliding windows (if enabled) are applied first, then noise
    and warp copies are added to the resulting windows.

    Returns
    -------
    X_aug : (n_augmented, n_channels, n_timepoints_or_window)
    y_aug : (n_augmented,)
    """
    aug = cfg.get("augmentation", {})
    if not aug.get("enabled", False):
        return X, y

    rng = np.random.default_rng(seed)

    # ── 1. Sliding window (changes time dimension) ──────────────────────────
    if aug.get("sliding_window", False):
        X, y = sliding_window_augment(
            X, y, sfreq,
            window_sec=aug.get("sliding_window_sec", 1.0),
            stride_sec=aug.get("sliding_stride_sec", 0.25),
        )
        print(
            f"  [aug] sliding window → {X.shape[0]} windows "
            f"({X.shape[2]} samples each)"
        )

    # ── 2. Gaussian noise copies ─────────────────────────────────────────────
    X_list = [X]
    y_list = [y]

    if aug.get("gaussian_noise", True):
        n_copies = int(aug.get("n_noise_copies", 2))
        scale = float(aug.get("noise_std_scale", 0.06))
        for _ in range(n_copies):
            X_list.append(add_gaussian_noise(X, noise_std_scale=scale, rng=rng))
            y_list.append(y.copy())
        print(f"  [aug] gaussian noise: {n_copies} copies (scale={scale})")

    # ── 3. Time warp copies ──────────────────────────────────────────────────
    if aug.get("time_warp", True):
        prob = float(aug.get("time_warp_prob", 0.5))
        scale_range = tuple(aug.get("time_warp_scale_range", [0.92, 1.08]))
        mask = rng.random(len(X)) < prob
        if mask.any():
            X_warped = X.copy()
            idx = np.where(mask)[0]
            X_warped[idx] = time_warp_epochs(X[idx], scale_range=scale_range, rng=rng)
            X_list.append(X_warped)
            y_list.append(y.copy())
            print(
                f"  [aug] time warp: {mask.sum()}/{len(X)} epochs warped "
                f"(prob={prob}, scale={scale_range})"
            )

    X_aug = np.concatenate(X_list, axis=0)
    y_aug = np.concatenate(y_list, axis=0)
    print(
        f"  [aug] dataset: {len(X)} → {len(X_aug)} samples "
        f"({len(X_aug)/len(X):.1f}x)"
    )
    return X_aug, y_aug
