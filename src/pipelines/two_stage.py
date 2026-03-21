"""
Two-stage pipeline: Mu-wave gating + binary L/R EEGNet.

Stage 1: Compute mu-band (8–13 Hz) power → gate rest vs active.
Stage 2: Classify active epochs as left or right with EEGNet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.signal import welch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset


def compute_mu_power(
    X: np.ndarray,
    sfreq: float = 160.0,
    mu_band: tuple[float, float] = (8, 13),
) -> np.ndarray:
    """
    Compute average mu-band power for each epoch.

    Parameters
    ----------
    X : (n_epochs, n_channels, n_timepoints)
    sfreq : float
    mu_band : (low, high)

    Returns
    -------
    mu_powers : (n_epochs,) array
    """
    mu_powers = []
    for i in range(X.shape[0]):
        epoch_power = []
        for ch in range(X.shape[1]):
            freqs, psd = welch(X[i, ch], fs=sfreq, nperseg=256)
            mu_mask = (freqs >= mu_band[0]) & (freqs <= mu_band[1])
            epoch_power.append(psd[mu_mask].mean())
        mu_powers.append(np.mean(epoch_power))
    return np.array(mu_powers)


def find_best_mu_threshold(
    mu_power: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float, pd.DataFrame]:
    """
    Find the mu-power threshold that best separates rest from active.

    Parameters
    ----------
    mu_power : (n_epochs,)
    y_true : (n_epochs,) — rest=0, left=1, right=2

    Returns
    -------
    best_thresh : float
    best_f1 : float
    results_df : DataFrame with columns [threshold, f1]
    """
    y_binary = (y_true > 0).astype(int)  # 0=rest, 1=active

    if thresholds is None:
        thresholds = np.linspace(
            np.percentile(mu_power, 10),
            np.percentile(mu_power, 90),
            100,
        )

    best_thresh, best_f1 = 0.0, 0.0
    results = []
    for t in thresholds:
        y_pred = (mu_power < t).astype(int)  # low mu → active
        f1 = f1_score(y_binary, y_pred, average="binary")
        results.append({"threshold": t, "f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1, pd.DataFrame(results)


def two_stage_predict(
    X: np.ndarray,
    mu_power: np.ndarray,
    mu_threshold: float,
    lr_model: torch.nn.Module,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Two-stage prediction:
      Stage 1 — mu-power gate: high mu → rest (0)
      Stage 2 — binary EEGNet: active epochs → left (1) or right (2)

    Returns
    -------
    predictions : (n_epochs,) with values {0, 1, 2}
    """
    predictions = np.zeros(len(X), dtype=int)

    active_mask = mu_power < mu_threshold
    predictions[~active_mask] = 0  # rest

    if active_mask.sum() > 0:
        X_active = X[active_mask]
        active_ds = DataLoader(
            EEGDataset(X_active, np.zeros(len(X_active))),
            batch_size=batch_size,
            shuffle=False,
        )

        lr_model.eval()
        active_preds = []
        with torch.inference_mode():
            for X_batch, _ in active_ds:
                logits = lr_model(X_batch.to(device))
                preds = logits.argmax(dim=1).cpu().numpy()
                active_preds.extend(preds)

        # Remap: 0→1 (left), 1→2 (right)
        predictions[active_mask] = np.array(active_preds) + 1

    return predictions
