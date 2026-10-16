"""
Replication of Bouchane, Guo, Yang 2025, "Hybrid CNN-GRU Models for Improved EEG Motor
Imagery Classification", Sensors 25(5):1399 (PMC11902626), under three split schemes.

Followed from the paper:
  - EEGMMIDB imagery runs; 5 classes: LF, RF (R04/R08/R12 T1/T2), both fists, both feet
    (R06/R10/R14 T1/T2), baseline B (T0, the majority class);
  - 4 s windows (640 samples at 160 Hz), 8-30 Hz 5th-order Butterworth, zero phase;
  - z-score within participant;
  - SMA E channel pairs, each pair a separate input pattern (640 x 2);
  - SMOTE (k=5) on the training part only, minority classes up to the majority;
  - CNN-GRU from Table 2; "accuracy" = (TP+TN)/N per class, averaged (one-vs-rest).
Not specified in the paper (our choices): the split for the main tables, which 7 subjects,
optimizer/lr/batch size, how B windows were taken; ICA is skipped (no details given).
"""

from __future__ import annotations

import re
from pathlib import Path

import mne
import numpy as np
import torch
import torch.nn as nn

RUNS = ["R04", "R06", "R08", "R10", "R12", "R14"]
LR_RUNS = {"R04", "R08", "R12"}
CLASSES = ["LF", "RF", "LRF", "BF", "B"]
PAIRS_E = [("Fc1.", "Fc2."), ("Fc3.", "Fc4."), ("C3..", "C4.."), ("C1..", "C2.."), ("Cp1.", "Cp2."), ("Cp3.", "Cp4.")]
PAPER_EXCLUDE = ["038", "088", "089", "092", "100", "104"]
WIN = 640


def _label(run: str, desc: str) -> int | None:
    if desc == "T0":
        return 4
    if desc in ("T1", "T2"):
        return (0 if run in LR_RUNS else 2) + (desc == "T2")
    return None


def load_subject(paths: list[str], band=(8.0, 30.0)):
    """-> X (n_windows, 12, 640) [pairs flattened in PAIRS_E order], y (n,), trial index (n,)."""
    chans = [c for p in PAIRS_E for c in p]
    Xs, ys = [], []
    for path in sorted(paths):
        run = re.search(r"(R\d{2})\.edf$", path, re.I).group(1).upper()
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        if raw.info["sfreq"] != 160.0:
            continue
        raw.pick(chans)
        raw.filter(*band, method="iir", iir_params=dict(order=5, ftype="butter"), phase="zero", verbose=False)
        data = raw.get_data()
        for on, desc in zip(raw.annotations.onset, raw.annotations.description):
            lab, i0 = _label(run, desc), int(round(on * 160.0))
            if lab is not None and i0 + WIN <= data.shape[1]:
                Xs.append(data[:, i0:i0 + WIN]); ys.append(lab)
    X = np.stack(Xs).astype(np.float32)
    X = (X - X.mean(axis=(0, 2), keepdims=True)) / X.std(axis=(0, 2), keepdims=True)  # within participant
    return X, np.array(ys), np.arange(len(ys))


def to_pairs(X, y, trial):
    """(n, 12, T) -> (n*6, 2, T): every channel pair of a window is a separate instance."""
    n, _, t = X.shape
    k = len(PAIRS_E)
    return (X.reshape(n, k, 2, t).reshape(n * k, 2, t), np.repeat(y, k), np.repeat(trial, k), np.tile(np.arange(k), n))


def task_mask(y, task: str):
    return y < 2 if task == "lr" else np.ones(len(y), bool)


def smote(X, y, k: int = 5, seed: int = 0):
    """SMOTE (Chawla et al. 2002) on flattened instances: every class up to the majority count."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.RandomState(seed)
    shape = X.shape[1:]
    F = X.reshape(len(X), -1)
    m = np.bincount(y).max()
    newX, newy = [], []
    for c in np.unique(y):
        Fc = F[y == c]
        need = m - len(Fc)
        if need <= 0:
            continue
        if len(Fc) == 1:
            newX.append(np.repeat(Fc, need, 0)); newy.append(np.full(need, c)); continue
        nn_idx = NearestNeighbors(n_neighbors=min(k, len(Fc) - 1) + 1).fit(Fc).kneighbors(Fc, return_distance=False)[:, 1:]
        i = rng.randint(len(Fc), size=need)
        j = nn_idx[i, rng.randint(nn_idx.shape[1], size=need)]
        lam = rng.rand(need, 1).astype(F.dtype)
        newX.append(Fc[i] + lam * (Fc[j] - Fc[i])); newy.append(np.full(need, c))
    if not newX:
        return X, y
    return (np.concatenate([X, np.concatenate(newX).reshape(-1, *shape)]), np.concatenate([y, np.concatenate(newy)]))


def splits(scheme: str, y, subj, trial, n_folds: int = 5, seed: int = 0, val_frac: float = 0.1):
    """Yield (train, val, test) index arrays.
    random_instance: instances shuffled (pairs of one window can sit in train and test);
    trial_grouped:   whole windows (all 6 pairs) in one fold; subjects shared;
    subject:         N-LNSO, subjects disjoint (src/eval/nlnso.py)."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    idx = np.arange(len(y))
    rng = np.random.RandomState(seed)
    if scheme == "subject":
        from src.eval.nlnso import nlnso_splits
        for _, tr, va, te in nlnso_splits(subj, n_folds, 0.15, seed):
            yield idx[np.isin(subj, tr)], idx[np.isin(subj, va)], idx[np.isin(subj, te)]
        return
    if scheme == "random_instance":
        for rest, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(idx, y):
            rest = rng.permutation(rest)
            n_va = max(1, int(round(len(rest) * val_frac)))
            yield np.sort(rest[n_va:]), np.sort(rest[:n_va]), te
        return
    if scheme == "trial_grouped":
        for rest, te in StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed).split(idx, y, trial):
            g = rng.permutation(np.unique(trial[rest]))
            va_g = g[:max(1, int(round(len(g) * val_frac)))]
            m = np.isin(trial[rest], va_g)
            yield rest[~m], rest[m], te
        return
    raise ValueError(scheme)


def ovr_accuracy(y, p, n_classes: int) -> float:
    """The paper's 'accuracy': (TP+TN)/N for each class, averaged over classes."""
    return float(np.mean([((y == c) == (p == c)).mean() for c in range(n_classes)]))


class CNNGRU(nn.Module):
    """Table 2 of the paper (Keras order: conv -> ReLU -> BN)."""

    def __init__(self, n_classes: int = 5, in_ch: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 32, 20, padding="same"), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, 20), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Dropout1d(0.5),
            nn.Conv1d(32, 32, 6), nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(32, 32, 6), nn.ReLU(),
            nn.Dropout1d(0.5),
        )
        self.gru = nn.GRU(32, 128, batch_first=True)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        h = self.features(x).transpose(1, 2)   # (B, T', 32)
        _, hn = self.gru(h)
        return self.fc(hn[-1])
