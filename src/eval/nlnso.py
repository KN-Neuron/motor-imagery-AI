"""
Nested Leave-N-Subjects-Out (N-LNSO), Del Pup et al. 2025 (arXiv:2505.13021).

Outer loop: disjoint groups of subjects, each subject is TEST exactly once.
Inner loop: validation subjects drawn from the remaining (training) subjects;
used only for checkpoint / hyperparameter selection. The test fold is touched
once per (pipeline, fold, seed), for the final prediction.
"""

from __future__ import annotations

from typing import Callable, Iterator

import time

import numpy as np
import pandas as pd


def nlnso_splits(
    subjects, n_outer: int = 5, val_frac: float = 0.15, seed: int = 0,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (fold, train_subjects, val_subjects, test_subjects), all disjoint."""
    uniq = np.array(sorted(set(subjects)))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(uniq)
    folds = np.array_split(perm, n_outer)
    for k, test in enumerate(folds):
        rest = np.array([s for s in perm if s not in set(test)])
        n_val = max(1, int(round(len(rest) * val_frac)))
        # inner split is deterministic per outer fold
        inner = np.random.RandomState(seed + 1000 + k).permutation(rest)
        yield k, np.sort(inner[n_val:]), np.sort(inner[:n_val]), np.sort(test)


def run_nlnso(
    X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
    make_pipeline: Callable[[int], object], name: str,
    n_outer: int = 5, seeds=(0,), val_frac: float = 0.15, split_seed: int = 0,
    balance_eval: bool = False, log: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ``make_pipeline(seed)`` returns an object with
    ``fit(X_tr, y_tr, X_val, y_val)`` and ``predict_proba(X) -> (n, n_classes)``.
    Returns (per_subject_df, per_trial_df). Per-subject accuracy is averaged
    over seeds; seeds are NOT independent samples.
    """
    trial_rows = []
    t0, n_runs = time.time(), 0
    for fold, tr, va, te in nlnso_splits(subjects, n_outer, val_frac, split_seed):
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
        m_tr, m_va, m_te = (np.isin(subjects, g) for g in (tr, va, te))
        for seed in seeds:
            pipe = make_pipeline(seed)
            pipe.fit(X[m_tr], y[m_tr], X[m_va], y[m_va])
            proba = pipe.predict_proba(X[m_te])  # test used once, after selection
            pred = proba.argmax(1)
            n_runs += 1
            if log:
                log(f"[{name}] fold {fold + 1}/{n_outer} seed {seed} done, {time.time() - t0:.0f}s elapsed")
            for s, yt, p, pr in zip(subjects[m_te], y[m_te], pred, proba[:, -1]):
                trial_rows.append((name, int(s), fold, seed, int(yt), int(p), float(pr)))
    trials = pd.DataFrame(trial_rows, columns=["pipeline", "subject", "fold", "seed", "y", "pred", "p_last"])
    trials["correct"] = (trials.y == trials.pred).astype(int)
    per = (trials.groupby(["pipeline", "subject", "seed"]).agg(
        n_trials=("correct", "size"), n_correct=("correct", "sum")).reset_index())
    per["acc"] = per.n_correct / per.n_trials
    per = per.groupby(["pipeline", "subject"]).agg(
        n_trials=("n_trials", "first"), n_correct=("n_correct", "mean"), acc=("acc", "mean")).reset_index()
    return per, trials
