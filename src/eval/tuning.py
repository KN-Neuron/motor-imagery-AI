"""
Hyperparameter selection inside N-LNSO (nested; Del Pup et al. 2025).

Per outer fold (train / val / test subjects from ``nlnso_splits``):
  - the TRAIN subjects are split into ``inner_folds`` groups; every candidate is fit on
    train minus one group (checkpoint on the outer VAL subjects) and scored on that group;
  - the candidate with the best mean inner score (mean per-subject accuracy) is refit on
    all TRAIN subjects, once per seed, checkpoint on VAL, and predicts TEST once.
Test subjects never enter selection. Scoring subjects never enter checkpointing.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.eval.nlnso import aggregate_trials, nlnso_splits


def sample_grid(grid: dict, n: int, seed: int = 0, default: dict | None = None) -> list[dict]:
    """Up to ``n`` distinct combinations of ``grid`` (the whole grid if it is not larger),
    drawn deterministically; ``default`` (if given) is always the first candidate."""
    keys = sorted(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]
    if default is not None:
        combos = [default] + [c for c in combos if c != default]
    if n >= len(combos):
        return combos
    head = combos[:1] if default is not None else []
    rest = combos[len(head):]
    idx = np.random.RandomState(seed).choice(len(rest), n - len(head), replace=False)
    return head + [rest[i] for i in sorted(idx)]


def _key(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def _subject_acc(y, pred, subjects) -> float:
    return float(pd.Series(y == pred).groupby(subjects).mean().mean())


def run_nlnso_tuned(
    X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
    make_from_params: Callable[[dict], Callable[[int], object]], candidates: list[dict], name: str,
    n_outer: int = 5, seeds=(0,), val_frac: float = 0.15, split_seed: int = 0, inner_folds: int = 3,
    cache: str | Path | None = None, log: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (per_subject_df, per_trial_df, selection_df). ``cache``: CSV of inner scores,
    appended after every inner fit so an interrupted run resumes where it stopped."""
    cache = Path(cache) if cache else None
    done = {}
    if cache is not None and cache.exists():
        for r in pd.read_csv(cache).itertuples():
            done[(int(r.fold), int(r.inner), r.params)] = float(r.score)
    keys = [_key(c) for c in candidates]
    total = n_outer * len(candidates) * inner_folds
    t0, n_new = time.time(), 0
    trial_rows, sel_rows = [], []
    for fold, tr, va, te in nlnso_splits(subjects, n_outer, val_frac, split_seed):
        m_va, m_te = np.isin(subjects, va), np.isin(subjects, te)
        groups = np.array_split(np.random.RandomState(split_seed + 2000 + fold).permutation(tr), inner_folds)
        for cand, key in zip(candidates, keys):
            scores = []
            for j, g in enumerate(groups):
                if (fold, j, key) not in done:
                    m_fit = np.isin(subjects, tr) & ~np.isin(subjects, g)
                    m_sc = np.isin(subjects, g)
                    pipe = make_from_params(cand)(seeds[0])
                    pipe.fit(X[m_fit], y[m_fit], X[m_va], y[m_va])
                    pred = pipe.predict_proba(X[m_sc]).argmax(1)
                    done[(fold, j, key)] = _subject_acc(y[m_sc], pred, subjects[m_sc])
                    if cache is not None:
                        pd.DataFrame([dict(fold=fold, inner=j, params=key, score=done[(fold, j, key)])]).to_csv(
                            cache, mode="a", header=not cache.exists(), index=False)
                    n_new += 1
                    if log:
                        el = time.time() - t0
                        left = total - len([k for k in done if k[2] in keys])
                        log(f"[{name}] fold {fold + 1}/{n_outer} inner {j + 1}/{inner_folds} {key} "
                            f"acc={done[(fold, j, key)]:.3f}, {el:.0f}s elapsed, ~{el / n_new * left / 3600:.1f} h left")
                scores.append(done[(fold, j, key)])
            sel_rows.append(dict(fold=fold, params=key, score=float(np.mean(scores)), selected=False))
        fold_sel = [r for r in sel_rows if r["fold"] == fold]
        best = max(range(len(fold_sel)), key=lambda i: (fold_sel[i]["score"], -i))  # ties -> earlier
        fold_sel[best]["selected"] = True
        if log:
            log(f"[{name}] fold {fold + 1}/{n_outer} selected {keys[best]} (inner acc {fold_sel[best]['score']:.3f})")
        m_tr = np.isin(subjects, tr)
        for seed in seeds:
            pipe = make_from_params(candidates[best])(seed)
            pipe.fit(X[m_tr], y[m_tr], X[m_va], y[m_va])
            proba = pipe.predict_proba(X[m_te])  # test used once, after selection
            for s, yt, p, pr in zip(subjects[m_te], y[m_te], proba.argmax(1), proba[:, -1]):
                trial_rows.append((name, int(s), fold, seed, int(yt), int(p), float(pr)))
    per, trials = aggregate_trials(trial_rows)
    return per, trials, pd.DataFrame(sel_rows)
