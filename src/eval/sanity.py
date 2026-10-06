"""
Signal sanity checks. NOT results: they only tell whether epochs/labels carry
class information at all, to separate pipeline bugs from small-sample effects.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.eval.pipelines import SklearnPipeline


def within_subject_cv(X, y, subjects, kind="ts_lr", n_splits=5, seed=0) -> dict[int, float]:
    """
    Stratified k-fold INSIDE each subject. Valid only as a within-subject
    statement (never a cross-subject claim): the question here is whether the
    signal contains left/right information for the same person.
    """
    out = {}
    for s in np.unique(subjects):
        m = subjects == s
        Xs, ys = X[m], y[m]
        k = min(n_splits, np.bincount(ys).min())
        accs = []
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(Xs, ys):
            pred = SklearnPipeline(kind, seed).fit(Xs[tr], ys[tr]).predict_proba(Xs[te]).argmax(1)
            accs.append((pred == ys[te]).mean())
        out[int(s)] = float(np.mean(accs))
    return out
