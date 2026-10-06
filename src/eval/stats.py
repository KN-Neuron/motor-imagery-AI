"""
Subject-level statistics. The independent unit is the subject, never the epoch
or the seed. Average over seeds BEFORE calling these functions.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def binomial_threshold(n_trials: int, alpha: float = 0.05, chance: float = 0.5) -> float:
    """
    Smallest accuracy significantly above chance for ``n_trials`` (one-sided
    exact binomial, Combrisson & Jerbi 2015, J Neurosci Methods 250:126,
    doi:10.1016/j.jneumeth.2015.01.010). Returns np.inf if unreachable.
    """
    k = int(stats.binom.isf(alpha, n_trials, chance)) + 1  # P(X >= k) <= alpha
    return k / n_trials if k <= n_trials else float("inf")


def significant(n_correct, n_trials, alpha: float = 0.05, chance: float = 0.5) -> np.ndarray:
    """Per-subject exact binomial test P(X >= n_correct) < alpha, with the real n."""
    n_correct, n_trials = np.asarray(n_correct), np.asarray(n_trials)
    return stats.binom.sf(n_correct - 1, n_trials, chance) < alpha


def bootstrap_ci(values, stat=np.mean, n_boot: int = 10000, level: float = 0.95, seed: int = 0):
    """Percentile bootstrap CI resampling SUBJECTS with replacement."""
    v = np.asarray(values, float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(v), size=(n_boot, len(v)))
    boots = np.apply_along_axis(stat, 1, v[idx])
    a = (1 - level) / 2
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def summarize(acc, n_correct=None, n_trials=None, seed: int = 0) -> dict:
    """Mean/median with bootstrap CIs and fractions above chance-threshold and 70%."""
    acc = np.asarray(acc, float)
    out = {
        "n_subjects": len(acc),
        "mean": acc.mean(), "mean_ci": bootstrap_ci(acc, np.mean, seed=seed),
        "median": float(np.median(acc)), "median_ci": bootstrap_ci(acc, np.median, seed=seed),
        "q25": float(np.quantile(acc, .25)), "q75": float(np.quantile(acc, .75)),
        "frac_ge_70": float((acc >= 0.70).mean()),
    }
    if n_correct is not None:
        out["frac_significant"] = float(significant(n_correct, n_trials).mean())
    return out


def paired_wilcoxon(a, b) -> dict:
    """Paired Wilcoxon signed-rank on the same subjects (two-sided)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return {"mean_diff": 0.0, "p": 1.0, "diff_ci": (0.0, 0.0)}
    res = stats.wilcoxon(a, b)
    return {"mean_diff": float(d.mean()), "p": float(res.pvalue), "diff_ci": bootstrap_ci(d)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def n_subjects_paired(sd_diff: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Subjects needed for a paired t-test to detect mean difference ``delta``
    given SD of the per-subject differences (noncentral t, two-sided)."""
    for n in range(3, 2000):
        df = n - 1
        nc = delta / sd_diff * np.sqrt(n)
        tcrit = stats.t.isf(alpha / 2, df)
        pw = stats.nct.sf(tcrit, df, nc) + stats.nct.cdf(-tcrit, df, nc)
        if pw >= power:
            return n
    return -1
