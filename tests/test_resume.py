import numpy as np
import pandas as pd

from src.eval.nlnso import run_nlnso
from src.eval.resume import cfg_hash, run_or_load


def test_run_or_load_skips_finished_and_separates_configs(tmp_path):
    calls = []

    def fn():
        calls.append(1)
        return pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [2]})

    h = cfg_hash({"x": 1})
    run_or_load(tmp_path, "eegnet|none", h, fn)
    per, tr = run_or_load(tmp_path, "eegnet|none", h, fn)
    assert len(calls) == 1 and per.a.iloc[0] == 1 and tr.b.iloc[0] == 2
    run_or_load(tmp_path, "eegnet|none", cfg_hash({"x": 2}), fn)  # changed config -> recompute
    assert len(calls) == 2


def test_nlnso_reports_progress():
    rng = np.random.RandomState(0)
    X, y = rng.randn(60, 4, 10), np.tile([0, 1], 30)
    s = np.repeat(np.arange(12), 5)

    class P:
        def fit(self, *a): pass
        def predict_proba(self, X): return np.full((len(X), 2), 0.5)

    msgs = []
    run_nlnso(X, y, s, lambda seed: P(), "p", n_outer=3, seeds=(0, 1), log=msgs.append)
    assert len(msgs) == 6 and "fold 1/3" in msgs[0] and "seed" in msgs[0]
