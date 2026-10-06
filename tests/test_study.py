import numpy as np
import pandas as pd
import pytest

from src.data.normalization import PreprocessingMismatch
from src.eval.degradation import (add_noise_and_drift, make_degradations, zero_channels, zero_pad_window)
from src.eval.montages import pick_channels, subset_indices
from src.eval.pipelines import make_registry
from src.eval.study import degradation_study, degradation_table, fewshot_study, trials_needed_curve

CH = ["Fc3.", "Fcz.", "C3..", "Cz..", "C4..", "Cp3.", "Pz..", "O1.."]


def _data(n_sub=10, n=20, t=160, seed=0):
    rng = np.random.RandomState(seed)
    X, y, s = [], [], []
    for i in range(n_sub):
        yi = np.tile([0, 1], n // 2)
        xi = (rng.randn(n, len(CH), t) * 1e-5).astype(np.float32)
        xi[yi == 1, 2] *= 2.5
        X.append(xi); y.append(yi); s.append(np.full(n, i))
    return np.concatenate(X), np.concatenate(y), np.concatenate(s)


def test_subset_never_zero_fills_and_name_mapping():
    X = np.random.randn(3, len(CH), 10)
    out = pick_channels(X, CH, ["C3", "CZ"])
    assert np.array_equal(out, X[:, [2, 3]])
    with pytest.raises(KeyError):
        subset_indices(CH, ["C3", "XX9"])


def test_degradation_functions():
    X = np.random.randn(4, len(CH), 100)
    z = zero_channels(X, CH, ["C3", "Cz"])
    assert (z[:, [0, 1, 4]] == 0).all() and np.array_equal(z[:, 2], X[:, 2])
    p = zero_pad_window(X, .5)
    assert (p[..., 50:] == 0).all() and np.array_equal(p[..., :50], X[..., :50])
    n = add_noise_and_drift(X, 160.0)
    assert n.shape == X.shape and not np.allclose(n, X)
    assert np.array_equal(n, add_noise_and_drift(X, 160.0))  # seeded


def test_degradation_study_runs_and_guard_flags_scale():
    X, y, s = _data()
    degs = make_degradations(CH, 160.0, {"sub4": ["C3", "Cz", "C4", "Pz"]})
    degs.pop("band_0.5-45Hz")
    reg = make_registry(epochs=2, device="cpu")
    df = degradation_study(X, y, s, CH, 160.0, reg["csp_lda"], degs,
                           train_norms=("none", "zscore_subject_channel"), n_outer=2)
    tab = degradation_table(df)
    assert set(tab.degradation) >= {"baseline", "scale_uV", "zeros_sub4", "subset_sub4_retrained"}
    base = tab[tab.degradation == "baseline"]
    assert np.allclose(base["drop"], 0)
    # same-subject coverage: every subject evaluated once per (norm, degradation)
    assert df.groupby(["train_norm", "degradation"]).subject.nunique().eq(10).all()
    # volts -> uV with train_norm none: guard must flag it
    g = tab[(tab.train_norm == "none") & (tab.degradation == "scale_uV")]
    assert g.guard_detects.iloc[0]


def test_fewshot_runs_all_methods():
    X, y, s = _data(n_sub=8, n=30)
    reg = make_registry(epochs=2, device="cpu")
    df = fewshot_study(X, y, s, reg["eegnet"], ks=(0, 3), train_norms=("euclidean_alignment",),
                       n_outer=2, n_draws=2)
    assert set(df.method) == {"no_adapt", "bn_adapt", "finetune_last"}
    assert set(df[df.k_per_class == 0].method) == {"no_adapt", "bn_adapt"}
    assert df.subject.nunique() == 8 and df.acc.between(0, 1).all()


def test_trials_needed_curve_monotone():
    rng = np.random.RandomState(0)
    rows = [(s, int(c)) for s in range(15) for c in rng.rand(200) < 0.75]
    cur = trials_needed_curve(pd.DataFrame(rows, columns=["subject", "correct"]), ns=(10, 45, 150))
    assert cur.frac_subjects_significant.is_monotonic_increasing
    assert cur.minutes.iloc[1] == pytest.approx(45 * 8 / 60)
