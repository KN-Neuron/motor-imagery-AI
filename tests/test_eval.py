import numpy as np
import pytest

from src.eval.nlnso import nlnso_splits, run_nlnso
from src.eval.pipelines import make_registry
from src.eval.stats import (binomial_threshold, bootstrap_ci, holm, n_subjects_paired,
                            paired_wilcoxon, significant, summarize)


def test_nlnso_each_subject_tested_once_and_disjoint():
    subj = np.repeat(np.arange(23), 5)
    seen = []
    for _, tr, va, te in nlnso_splits(subj, n_outer=5, seed=3):
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
        assert len(va) >= 1
        seen += list(te)
    assert sorted(seen) == list(range(23))


def _synthetic(n_sub=12, n=24, c=6, t=160, seed=0):
    rng = np.random.RandomState(seed)
    X, y, s = [], [], []
    for i in range(n_sub):
        yi = np.tile([0, 1], n // 2)
        xi = rng.randn(n, c, t).astype(np.float32)
        xi[yi == 1, 0] *= 2.5  # class info in channel-0 variance
        X.append(xi); y.append(yi); s.append(np.full(n, i))
    return np.concatenate(X), np.concatenate(y), np.concatenate(s)


@pytest.mark.parametrize("name", ["csp_lda", "ts_lr", "eegnet", "eegnet_maxnorm"])
def test_pipelines_run_nlnso_and_learn(name):
    X, y, s = _synthetic()
    reg = make_registry(epochs=3, device="cpu")
    per, trials = run_nlnso(X, y, s, reg[name], name, n_outer=3)
    assert sorted(per.subject) == list(range(12))          # everyone tested exactly once
    assert (per.n_trials == 24).all()
    assert per.acc.mean() > 0.6 if name in ("csp_lda", "ts_lr") else per.acc.between(0, 1).all()


def test_test_set_never_seen_in_fit():
    X, y, s = _synthetic(n_sub=8)
    seen = {}

    class Spy:
        def fit(self, Xt, yt, Xv, yv):
            seen["fit"] = [np.asarray(a).copy() for a in (Xt, Xv)]; return self
        def predict_proba(self, X):
            fit = np.concatenate(seen["fit"])
            assert not any((fit == x).all() for x in X[:3]), "test epoch present in fit data"
            return np.full((len(X), 2), .5)

    run_nlnso(X, y, s, lambda seed: Spy(), "spy", n_outer=4)


def test_binomial_threshold_matches_combrisson():
    # Combrisson & Jerbi 2015: n=200 -> ~56% (p<0.05), ~58% (p<0.01), ~61% (p<0.001)
    assert abs(binomial_threshold(200, .05) - .56) < .01
    assert abs(binomial_threshold(200, .01) - .58) < .015
    assert abs(binomial_threshold(200, .001) - .61) < .015
    assert binomial_threshold(45, .05) > binomial_threshold(200, .05)  # threshold depends on real n
    assert significant([30, 22], [45, 45]).tolist() == [True, False]


def test_bootstrap_and_wilcoxon_and_holm():
    rng = np.random.RandomState(0)
    a = rng.uniform(.5, .9, 30)
    lo, hi = bootstrap_ci(a)
    assert lo < a.mean() < hi
    assert paired_wilcoxon(a + .05 + rng.randn(30) * .01, a)["p"] < 1e-3
    assert holm({"a": .01, "b": .02, "c": .5})["a"] == pytest.approx(.03)
    assert summarize(a, np.round(a * 45), np.full(30, 45))["n_subjects"] == 30
    assert 10 < n_subjects_paired(.1, .08) < 40


def test_within_subject_cv_detects_signal_and_chance():
    from src.eval.sanity import within_subject_cv
    X, y, s = _synthetic(n_sub=3, n=40)
    assert np.mean(list(within_subject_cv(X, y, s, "csp_lda").values())) > 0.8
    rng = np.random.RandomState(1)
    assert abs(np.mean(list(within_subject_cv(X, rng.permutation(y), s, "csp_lda").values())) - 0.5) < 0.2
