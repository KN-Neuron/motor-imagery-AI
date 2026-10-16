import numpy as np

from src.eval.nlnso import nlnso_splits
from src.eval.tuning import run_nlnso_tuned, sample_grid


def test_sample_grid_deterministic_unique_default_first():
    grid = {"a": [1, 2, 3], "b": [10, 20], "c": ["x", "y"]}
    default = {"a": 2, "b": 10, "c": "y"}
    c1 = sample_grid(grid, 5, seed=0, default=default)
    assert c1 == sample_grid(grid, 5, seed=0, default=default)
    assert len(c1) == 5 and c1[0] == default
    assert len({tuple(sorted(c.items())) for c in c1}) == 5
    assert all(c["a"] in grid["a"] and c["b"] in grid["b"] for c in c1)
    assert len(sample_grid(grid, 100, seed=0)) == 12  # whole grid when n >= size


def _data(n_sub=15, n=10):
    """X[:, 0, 0] = subject id, X[:, 0, 1] = label, so a fake pipeline can see who it was given."""
    rng = np.random.RandomState(0)
    s = np.repeat(np.arange(n_sub), n)
    y = np.tile([0, 1], n_sub * n // 2)
    X = rng.randn(len(s), 2, 4)
    X[:, 0, 0], X[:, 0, 1] = s, y
    return X, y, s


class Fake:
    """``good`` reads the label from the data; otherwise predicts constant class 0."""
    log = []

    def __init__(self, good, seed):
        self.good = good

    def fit(self, X_tr, y_tr, X_val, y_val):
        Fake.log.append(("fit", set(X_tr[:, 0, 0].astype(int)), set(X_val[:, 0, 0].astype(int))))
        return self

    def predict_proba(self, X):
        Fake.log.append(("predict", set(X[:, 0, 0].astype(int))))
        p1 = X[:, 0, 1] if self.good else np.zeros(len(X))
        return np.stack([1 - p1, p1], 1)


def test_tuned_selects_on_training_subjects_only_and_picks_best(tmp_path):
    X, y, s = _data()
    Fake.log = []
    cands = [{"good": False}, {"good": True}]
    per, trials, sel = run_nlnso_tuned(
        X, y, s, lambda p: (lambda seed: Fake(p["good"], seed)), cands, "fake",
        n_outer=3, seeds=(0, 1), inner_folds=2, cache=tmp_path / "sel.csv")
    assert sorted(per.subject) == list(range(15)) and (per.acc == 1).all()
    assert list(sel.groupby("fold").apply(lambda d: d.loc[d.score.idxmax(), "params"])) == ['{"good": true}'] * 3
    # replay the log against the outer splits: test subjects reach predict only in the final fits
    folds = list(nlnso_splits(s, 3, 0.15, 0))
    i = 0
    for _, tr, va, te in folds:
        n_inner = len(cands) * 2
        for _ in range(n_inner):
            (_, f_tr, f_va), (_, f_pr) = Fake.log[i], Fake.log[i + 1]
            assert f_tr | f_pr <= set(tr) and not (f_tr & f_pr) and f_va == set(va)
            i += 2
        for _ in range(2):  # final fits, one per seed
            (_, f_tr, f_va), (_, f_pr) = Fake.log[i], Fake.log[i + 1]
            assert f_tr == set(tr) and f_va == set(va) and f_pr == set(te)
            i += 2
    assert i == len(Fake.log)


def test_tuned_resumes_inner_scores(tmp_path):
    X, y, s = _data()
    cands = [{"good": False}, {"good": True}]
    kw = dict(n_outer=3, seeds=(0,), inner_folds=2, cache=tmp_path / "sel.csv")
    make = lambda p: (lambda seed: Fake(p["good"], seed))  # noqa: E731
    run_nlnso_tuned(X, y, s, make, cands, "fake", **kw)
    Fake.log = []
    run_nlnso_tuned(X, y, s, make, cands, "fake", **kw)
    assert sum(1 for e in Fake.log if e[0] == "fit") == 3  # only the final fits, inner scores reused


def test_eegnet_factory_accepts_hyperparameters():
    from src.eval.pipelines import eegnet
    pipe = eegnet(False, epochs=1, device="cpu", f1=4, d=1, temp_kernel=32, dropout_rate=0.25, lr=5e-4)(0)
    assert pipe.lr == 5e-4
    m = pipe.build(21, 2, 561)
    assert m.block1[0].out_channels == 4 and m.block1[0].kernel_size == (1, 32)
