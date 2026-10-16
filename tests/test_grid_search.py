"""Every grid function must evaluate EVERY combination (loop body must contain the CV, not only the print)."""
import numpy as np
import pandas as pd

import src.pipelines.grid_search as gs


def _data(n_subj=6, per=8, chans=4, T=256, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_subj * per, chans, T).astype("float32")
    y = np.tile([0, 1], n_subj * per // 2)
    s = np.repeat(np.arange(n_subj), per)
    return X, y, s


def test_eegnet_grid_evaluates_every_combination():
    X, y, s = _data()
    grid = {"lr": [1e-3, 5e-4], "dropout_rate": [0.25, 0.5], "f1": [4], "d": [2]}
    res = gs.run_eegnet_grid(X, y, s, param_grid=grid, chans=4, n_splits=2, epochs_train=1, device="cpu", verbose=False)
    assert len(res) == 4
    assert {(r["lr"], r["dropout_rate"]) for r in res} == {(1e-3, 0.25), (1e-3, 0.5), (5e-4, 0.25), (5e-4, 0.5)}


def test_shallow_grid_evaluates_every_combination():
    X, y, s = _data()
    grid = {"n_filters_time": [4], "n_filters_spat": [4], "filter_time_length": [25], "pool_time_length": [75],
            "pool_time_stride": [15], "drop_prob": [0.25, 0.5], "lr": [1e-3, 5e-4]}
    res = gs.run_shallow_grid(X, y, s, param_grid=grid, chans=4, n_splits=2, epochs_train=1, device="cpu", verbose=False)
    assert len(res) == 4
    assert {(r["drop_prob"], r["lr"]) for r in res} == {(0.25, 1e-3), (0.25, 5e-4), (0.5, 1e-3), (0.5, 5e-4)}


def test_eegnet_grid_checkpoint_resumes_without_repeating(tmp_path):
    X, y, s = _data()
    grid = {"lr": [1e-3, 5e-4], "dropout_rate": [0.5], "f1": [4], "d": [2]}
    ck = tmp_path / "ck.csv"
    kw = dict(param_grid=grid, chans=4, n_splits=2, epochs_train=1, device="cpu", verbose=False, checkpoint_path=ck)
    first = gs.run_eegnet_grid(X, y, s, **kw)
    again = gs.run_eegnet_grid(X, y, s, **kw)
    assert len(first) == 2 and len(again) == 2 and len(pd.read_csv(ck)) == 2


def test_preprocessing_grid_evaluates_every_combination(monkeypatch):
    seen = []

    def fake_epoch(raw, low, high, tmin, tmax, **kw):
        seen.append((low, high, tmin, tmax))
        return np.zeros((4, 2, 8)), np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])

    monkeypatch.setattr(gs, "epoch_with_params", fake_epoch)
    monkeypatch.setattr(gs, "cv_for_preprocessing", lambda *a, **k: (0.6, 0.01))
    grid = {"bandpass": [(4, 40), (8, 30)], "time_window": [(0.5, 2.5), (0.0, 4.0)], "baseline": [None]}
    df = gs.run_preprocessing_grid({}, grid, n_splits=2, epochs=1, verbose=False)
    assert len(df) == 4 and len(set(seen)) == 4


def test_joint_grid_processes_every_preprocessing_row(monkeypatch, tmp_path):
    calls = {"eegnet": 0, "shallow": 0, "csp": 0}

    monkeypatch.setattr(gs, "epoch_with_params", lambda *a, **k: (np.zeros((4, 2, 8)), np.array([0, 1, 0, 1]), np.arange(4)))

    def eegnet(*a, **k):
        calls["eegnet"] += 1
        return [dict(f1=8, d=2, dropout_rate=0.5, lr=1e-3, mean_acc=0.6, std_acc=0.01)]

    def shallow(*a, **k):
        calls["shallow"] += 1
        return [dict(n_filters_time=4, n_filters_spat=4, drop_prob=0.5, lr=1e-3, mean_acc=0.55, std_acc=0.01)]

    def csp(*a, **k):
        calls["csp"] += 1
        return [dict(model="LDA", best_cv_acc=0.58, best_params={"csp__n_components": 4})]

    monkeypatch.setattr(gs, "run_eegnet_grid", eegnet)
    monkeypatch.setattr(gs, "run_shallow_grid", shallow)
    monkeypatch.setattr("src.pipelines.csp_ml.run_csp_ml_grid", csp)
    top = pd.DataFrame([dict(low_freq=l, high_freq=h, tmin=0.5, tmax=2.5, baseline="no") for l, h in [(4, 40), (8, 30), (7, 30)]])
    ck = tmp_path / "joint.csv"
    df = gs.run_joint_grid({}, top, verbose=False, checkpoint_path=ck)
    assert calls == {"eegnet": 3, "shallow": 3, "csp": 3}
    assert len(df) == 9 and len(pd.read_csv(ck)) == 9


def test_joint_grid_checkpoint_saved_even_without_shallow(monkeypatch, tmp_path):
    monkeypatch.setattr(gs, "epoch_with_params", lambda *a, **k: (np.zeros((4, 2, 8)), np.array([0, 1, 0, 1]), np.arange(4)))
    monkeypatch.setattr(gs, "run_eegnet_grid", lambda *a, **k: [dict(f1=8, d=2, dropout_rate=0.5, lr=1e-3, mean_acc=0.6, std_acc=0.0)])
    top = pd.DataFrame([dict(low_freq=4, high_freq=40, tmin=0.5, tmax=2.5, baseline="no")])
    ck = tmp_path / "joint.csv"
    gs.run_joint_grid({}, top, include_csp_ml=False, include_shallow=False, verbose=False, checkpoint_path=ck)
    assert ck.exists() and len(pd.read_csv(ck)) == 1


def test_csp_ml_pipeline_uses_selected_hyperparameters():
    from src.pipelines.csp_ml import as_params, build_csp_ml_pipeline
    bp = {"classifier__solver": "lsqr", "classifier__shrinkage": "auto", "csp__n_components": 8}
    pipe = build_csp_ml_pipeline("LDA", "binary", bp)
    p = pipe.get_params()
    assert p["csp__n_components"] == 8 and p["classifier__shrinkage"] == "auto" and p["classifier__solver"] == "lsqr"
    # a checkpoint CSV stores the dict as text: it must round-trip, including tuples and None
    txt = str({"classifier__hidden_layer_sizes": (64, 32), "classifier__alpha": 0.001, "csp__n_components": 6,
               "classifier__shrinkage": None})
    assert as_params(txt)["classifier__hidden_layer_sizes"] == (64, 32) and as_params(bp) == bp


def test_joint_grid_keeps_csp_best_params(monkeypatch):
    monkeypatch.setattr(gs, "epoch_with_params", lambda *a, **k: (np.zeros((4, 2, 8)), np.array([0, 1, 0, 1]), np.arange(4)))
    monkeypatch.setattr("src.pipelines.csp_ml.run_csp_ml_grid",
                        lambda *a, **k: [dict(model="SVM", best_cv_acc=0.6, best_params={"classifier__C": 10.0, "csp__n_components": 8})])
    top = pd.DataFrame([dict(low_freq=4, high_freq=40, tmin=0.5, tmax=2.5, baseline="no")])
    df = gs.run_joint_grid({}, top, include_eegnet=False, include_shallow=False, verbose=False)
    assert df.iloc[0]["best_params"] == {"classifier__C": 10.0, "csp__n_components": 8}
