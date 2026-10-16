"""Training curves: every TorchPipeline fit records per-epoch losses; N-LNSO collects them."""
import numpy as np

from src.eval.nlnso import run_nlnso
from src.eval.pipelines import SklearnPipeline, make_registry


def _data():
    rng = np.random.RandomState(0)
    X = rng.randn(60, 4, 64).astype("float32")
    y = np.tile([0, 1], 30)
    X[y == 1, 0] *= 3
    return X, y, np.repeat(np.arange(6), 10)


def test_torch_pipeline_records_history_and_best_epoch():
    X, y, _ = _data()
    pipe = make_registry(epochs=3, device="cpu")["eegnet"](0)
    pipe.fit(X[:40], y[:40], X[40:], y[40:])
    h = pipe.history_
    assert list(h.epoch) == [0, 1, 2]
    assert {"train_loss", "train_acc", "val_loss", "val_acc", "lr"} <= set(h.columns)
    assert pipe.best_epoch_ == int(h.val_loss.idxmin())


def test_nlnso_collects_histories_with_fold_and_seed():
    X, y, s = _data()
    hist = []
    run_nlnso(X, y, s, make_registry(epochs=2, device="cpu")["eegnet"], "e", n_outer=3, seeds=(0, 1), history=hist)
    import pandas as pd
    h = pd.concat(hist)
    assert len(h) == 3 * 2 * 2 and set(h.fold) == {0, 1, 2} and set(h.seed) == {0, 1}
    assert (h.pipeline == "e").all() and "best_epoch" in h.columns


def test_sklearn_pipelines_have_no_history():
    X, y, s = _data()
    hist = []
    run_nlnso(X, y, s, lambda seed: SklearnPipeline("csp_lda", seed), "c", n_outer=3, history=hist)
    assert hist == []


def test_select_balanced_accuracy_records_it_and_picks_best_epoch():
    rng = np.random.RandomState(1)
    X = rng.randn(120, 4, 64).astype("float32")
    y = np.r_[np.zeros(20), np.ones(20), np.full(80, 2)].astype(int)   # imbalanced validation set
    X[y == 1, 0] *= 3
    from src.eval.pipelines import TorchPipeline
    from src.models.eegnet import EEGNet
    pipe = TorchPipeline(lambda c, k, t: EEGNet(chans=c, classes=k, time_points=t), epochs=4, device="cpu",
                         select="bal_acc")
    pipe.fit(X, y, X, y)
    h = pipe.history_
    assert "val_bal_acc" in h.columns
    order = h.sort_values(["val_bal_acc", "val_loss"], ascending=[False, True])
    assert pipe.best_epoch_ == int(order.epoch.iloc[0])


def test_balanced_accuracy_helper():
    from src.eval.pipelines import balanced_accuracy
    y = np.array([0, 0, 0, 0, 1])
    assert abs(balanced_accuracy(y, np.zeros(5, int)) - 0.5) < 1e-9      # always class 0: (1 + 0) / 2
    assert abs(balanced_accuracy(y, y) - 1.0) < 1e-9


def test_unknown_select_raises():
    import pytest
    from src.eval.pipelines import TorchPipeline
    with pytest.raises(ValueError):
        TorchPipeline(lambda c, k, t: None, select="bogus")
