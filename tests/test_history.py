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
