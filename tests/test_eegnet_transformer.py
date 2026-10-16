import pytest
import torch

from src.eval.pipelines import make_registry


@pytest.mark.parametrize("t", [561, 639, 641, 321])  # 639: conv padding 32 gives T+1 samples, (T+1)//8 != T//8
def test_forward_shape_for_any_length(t):
    from src.models.eegnet_transformer import SpatialEEGNetTransformer
    m = SpatialEEGNetTransformer(n_classes=2, channels=21, samples=t).eval()
    assert m(torch.randn(3, 21, t)).shape == (3, 2)        # (B, C, T) as fed by TorchPipeline
    assert m(torch.randn(3, 1, 21, t)).shape == (3, 2)     # (B, 1, C, T) as in the original


def test_registry_pipeline_trains():
    import numpy as np
    rng = np.random.RandomState(0)
    X, y = rng.randn(40, 4, 120).astype("float32"), np.tile([0, 1], 20)
    pipe = make_registry(epochs=1, device="cpu")["eegnet_transformer"](0)
    pipe.fit(X[:30], y[:30], X[30:], y[30:])
    assert pipe.predict_proba(X[30:]).shape == (10, 2)
