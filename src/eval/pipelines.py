"""
Pipelines with a common interface: fit(X_tr, y_tr, X_val, y_val) / predict_proba(X).
Checkpoint selection uses ONLY (X_val, y_val), which are validation subjects
drawn from the training pool.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from src.models.eegnet import EEGNet


def _device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TorchPipeline:
    """Generic CNN trainer; best epoch picked on validation loss (acc as tiebreak)."""

    def __init__(self, build, epochs=50, lr=1e-3, batch_size=64, seed=0, weight_decay=0.0,
                 device=None, max_norm=False):
        self.build, self.epochs, self.lr, self.bs = build, epochs, lr, batch_size
        self.seed, self.wd, self.max_norm = seed, weight_decay, max_norm
        self.device = device or _device()
        self.model = None

    @staticmethod
    def _t(X):
        return torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32)

    def _eval(self, X, y):
        self.model.eval()
        with torch.no_grad():
            logits = torch.cat([self.model(xb.to(self.device)) for xb in self._t(X).split(256)])
        yt = torch.as_tensor(y).to(self.device)
        return nn.functional.cross_entropy(logits, yt).item(), (logits.argmax(1) == yt).float().mean().item()

    def fit(self, X_tr, y_tr, X_val, y_val):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        n_chans, n_times = X_tr.shape[1], X_tr.shape[2]
        n_classes = int(max(y_tr.max(), y_val.max())) + 1
        self.model = self.build(n_chans, n_classes, n_times).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        Xt, yt = self._t(X_tr), torch.as_tensor(y_tr, dtype=torch.long)
        g = torch.Generator().manual_seed(self.seed)
        best, best_state = (np.inf, 0.0), None
        for _ in range(self.epochs):
            self.model.train()
            for idx in torch.randperm(len(Xt), generator=g).split(self.bs):
                if len(idx) < 2:
                    continue
                loss = nn.functional.cross_entropy(self.model(Xt[idx].to(self.device)), yt[idx].to(self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                if hasattr(self.model, "apply_max_norm"):
                    self.model.apply_max_norm()
            sched.step()
            vl, va = self._eval(X_val, y_val)
            if (vl, -va) < (best[0], -best[1]):
                best, best_state = (vl, va), copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_state)
        self.val_loss_, self.val_acc_ = best
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            out = torch.cat([self.model(xb.to(self.device)) for xb in self._t(X).split(256)])
        return out.softmax(1).cpu().numpy()


def eegnet(max_norm: bool, **kw):
    def build(c, k, t):
        return EEGNet(chans=c, classes=k, time_points=t, use_max_norm=max_norm)
    return lambda seed: TorchPipeline(build, seed=seed, **kw)


def shallow(**kw):
    from braindecode.models import ShallowFBCSPNet
    return lambda seed: TorchPipeline(
        lambda c, k, t: ShallowFBCSPNet(n_chans=c, n_outputs=k, n_times=t, final_conv_length="auto"),
        seed=seed, **kw)


def deep(**kw):
    from braindecode.models import Deep4Net
    return lambda seed: TorchPipeline(
        lambda c, k, t: Deep4Net(n_chans=c, n_outputs=k, n_times=t, final_conv_length="auto"),
        seed=seed, **kw)


class SklearnPipeline:
    """CSP+LDA or tangent space + LR. Validation data is unused (no checkpoints)."""

    def __init__(self, kind: str, seed: int = 0):
        self.kind, self.seed = kind, seed

    def fit(self, X_tr, y_tr, X_val=None, y_val=None):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        if self.kind == "csp_lda":
            from mne.decoding import CSP
            self.clf = make_pipeline(CSP(n_components=6, reg="ledoit_wolf", log=True),
                                     LinearDiscriminantAnalysis())
        elif self.kind == "ts_lr":
            from pyriemann.estimation import Covariances
            from pyriemann.tangentspace import TangentSpace
            self.clf = make_pipeline(Covariances(estimator="oas"), TangentSpace(metric="riemann"),
                                     LogisticRegression(max_iter=2000, random_state=self.seed))
        else:
            raise ValueError(self.kind)
        self.clf.fit(X_tr.astype(np.float64), y_tr)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X.astype(np.float64))


def make_registry(epochs: int = 50, device=None) -> dict:
    kw = dict(epochs=epochs, device=device)
    return {
        "eegnet": eegnet(False, **kw),
        "eegnet_maxnorm": eegnet(True, **kw),
        "shallow": shallow(**kw),
        "deep": deep(**kw),
        "csp_lda": lambda seed: SklearnPipeline("csp_lda", seed),
        "ts_lr": lambda seed: SklearnPipeline("ts_lr", seed),
    }
