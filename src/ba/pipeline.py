"""
Single BrainAccess module: shared loader, preprocessing forced from model
metadata, common channel subset (never zeros), exact target window length
(never zero-padded), session/block splits, no epoch selection on test data.

TASK KIND: the current "hand clench" recordings are movement EXECUTION (ME),
not imagery (MI). ME results must never be reported as MI validation. A
separate MI block and an EMG/accelerometer control are reserved (see
``TASK_KIND`` and ``artifact_control``).
"""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import torch

from src.data.normalization import (PreprocMeta, PreprocessingMismatch, check_compatibility,
                                    exp_moving_standardize, normalize_subject)
from src.eval.montages import norm_name
from src.eval.pipelines import SklearnPipeline, TorchPipeline
from src.eval.study import _bn_adapt, _finetune_last
from src.models.eegnet import EEGNet

# trial_type -> (class index, task kind). ME = executed movement, MI = imagery.
TASK_KIND = {
    "LEFT_HAND_CLENCH": (0, "ME"), "RIGHT_HAND_CLENCH": (1, "ME"),
    "LEFT_HAND_IMAGERY": (0, "MI"), "RIGHT_HAND_IMAGERY": (1, "MI"),  # reserved for the future MI block
}


def artifact_control(raw: mne.io.BaseRaw, events: np.ndarray):
    """Reserved: EMG / accelerometer control for ME blocks (high-frequency power, ACC)."""
    raise NotImplementedError("EMG/accelerometer control is not implemented yet")


# ── loading ────────────────────────────────────────────────────────

def find_sessions(root: str | Path) -> list[tuple[Path, Path]]:
    """(edf, events.tsv) pairs: either <dir>/session.edf + events.tsv or BIDS *_eeg.edf + *_events.tsv."""
    root = Path(root)
    pairs = []
    for edf in sorted(root.rglob("*.edf")):
        ev = edf.with_name("events.tsv") if edf.name == "session.edf" else \
            edf.with_name(edf.name.replace("_eeg.edf", "_events.tsv"))
        if ev.exists():
            pairs.append((edf, ev))
    return pairs


def read_events(tsv: Path, task_kind: str = "ME") -> tuple[np.ndarray, np.ndarray]:
    """onset seconds and class labels, for rows of the requested task kind only."""
    import pandas as pd
    df = pd.read_csv(tsv, sep="\t")
    keep = df.trial_type.map(lambda t: TASK_KIND.get(t, (None, None))[1] == task_kind)
    df = df[keep]
    return df.onset.values.astype(float), df.trial_type.map(lambda t: TASK_KIND[t][0]).values.astype(int)


def load_session(edf: Path, tsv: Path, meta: PreprocMeta, task_kind: str = "ME"):
    """
    Epochs of ONE session, preprocessed exactly as ``meta`` demands:
    resample -> bandpass -> common channels (error if the model needs a channel
    that is missing) -> continuous EMS or per-session epoch normalization ->
    window of exactly meta.n_times samples. Returns X, y, trial_idx.
    """
    raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)
    lookup = {norm_name(c): c for c in raw.ch_names}
    missing = [c for c in meta.channels if norm_name(c) not in lookup]
    if missing:
        raise PreprocessingMismatch(f"session lacks model channels {missing}; use a model trained on a common subset")
    raw.pick([lookup[norm_name(c)] for c in meta.channels])
    raw.resample(meta.sfreq)
    raw.filter(*meta.bandpass, fir_design="firwin", verbose=False)
    data = raw.get_data()  # volts
    if meta.normalization == "exp_moving_standardization":
        data = exp_moving_standardize(data)
    onsets, y = read_events(tsv, task_kind)
    start = np.round((onsets + meta.tmin) * meta.sfreq).astype(int)
    keep = (start >= 0) & (start + meta.n_times <= data.shape[1])  # never zero-pad: drop incomplete trials
    if (~keep).any():
        print(f"[ba] {edf.name}: dropped {(~keep).sum()} trials shorter than the {meta.n_times}-sample window")
    start, y = start[keep], y[keep]
    X = np.stack([data[:, s:s + meta.n_times] for s in start]).astype(np.float32)
    if meta.normalization in ("zscore_subject_channel", "euclidean_alignment"):
        X = normalize_subject(X, meta.normalization)
    check_compatibility(meta, X, list(meta.channels), meta.sfreq)  # raises on scale/band/window mismatch
    return X, y, np.where(keep)[0]


def load_all(root: str | Path, meta: PreprocMeta, task_kind: str = "ME", block_size: int = 10):
    """Pool sessions; ``groups`` = session id * 1000 + trial block (contiguous trials). Normalization stays per session."""
    Xs, ys, gs = [], [], []
    for i, (edf, tsv) in enumerate(find_sessions(root)):
        X, y, idx = load_session(edf, tsv, meta, task_kind)
        Xs.append(X); ys.append(y); gs.append(i * 1000 + idx // block_size)
    if not Xs:
        raise FileNotFoundError(f"no sessions under {root}")
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


# ── splits (by session / block, never random epochs) ───────────────

def block_folds(groups: np.ndarray, n_folds: int = 5):
    """Contiguous, disjoint group folds in recording order. Yields (train_idx, val_idx, test_idx) with
    val = the block group just before test (adjacent blocks are NOT used for selection AND test)."""
    ug = np.unique(groups)
    n_folds = min(n_folds, len(ug))
    for te_groups in np.array_split(ug, n_folds):
        rest = np.array([g for g in ug if g not in set(te_groups)])
        if len(rest) < 2:
            continue
        val_g = rest[-max(1, len(rest) // 5):]
        tr_g = np.setdiff1d(rest, val_g)
        yield tuple(np.where(np.isin(groups, g))[0] for g in (tr_g, val_g, te_groups))


# ── model I/O with metadata ────────────────────────────────────────

def save_checkpoint(model, path: str | Path, meta: PreprocMeta, **extra):
    path = Path(path)
    torch.save(model.state_dict(), path)
    json.dump({"preproc": meta.to_dict(), "model": extra}, open(path.with_name(path.stem + "_meta.json"), "w"), indent=2)


def load_checkpoint(path: str | Path):
    """Returns (EEGNet, PreprocMeta). Legacy checkpoints without preprocessing metadata are refused."""
    path = Path(path)
    mp = path.with_name(path.stem + "_meta.json")
    if not mp.exists() or "preproc" not in json.load(open(mp)):
        raise PreprocessingMismatch(f"{path} has no preprocessing metadata; cannot enforce compatible preprocessing")
    d = json.load(open(mp))
    meta = PreprocMeta.from_dict(d["preproc"])
    m = d.get("model", {})
    net = EEGNet(chans=len(meta.channels), classes=m.get("classes", 2), time_points=meta.n_times,
                 f1=m.get("f1", 8), d=m.get("d", 2), f2=m.get("f1", 8) * m.get("d", 2))
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.eval(), meta


# ── methods, all evaluated on the SAME block folds ─────────────────

def evaluate_methods(X, y, groups, net: EEGNet, methods=("frozen", "bn_adapt", "finetune_last", "scratch", "csp_lda"),
                     n_folds: int = 5, seed: int = 0, device: str = "cpu"):
    """
    Returns {method: list of (n_test, n_correct) per fold}. Checkpoint/epoch choice only on the val
    blocks; test blocks used once for the final prediction. With little data this is a debugging tool, not evidence.
    """
    import copy
    res = {m: [] for m in methods}
    for tr, va, te in block_folds(groups, n_folds):
        for m in methods:
            if m == "frozen":
                mod = net
            elif m == "bn_adapt":
                mod = _bn_adapt(net, X[tr], device)
            elif m == "finetune_last":
                mod = _finetune_last(copy.deepcopy(net), X[tr], y[tr], device, seed)
            elif m == "scratch":
                p = TorchPipeline(lambda c, k, t: EEGNet(chans=c, classes=k, time_points=t), epochs=30, seed=seed, device=device)
                p.fit(X[tr], y[tr], X[va], y[va]); mod = p.model
            elif m == "csp_lda":
                p = SklearnPipeline("csp_lda", seed).fit(X[tr], y[tr])
                pred = p.predict_proba(X[te]).argmax(1)
                res[m].append((len(te), int((pred == y[te]).sum()))); continue
            else:
                raise ValueError(m)
            mod.eval()
            with torch.no_grad():
                pred = mod(torch.as_tensor(X[te]).to(device)).argmax(1).cpu().numpy()
            res[m].append((len(te), int((pred == y[te]).sum())))
    return res
