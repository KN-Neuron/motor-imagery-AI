"""
Simulation studies on PhysioNet (no BrainAccess data involved):
  1. degradation_study: pseudo-BrainAccess degradations vs accuracy drop,
  2. fewshot_study: calibration with k trials/class on a held-out subject,
  3. trials_needed_curve: how many trials (minutes) give a significant result.

Protocol everywhere: subject-wise N-LNSO folds, model selection on validation
subjects only, test subjects evaluated after training, normalization per
subject and label-free.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.normalization import PreprocMeta, check_compatibility, normalize_subject
from src.eval.montages import pick_channels
from src.eval.nlnso import nlnso_splits
from src.eval.stats import binomial_threshold


def norm_all(X, subjects, method):
    """Per-subject normalization of a pooled array."""
    out = np.empty_like(X, dtype=np.float32)
    for s in np.unique(subjects):
        m = subjects == s
        out[m] = normalize_subject(X[m], method)
    return out


def _per_subject(rows, **fixed):
    df = pd.DataFrame(rows, columns=["subject", "y", "pred"])
    g = df.assign(c=(df.y == df.pred).astype(int)).groupby("subject").c.agg(n_trials="size", n_correct="sum")
    g["acc"] = g.n_correct / g.n_trials
    g = g.reset_index()
    for k, v in fixed.items():
        g[k] = v
    return g


# ── 1. Degradation study ───────────────────────────────────────────

def degradation_study(
    X, y, subjects, ch_names, sfreq, make_pipeline, degradations,
    train_norms=("none", "zscore_subject_channel", "euclidean_alignment"),
    X_band_variants: dict | None = None, train_band=(7.0, 30.0), n_outer=5, seed=0,
) -> pd.DataFrame:
    """
    ``X`` raw volts (n, C, T). ``X_band_variants``: {band: array of the SAME
    epochs filtered with another band}. Returns per-subject rows per
    (train_norm, degradation) with ``detected`` = whether the compatibility
    guard flags the degraded input.
    """
    X_band_variants = X_band_variants or {}
    rows = []
    for fold, tr, va, te in nlnso_splits(subjects, n_outer, seed=seed):
        m_tr, m_va, m_te = (np.isin(subjects, g) for g in (tr, va, te))
        for tn in train_norms:
            Xn = norm_all(X, subjects, tn)
            pipe = make_pipeline(seed).fit(Xn[m_tr], y[m_tr], Xn[m_va], y[m_va])
            meta = PreprocMeta.from_training_data(
                Xn[m_tr], bandpass=train_band,
                tmin=0.0, tmax=X.shape[-1] / sfreq, sfreq=sfreq, channels=list(ch_names), normalization=tn)
            for name, d in degradations.items():
                if d["kind"] == "retrain_subset":
                    sub = pick_channels(X, ch_names, d["keep"])
                    Xs = norm_all(sub, subjects, tn)
                    p2 = make_pipeline(seed).fit(Xs[m_tr], y[m_tr], Xs[m_va], y[m_va])
                    pred = p2.predict_proba(Xs[m_te]).argmax(1)
                    detected = True  # channel set differs by construction (guard checks names)
                else:
                    if d["kind"] == "band":
                        src = X_band_variants.get(d["band"])
                        if src is None:
                            continue
                        Xd, norm = src[m_te], tn
                    elif d["kind"] == "norm_mismatch":
                        Xd, norm = X[m_te], d["test_norm"]
                    else:
                        Xd, norm = d["fn"](X[m_te]), tn
                    Xdn = norm_all(Xd, subjects[m_te], norm)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        detected = bool(check_compatibility(meta, Xdn, list(ch_names), sfreq, strict=False))
                    pred = pipe.predict_proba(Xdn).argmax(1)
                r = _per_subject(list(zip(subjects[m_te], y[m_te], pred)),
                                 train_norm=tn, degradation=name, fold=fold, detected=detected)
                rows.append(r)
    return pd.concat(rows, ignore_index=True)


def degradation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy drop vs baseline (paired by subject), mean and 95% bootstrap CI over subjects."""
    from src.eval.stats import bootstrap_ci
    out = []
    for (tn, dg), g in df.groupby(["train_norm", "degradation"]):
        base = df[(df.train_norm == tn) & (df.degradation == "baseline")].set_index("subject").acc
        a = g.set_index("subject").acc
        d = (base.loc[a.index] - a).values
        lo, hi = bootstrap_ci(d)
        out.append(dict(train_norm=tn, degradation=dg, acc=a.mean(), drop=d.mean(),
                        drop_ci_lo=lo, drop_ci_hi=hi, guard_detects=bool(g.detected.all())))
    return pd.DataFrame(out)


# ── 2. Few-shot calibration ────────────────────────────────────────

def _bn_adapt(model, Xc, device):
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.reset_running_stats()
            mod.momentum = None  # cumulative average over calibration data
    m.eval()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.train()
    with torch.no_grad():
        m(torch.as_tensor(Xc, dtype=torch.float32).to(device))
    m.eval()
    return m


def _finetune_last(model, Xc, yc, device, seed, epochs=30, lr=1e-3):
    """Fine-tune fc + block3 only; epoch chosen on an INTERNAL stratified split of the calibration set."""
    rng = np.random.RandomState(seed)
    tr_idx, va_idx = [], []
    for c in np.unique(yc):
        idx = rng.permutation(np.where(yc == c)[0])
        nv = max(1, int(round(len(idx) * 0.25)))
        va_idx += list(idx[:nv]); tr_idx += list(idx[nv:])
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad = False
    params = list(m.fc.parameters()) + list(m.block3.parameters())
    for p in params:
        p.requires_grad = True
    opt = torch.optim.Adam(params, lr=lr)
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt).to(device)
    Xt, yt, Xv, yv = T(Xc[tr_idx]), T(yc[tr_idx], torch.long), T(Xc[va_idx]), T(yc[va_idx], torch.long)
    best, state = np.inf, copy.deepcopy(m.state_dict())
    for _ in range(epochs):
        m.eval()  # BN/dropout frozen; only last layers' weights change
        loss = nn.functional.cross_entropy(m(Xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            vl = nn.functional.cross_entropy(m(Xv), yv).item()
        if vl < best:
            best, state = vl, copy.deepcopy(m.state_dict())
    m.load_state_dict(state)
    m.eval()
    return m


def fewshot_study(
    X, y, subjects, make_pipeline, ks=(0, 10, 20, 40), train_norms=("zscore_subject_channel", "euclidean_alignment"),
    n_outer=5, n_draws=5, seed=0,
) -> pd.DataFrame:
    """
    Per held-out subject, k trials/class of calibration (random, seeded, repeated
    ``n_draws`` times); evaluation on that subject's remaining trials. Methods:
    ``no_adapt`` (only the normalization re-estimated from calibration trials;
    for k=0 from the subject's unlabeled trials), ``bn_adapt`` (BN statistics,
    no labels), ``finetune_last``. ``make_pipeline`` must build an EEGNet TorchPipeline.
    """
    rows = []
    for fold, tr, va, te in nlnso_splits(subjects, n_outer, seed=seed):
        m_tr, m_va = np.isin(subjects, tr), np.isin(subjects, va)
        for tn in train_norms:
            Xn = norm_all(X, subjects, tn)
            pipe = make_pipeline(seed).fit(Xn[m_tr], y[m_tr], Xn[m_va], y[m_va])
            dev = pipe.device
            for s in te:
                ms = subjects == s
                Xs, ys = X[ms], y[ms]
                for k in ks:
                    for draw in range(n_draws if k > 0 else 1):
                        rng = np.random.RandomState(seed * 1000 + int(s) * 10 + draw)
                        cal = np.concatenate([rng.permutation(np.where(ys == c)[0])[:k] for c in np.unique(ys)]) if k else np.array([], int)
                        ev = np.setdiff1d(np.arange(len(ys)), cal)
                        ref = Xs[cal] if k else Xs  # label-free statistics source
                        if tn == "euclidean_alignment":
                            from src.data.normalization import ea_reference
                            R = ea_reference(ref)
                            Xa = normalize_subject(Xs, tn, ref_inv_sqrt=R)
                        else:
                            mean, std = ref.mean((0, 2), keepdims=True), ref.std((0, 2), keepdims=True)
                            Xa = ((Xs - mean) / np.where(std > 0, std, 1)).astype(np.float32)
                        models = {"no_adapt": pipe.model}
                        if k > 0:
                            models["bn_adapt"] = _bn_adapt(pipe.model, Xa[cal], dev)
                            models["finetune_last"] = _finetune_last(pipe.model, Xa[cal], ys[cal], dev, seed + draw)
                        else:
                            models["bn_adapt"] = _bn_adapt(pipe.model, Xa, dev)  # unlabeled stream
                        for mname, mod in models.items():
                            with torch.no_grad():
                                pred = mod(torch.as_tensor(Xa[ev], dtype=torch.float32).to(dev)).argmax(1).cpu().numpy()
                            rows.append((tn, mname, k, int(s), draw, float((pred == ys[ev]).mean()), len(ev)))
    df = pd.DataFrame(rows, columns=["train_norm", "method", "k_per_class", "subject", "draw", "acc", "n_eval"])
    return (df.groupby(["train_norm", "method", "k_per_class", "subject"])
              .agg(acc=("acc", "mean"), n_eval=("n_eval", "first")).reset_index())


# ── 3. Trials / minutes needed ─────────────────────────────────────

def trials_needed_curve(trials: pd.DataFrame, ns=(10, 20, 30, 45, 60, 90, 120, 200), trial_s: float = 8.0,
                        n_boot: int = 500, alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """
    For each subject, resample n trials (with replacement) from that subject's
    real per-trial correctness and test against the exact binomial threshold
    for n. Reports the share of subjects that reach significance.
    ``trial_s``: seconds per trial incl. rest (PhysioNet imagery runs: ~2 min /
    15 trials = 8 s; do weryfikacji).  Minutes are therefore n * trial_s / 60.
    """
    rng = np.random.RandomState(seed)
    c = {s: g.correct.values for s, g in trials.groupby("subject")}
    rows = []
    for n in ns:
        thr = binomial_threshold(n, alpha)
        p_sig = {s: float((rng.choice(v, size=(n_boot, n)).mean(1) >= thr).mean()) for s, v in c.items()}
        acc = {s: v.mean() for s, v in c.items()}
        good = [p_sig[s] for s in c if acc[s] >= 0.7]
        rows.append(dict(n_trials=n, minutes=n * trial_s / 60, threshold=thr,
                         frac_subjects_significant=float(np.mean(list(p_sig.values()))),
                         frac_significant_among_ge70=float(np.mean(good)) if good else np.nan))
    return pd.DataFrame(rows)
