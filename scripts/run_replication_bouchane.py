"""Bouchane et al. 2025 replication: one model, three split schemes (src/replication/bouchane.py).
Writes <out>/<experiment>_<hash>_trials.csv per experiment and <out>/summary.txt.

    python scripts/run_replication_bouchane.py --config configs/replication_bouchane.yaml --out results/replication_bouchane
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.pipelines import TorchPipeline  # noqa: E402
from src.eval.resume import cfg_hash  # noqa: E402
from src.eval.stats import bootstrap_ci  # noqa: E402
from src.replication import bouchane as B  # noqa: E402
from src.utils import set_seeds  # noqa: E402

mne.set_log_level("WARNING")
warnings.filterwarnings("once")


def load_all(data_cfg, n_max):
    from src.data.loader import find_edf_files
    files = (find_edf_files(data_cfg["data_dir"], B.RUNS) if data_cfg.get("data_dir")
             else __import__("src.data.loading", fromlist=["x"]).download_dataset(desired_runs=B.RUNS, exclude=set()))
    sids = [s for s in sorted(files) if s not in set(data_cfg.get("exclude") or [])][:n_max]
    cache = Path(data_cfg["cache_dir"]) if data_cfg.get("cache_dir") else None
    out = {}
    for sid in sids:
        f = cache / f"{sid}_{cfg_hash(sorted(files[sid]))}.npz" if cache else None
        if f is not None and f.exists():
            z = np.load(f); out[sid] = (z["X"], z["y"], z["t"]); continue
        try:
            out[sid] = B.load_subject(files[sid])
        except Exception as e:  # logged, never silent
            print(f"[data] skipped {sid}: {type(e).__name__}: {e}", flush=True); continue
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, X=out[sid][0], y=out[sid][1], t=out[sid][2])
    print(f"[data] {len(out)} subjects loaded", flush=True)
    return out


def run_experiment(data, exp, tr_cfg, seed):
    sids = sorted(data)[: exp["subjects"]] if exp["subjects"] else sorted(data)
    Xs, ys, ss, ts, ps = [], [], [], [], []
    for k, sid in enumerate(sids):
        X, y, t = data[sid]
        Xi, yi, ti, pi = B.to_pairs(X, y, t)
        Xs.append(Xi); ys.append(yi); ss.append(np.full(len(yi), int(sid))); ts.append(k * 100000 + ti); ps.append(pi)
    X, y, s, t, p = (np.concatenate(a) for a in (Xs, ys, ss, ts, ps))
    m = B.task_mask(y, exp["task"])
    X, y, s, t, p = X[m], y[m], s[m], t[m], p[m]
    rows, t0 = [], time.time()
    for fold, (tr, va, te) in enumerate(B.splits(exp["scheme"], y, s, t, exp["n_folds"], seed)):
        Xtr, ytr = (B.smote(X[tr], y[tr], 5, seed + fold) if tr_cfg["smote"] else (X[tr], y[tr]))
        pipe = TorchPipeline(lambda c, k, T: B.CNNGRU(n_classes=k, in_ch=c), epochs=tr_cfg["epochs"],
                             lr=tr_cfg["lr"], batch_size=tr_cfg["batch_size"], seed=seed)
        pipe.fit(Xtr, ytr, X[va], y[va])
        pred = pipe.predict_proba(X[te]).argmax(1)
        rows.append(pd.DataFrame(dict(fold=fold, subject=s[te], trial=t[te], pair=p[te], y=y[te], pred=pred)))
        print(f"[{exp['name']}] fold {fold + 1}/{exp['n_folds']} acc={np.mean(pred == y[te]):.3f} "
              f"train={len(ytr)} (smote) test={len(te)}, {time.time() - t0:.0f}s elapsed", flush=True)
    return pd.concat(rows)


def summarize(name, exp, df):
    n_cls = 2 if exp["task"] == "lr" else 5
    yy, pp = df.y.values, df.pred.values
    per = df.assign(c=yy == pp).groupby("subject").c.mean()
    lo, hi = bootstrap_ci(per.values)
    bal = np.mean([np.mean(pp[yy == c] == c) for c in range(n_cls) if (yy == c).any()])
    major = np.bincount(yy).max() / len(yy)
    return (f"{name:<16} {len(per):>4} {exp['task']:<7} {exp['scheme']:<16} {100 * np.mean(yy == pp):>6.1f} "
            f"{100 * bal:>6.1f} {100 * B.ovr_accuracy(yy, pp, n_cls):>6.1f} "
            f"{100 * per.mean():>6.1f} [{100 * lo:.1f}, {100 * hi:.1f}] {100 * major:>6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/replication_bouchane.yaml")
    ap.add_argument("--out", default="results/replication_bouchane")
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.data_dir:
        cfg["data"]["data_dir"] = a.data_dir
    set_seeds(cfg["seed"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n_max = None if any(e["subjects"] is None for e in cfg["experiments"]) else max(e["subjects"] for e in cfg["experiments"])
    data = load_all(cfg["data"], n_max)
    lines = [f"{'experiment':<16} {'N':>4} {'task':<7} {'scheme':<16} {'acc':>6} {'bal':>6} {'ovr':>6} "
             f"{'per-subject [95% CI]':>22} {'major':>6}"]
    for exp in cfg["experiments"]:
        h = cfg_hash({"exp": exp, "train": cfg["train"], "data": cfg["data"], "seed": cfg["seed"]})
        f = out / f"{exp['name']}_{h}_trials.csv"
        print(f"=== {exp['name']} {exp}", flush=True)
        if f.exists():
            print(f"[{exp['name']}] done earlier, loading {f}", flush=True)
            df = pd.read_csv(f)
        else:
            df = run_experiment(data, exp, cfg["train"], cfg["seed"])
            df.to_csv(f, index=False)
        lines.append(summarize(exp["name"], exp, df))
        (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
