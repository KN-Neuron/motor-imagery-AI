"""N-LNSO with hyperparameters selected in the inner loop (src/eval/tuning.py).
Writes per_subject.csv, per_trial.csv, selection.csv, run_meta.json.

    python scripts/run_tuned.py --config configs/mi_tuned.yaml --out results/mi_tuned
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import mne
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_benchmark import git_hash  # noqa: E402
from src.eval.data import build_epochs, load_raw  # noqa: E402
from src.eval.pipelines import SklearnPipeline, eegnet, eegnet_transformer, shallow  # noqa: E402
from src.eval.resume import cfg_hash, run_or_load  # noqa: E402
from src.eval.tuning import run_nlnso_tuned, sample_grid  # noqa: E402
from src.utils import set_seeds  # noqa: E402

mne.set_log_level("WARNING")
warnings.filterwarnings("once")


def factories(epochs):
    return {
        "eegnet": lambda p: eegnet(False, epochs=epochs, **p),
        "shallow": lambda p: shallow(epochs=epochs, **p),
        "eegnet_transformer": lambda p: eegnet_transformer(epochs=epochs, **p),
        "csp_lda": lambda p: (lambda seed: SklearnPipeline("csp_lda", seed, **p)),
        "ts_lr": lambda p: (lambda seed: SklearnPipeline("ts_lr", seed, **p)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mi_tuned.yaml")
    ap.add_argument("--out", default="results/mi_tuned")
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.data_dir:
        cfg["data"]["data_dir"] = a.data_dir
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    set_seeds(cfg["seed"])
    raw = load_raw(cfg["data"])
    pp, ev = cfg["preprocessing"], cfg["eval"]
    fac = factories(ev["epochs"])
    per_all, trials_all, sel_all = [], [], []
    for i, t in enumerate(ev["tuned"], 1):
        label = f"{t['name']}_tuned|{t['normalization']}"
        X, y, s, ch, _ = build_epochs(raw, pp["band"], pp["tmin"], pp["tmax"], t["normalization"],
                                      pp["channels"], cfg["data"].get("cache_dir"), eog_regress=pp.get("eog_regress"))
        cands = sample_grid(t["grid"], t["n_candidates"], seed=cfg["seed"], default=t.get("default"))
        print(f"=== {i}/{len(ev['tuned'])} {label} X={X.shape} candidates={len(cands)}", flush=True)
        h = cfg_hash({k: v for k, v in cfg.items() if k != "eval"} | {"eval": {**ev, "tuned": t}})
        part = out / "parts" / h
        stem = label.replace("|", "_")

        def fn():
            per, trials, sel = run_nlnso_tuned(
                X, y, s, fac[t["name"]], cands, label, n_outer=ev["n_outer"], seeds=ev["seeds"],
                val_frac=ev["val_frac"], split_seed=cfg["seed"], inner_folds=ev["inner_folds"],
                cache=part / f"{stem}_inner.csv", log=lambda m: print(m, flush=True))
            sel.insert(0, "pipeline", label)
            sel.to_csv(part / f"{stem}_selection.csv", index=False)
            return per, trials

        per, trials = run_or_load(out, label, h, fn)
        per_all.append(per); trials_all.append(trials)
        sel_all.append(pd.read_csv(part / f"{stem}_selection.csv"))
        pd.concat(per_all).to_csv(out / "per_subject.csv", index=False)
        pd.concat(sel_all).to_csv(out / "selection.csv", index=False)
    pd.concat(trials_all).to_csv(out / "per_trial.csv", index=False)
    json.dump({"commit": git_hash(), "config": cfg}, open(out / "run_meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
