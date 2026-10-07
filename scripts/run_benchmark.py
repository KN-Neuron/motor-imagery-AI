"""N-LNSO benchmark on PhysioNet. Writes per_subject.csv, per_trial.csv, run_meta.json.

    python scripts/run_benchmark.py --config configs/benchmark.yaml --out results/main
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import mne
import warnings
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.data import build_epochs, load_raw  # noqa: E402
from src.eval.nlnso import run_nlnso  # noqa: E402
from src.eval.resume import cfg_hash, run_or_load  # noqa: E402
from src.eval.pipelines import make_registry  # noqa: E402
from src.utils import set_seeds  # noqa: E402


mne.set_log_level("WARNING")
warnings.filterwarnings("once")  # one line per distinct warning, not one per model


def git_hash():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--data-dir", default=None, help="local PhysioNet EDF directory (overrides config)")
    ap.add_argument("--n-subjects", type=int, default=None, help="debug: first N subjects")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.data_dir:
        cfg["data"]["data_dir"] = a.data_dir
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    set_seeds(cfg["seed"])

    raw = load_raw(cfg["data"])
    if a.n_subjects:
        raw = dict(sorted(raw.items())[: a.n_subjects])
    pp, ev = cfg["preprocessing"], cfg["eval"]
    reg = make_registry(epochs=ev["epochs"])
    per_all, trials_all, metas = [], [], {}
    h = cfg_hash({**cfg, "_n_subjects": a.n_subjects})  # data_dir/n-subjects changes invalidate parts
    n_grid = len(ev["grid"])
    for i, (pipe_name, norm) in enumerate(ev["grid"], 1):
        label = f"{pipe_name}|{norm}"
        X, y, s, ch, meta = build_epochs(raw, pp["band"], pp["tmin"], pp["tmax"], norm,
                                         pp["channels"], cfg["data"].get("cache_dir"),
                                         eog_regress=pp.get("eog_regress"))
        print(f"=== {i}/{n_grid} {label} X={X.shape}", flush=True)
        per, trials = run_or_load(out, label, h, lambda: run_nlnso(
            X, y, s, reg[pipe_name], label, n_outer=ev["n_outer"], seeds=ev["seeds"],
            val_frac=ev["val_frac"], split_seed=cfg["seed"], log=lambda m: print(m, flush=True)))
        per_all.append(per); trials_all.append(trials); metas[label] = meta.to_dict()
        pd.concat(per_all).to_csv(out / "per_subject.csv", index=False)  # incremental save
    pd.concat(trials_all).to_csv(out / "per_trial.csv", index=False)
    json.dump({"commit": git_hash(), "config": cfg, "preproc_meta": metas}, open(out / "run_meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
