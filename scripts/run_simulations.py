"""Pseudo-BrainAccess degradations, few-shot calibration, trials-needed curve (all PhysioNet).

    python scripts/run_simulations.py --config configs/benchmark.yaml --out results/sim [--trials results/main/per_trial.csv]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import mne
import warnings
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_benchmark import git_hash  # noqa: E402
from src.eval.data import build_epochs, load_raw  # noqa: E402
from src.eval.degradation import make_degradations  # noqa: E402
from src.eval.montages import MONTAGES  # noqa: E402
from src.eval.pipelines import make_registry  # noqa: E402
from src.eval.study import degradation_study, degradation_table, fewshot_study, trials_needed_curve  # noqa: E402


mne.set_log_level("WARNING")
warnings.filterwarnings("once")  # one line per distinct warning, not one per model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--out", default="results/sim")
    ap.add_argument("--trials", default=None, help="per_trial.csv from run_benchmark for the trials curve")
    ap.add_argument("--data-dir", default=None, help="local PhysioNet EDF directory (overrides config)")
    ap.add_argument("--n-subjects", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.data_dir:
        cfg["data"]["data_dir"] = a.data_dir
    sim, pp, ev = cfg["simulation"], cfg["preprocessing"], cfg["eval"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    raw = load_raw(cfg["data"])
    if a.n_subjects:
        raw = dict(sorted(raw.items())[: a.n_subjects])
    cache = cfg["data"].get("cache_dir")
    X, y, s, ch, _ = build_epochs(raw, pp["band"], pp["tmin"], pp["tmax"], "none", pp["channels"], cache)
    Xb, _, sb, _, _ = build_epochs(raw, sim["alt_band"], pp["tmin"], pp["tmax"], "none", pp["channels"], cache)
    assert Xb.shape == X.shape and (sb == s).all(), "band variant epochs are not aligned"
    reg = make_registry(epochs=ev["epochs"])
    mk = reg[sim["pipeline"]]

    degs = make_degradations(ch, cfg["data"]["sfreq"], {m: MONTAGES[m] for m in sim["montages"]})
    df = degradation_study(X, y, s, ch, cfg["data"]["sfreq"], mk, degs, tuple(sim["train_norms"]),
                           {tuple(sim["alt_band"]): Xb}, tuple(pp["band"]), ev["n_outer"], cfg["seed"])
    df.to_csv(out / "degradation_per_subject.csv", index=False)
    degradation_table(df).to_csv(out / "degradation_table.csv", index=False)

    fs = fewshot_study(X, y, s, mk, tuple(sim["fewshot_ks"]), n_outer=ev["n_outer"], seed=cfg["seed"])
    fs.to_csv(out / "fewshot_per_subject.csv", index=False)

    if a.trials:
        t = pd.read_csv(a.trials)
        t = t[t.pipeline == ev["reference"]]
        trials_needed_curve(t, trial_s=sim["trial_seconds"]).to_csv(out / "trials_needed.csv", index=False)
    json.dump({"commit": git_hash(), "config": cfg}, open(out / "run_meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
