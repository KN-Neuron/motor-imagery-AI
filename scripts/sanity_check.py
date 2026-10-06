"""Does the data carry L/R information? Within-subject CV (signal check only) + cross-subject N-LNSO with cheap models.

    python scripts/sanity_check.py --config configs/benchmark.yaml            # all subjects
    python scripts/sanity_check.py --config configs/benchmark.yaml --n-subjects 15
Prints per-model means; expected for a healthy pipeline: within-subject clearly > 55-60%.
"""
import argparse
import sys
import warnings
from pathlib import Path

import mne
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.data import build_epochs, load_raw  # noqa: E402
from src.eval.nlnso import run_nlnso  # noqa: E402
from src.eval.pipelines import make_registry  # noqa: E402
from src.eval.sanity import within_subject_cv  # noqa: E402

mne.set_log_level("WARNING")
warnings.filterwarnings("once")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--n-subjects", type=int, default=None)
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.data_dir:
        cfg["data"]["data_dir"] = a.data_dir
    raw = load_raw(cfg["data"])
    if a.n_subjects:
        raw = dict(sorted(raw.items())[: a.n_subjects])
    pp = cfg["preprocessing"]
    for band, (tmin, tmax) in [(pp["band"], (pp["tmin"], pp["tmax"])), ((8.0, 30.0), (0.5, 3.5))]:
        X, y, s, ch, _ = build_epochs(raw, band, tmin, tmax, "euclidean_alignment", pp["channels"], cfg["data"].get("cache_dir"))
        print(f"\n== band {band}, window {tmin}-{tmax}s, X={X.shape}, class balance={np.bincount(y).tolist()}")
        for kind in ("csp_lda", "ts_lr"):
            w = np.array(list(within_subject_cv(X, y, s, kind).values()))
            print(f"within-subject CV  {kind:8s} mean {w.mean():.3f}  median {np.median(w):.3f}  "
                  f"subjects>=0.6: {(w >= .6).mean():.0%}")
        per, _ = run_nlnso(X, y, s, make_registry()["ts_lr"], "ts_lr", n_outer=cfg["eval"]["n_outer"], split_seed=cfg["seed"])
        print(f"cross-subject N-LNSO ts_lr   mean {per.acc.mean():.3f}  median {per.acc.median():.3f}  (N={len(per)})")


if __name__ == "__main__":
    main()
