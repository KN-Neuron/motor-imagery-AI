"""Print per-subject accuracy summaries (bootstrap CI over subjects) for one or more result dirs.

    python scripts/summarize_results.py results/main results/legacy_replica ...
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.stats import summarize  # noqa: E402


def main(dirs):
    print(f"{'run':<28} {'pipeline':<36} {'N':>4} {'mean [95% CI]':>20} {'median':>7} {'%sig':>6} {'%>=70':>6}")
    for d in dirs:
        f = Path(d) / "per_subject.csv"
        if not f.exists():
            print(f"{Path(d).name:<28} (missing {f})")
            continue
        for pipe, g in pd.read_csv(f).groupby("pipeline"):
            s = summarize(g.acc, g.n_correct.round().astype(int), g.n_trials)
            lo, hi = s["mean_ci"]
            print(f"{Path(d).name:<28} {pipe:<36} {s['n_subjects']:>4} "
                  f"{100 * s['mean']:>6.1f} [{100 * lo:.1f}, {100 * hi:.1f}] {100 * s['median']:>7.1f} "
                  f"{100 * s['frac_significant']:>6.1f} {100 * s['frac_ge_70']:>6.1f}")


if __name__ == "__main__":
    main(sys.argv[1:])
