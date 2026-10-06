"""Power analysis from REAL per-subject results (results/main/per_subject.csv), or from stated assumptions.

    python scripts/power_analysis.py --per-subject results/main/per_subject.csv --ref "eegnet|euclidean_alignment"
    python scripts/power_analysis.py --assume-sd 0.10      # no data: assumption-based table
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.stats import binomial_threshold, n_subjects_paired  # noqa: E402


def n_one_sample_above(mu, sd, null, alpha=0.05, power=0.8):
    return n_subjects_paired(sd, mu - null, alpha, power)


def table(sd_acc, sd_diff, source):
    L = [f"Źródło SD: {source}\n", "| scenariusz | wynik |", "|---|---|"]
    for n in (45, 100, 200):
        L.append(f"| próg istotności per osoba, {n} prób (p<0,05) | {100 * binomial_threshold(n):.1f}% |")
    for mu in (0.60, 0.65, 0.70):
        L.append(f"| osób do wykazania średniej {mu:.0%} > 50% (SD {100*sd_acc:.1f} pp, moc 0,8) | {n_one_sample_above(mu, sd_acc, 0.5)} |")
    for delta in (0.02, 0.05, 0.08):
        L.append(f"| osób do wykrycia różnicy {100*delta:.0f} pp między pipeline'ami (SD różnic {100*sd_diff:.1f} pp) | {n_subjects_paired(sd_diff, delta)} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject"); ap.add_argument("--ref"); ap.add_argument("--assume-sd", type=float)
    a = ap.parse_args()
    if a.per_subject:
        df = pd.read_csv(a.per_subject).pivot(index="subject", columns="pipeline", values="acc")
        sd_acc = df[a.ref].std()
        sd_diff = np.mean([(df[c] - df[a.ref]).std() for c in df if c != a.ref]) if df.shape[1] > 1 else sd_acc
        print(table(sd_acc, sd_diff, f"zmierzone na PhysioNet N-LNSO ({len(df)} osób)"))
    else:
        print(table(a.assume_sd, a.assume_sd, f"ZAŁOŻENIE {a.assume_sd}, do zastąpienia wartością z PhysioNet"))


if __name__ == "__main__":
    main()
