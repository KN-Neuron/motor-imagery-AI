"""Generate docs/results.md and figures ONLY from CSVs written by run_benchmark / run_simulations.

    python scripts/make_report.py --main results/main --sim results/sim --out docs/results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.stats import binomial_threshold, holm, n_subjects_paired, paired_wilcoxon, summarize  # noqa: E402


def pct(x):
    return f"{100 * x:.1f}"


def ci(t):
    return f"[{pct(t[0])}, {pct(t[1])}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results/main")
    ap.add_argument("--sim", default="results/sim")
    ap.add_argument("--out", default="docs/results.md")
    a = ap.parse_args()
    main_dir, sim_dir, out = Path(a.main), Path(a.sim), Path(a.out)
    meta = json.load(open(main_dir / "run_meta.json"))
    per = pd.read_csv(main_dir / "per_subject.csv")
    ref = meta["config"]["eval"]["reference"]
    L = [f"# Wyniki (generowane automatycznie)\n",
         f"- commit: `{meta['commit']}`",
         "- konfiguracja:\n\n```json\n" + json.dumps(meta["config"], indent=1) + "\n```\n",
         "Protokół: zagnieżdżona walidacja po osobach (N-LNSO), każda osoba testowana raz, wybór checkpointu na "
         "walidacji z puli treningowej, normalizacja per osoba bez etykiet. Jednostką niezależną jest osoba; "
         "przedziały ufności to bootstrap po osobach (95%). Wyniki w %.\n",
         "## Porównanie modeli\n",
         "| pipeline | N os. | średnia [CI] | mediana [CI] | IQR | % os. istotnych (binom., p<0,05) | % os. >= 70% |",
         "|---|---|---|---|---|---|---|"]
    wide = per.pivot(index="subject", columns="pipeline", values="acc")
    for name, g in per.groupby("pipeline"):
        s = summarize(g.acc, g.n_correct.round(), g.n_trials)
        L.append(f"| {name} | {s['n_subjects']} | {pct(s['mean'])} {ci(s['mean_ci'])} | {pct(s['median'])} "
                 f"{ci(s['median_ci'])} | {pct(s['q25'])} do {pct(s['q75'])} | {pct(s['frac_significant'])} | {pct(s['frac_ge_70'])} |")
    n_tr = int(per.n_trials.median())
    L.append(f"\nPróg istotności per osoba dla mediany liczby prób ({n_tr}): {pct(binomial_threshold(n_tr))}% (p<0,05, rozkład dwumianowy).\n")
    if ref in wide:
        L += [f"## Testy parowane względem `{ref}` (Wilcoxon, korekta Holma)\n",
              "| pipeline | średnia różnica (pp) | CI różnicy | p | p (Holm) |", "|---|---|---|---|---|"]
        res = {c: paired_wilcoxon(wide[c].dropna(), wide[ref].loc[wide[c].dropna().index]) for c in wide if c != ref}
        adj = holm({c: r["p"] for c, r in res.items()})
        for c, r in res.items():
            L.append(f"| {c} | {100 * r['mean_diff']:.2f} | {ci(r['diff_ci'])} | {r['p']:.4f} | {adj[c]:.4f} |")
        diffs = [(wide[c] - wide[ref]).dropna().std() for c in wide if c != ref]
        if diffs:
            L.append(f"\nMoc: przy SD różnic per osoba = {100 * np.mean(diffs):.1f} pp potrzeba "
                     f"{n_subjects_paired(np.mean(diffs), 0.02)} osób na wykrycie różnicy 2 pp, "
                     f"{n_subjects_paired(np.mean(diffs), 0.05)} osób na 5 pp (t sparowany, alfa 0,05, moc 0,8).\n")
    # histogram
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dir = out.parent / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
        g = per[per.pipeline == ref] if ref in set(per.pipeline) else per
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(100 * g.acc, bins=15, color="#4477aa")
        ax.axvline(100 * binomial_threshold(n_tr), color="k", ls="--", label="próg istotności")
        ax.axvline(70, color="r", ls=":", label="70%")
        ax.set_xlabel("dokładność per osoba (%)"); ax.set_ylabel("liczba osób"); ax.legend()
        fig.tight_layout(); fig.savefig(fig_dir / "per_subject_hist.png", dpi=150)
        L.append(f"![histogram]({'figures/per_subject_hist.png'})\n")
    except Exception as e:  # plotting must never break the tables
        L.append(f"(histogram niedostępny: {e})\n")
    # simulations
    t = sim_dir / "degradation_table.csv"
    if t.exists():
        d = pd.read_csv(t)
        L += ["## Symulacja pseudo-BrainAccess: degradacja a spadek dokładności\n",
              "Spadek = dokładność bazowa minus dokładność po degradacji, parowany po osobach, CI bootstrap po osobach. "
              "`guard` = czy walidacja zgodności preprocessingu wykrywa niezgodność.\n",
              "| trening: normalizacja | degradacja | dokładność | spadek (pp) | CI spadku (pp) | guard |", "|---|---|---|---|---|---|"]
        for _, r in d.iterrows():
            L.append(f"| {r.train_norm} | {r.degradation} | {pct(r.acc)} | {100 * r['drop']:.1f} | "
                     f"[{100 * r.drop_ci_lo:.1f}, {100 * r.drop_ci_hi:.1f}] | {'tak' if r.guard_detects else 'nie'} |")
        L.append("")
    t = sim_dir / "fewshot_per_subject.csv"
    if t.exists():
        f = pd.read_csv(t)
        L += ["## Kalibracja few-shot (k prób na klasę z nowej osoby)\n",
              "| trening: normalizacja | metoda | k | średnia [CI] |", "|---|---|---|---|"]
        for (tn, m, k), g in f.groupby(["train_norm", "method", "k_per_class"]):
            s = summarize(g.acc)
            L.append(f"| {tn} | {m} | {k} | {pct(s['mean'])} {ci(s['mean_ci'])} |")
        L.append("")
    t = sim_dir / "trials_needed.csv"
    if t.exists():
        L += ["## Ile prób (minut) z jednej osoby daje wynik istotny\n",
              pd.read_csv(t).to_markdown(index=False, floatfmt=".3f"), ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
