"""Figures and a markdown index for a whole results tree (scripts/run_all.sh writes one).

    python scripts/make_figures.py --root results/final

Reads every <root>/<run>/per_subject.csv (+ history.csv, selection.csv) and
<root>/replication_bouchane/*_trials.csv; writes <root>/figures/*.png and <root>/REPORT.md.
Missing inputs are skipped, never invented.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.stats import binomial_threshold, bootstrap_ci, holm, paired_wilcoxon  # noqa: E402
from src.replication.bouchane import ovr_accuracy  # noqa: E402

REF = "eegnet|zscore_subject_channel"
STORY = [  # (run, label): the eye-movement argument, EEGNet + zscore everywhere
    ("legacy_wideband", "64 kan., 0,5-45 Hz"),
    ("legacy_eye_heog", "tylko F7/F8, <4 Hz"),
    ("legacy_wb_motor21", "21 ruchowych, 0,5-45 Hz"),
    ("mi_ctrl_lowfreq_raw", "21 ruchowych, <4 Hz"),
    ("mi_ctrl_lowfreq_reg", "21 ruchowych, <4 Hz, po regresji EOG"),
    ("legacy_motor21", "21 ruchowych, 7-30 Hz"),
    ("mi_reg_mu_beta_nocue", "benchmark MI (po regresji, 7-30 Hz, bez cue)"),
]
FAMILY_COLORS = {"main": "#555555", "legacy": "#c0504d", "mi": "#4f81bd", "mm": "#9bbb59", "other": "#8064a2"}
THRESH = binomial_threshold(45)


def md_table(df, floatfmt=".3g"):
    """Markdown table without the optional ``tabulate`` dependency of DataFrame.to_markdown."""
    fmt = lambda v: format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _family(run):
    for k in ("legacy", "mm", "mi", "main"):
        if run.startswith(k):
            return k
    return "other"


def load_runs(root):
    runs = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / "per_subject.csv").exists():
            runs[d.name] = pd.read_csv(d / "per_subject.csv")
    return runs


def fig_overview(runs, out):
    rows = []
    for run, df in runs.items():
        for pipe, g in df.groupby("pipeline"):
            lo, hi = bootstrap_ci(g.acc.values)
            rows.append((run, pipe, g.acc.mean(), lo, hi))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(9, 0.22 * len(rows) + 1.5))
    for i, (run, pipe, m, lo, hi) in enumerate(rows[::-1]):
        c = FAMILY_COLORS[_family(run)]
        ax.plot([lo, hi], [i, i], color=c, lw=2); ax.plot(m, i, "o", color=c, ms=4)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{r}  {p}" for r, p, *_ in rows[::-1]], fontsize=6)
    ax.axvline(.5, color="k", lw=.8, ls=":"); ax.axvline(THRESH, color="k", lw=.8, ls="--")
    ax.set_xlabel("dokładność (średnia po osobach, 95% CI bootstrap)")
    ax.set_title(f"Wszystkie biegi (linia przerywana: próg istotności dla 45 prób = {THRESH:.1%})", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return pd.DataFrame(rows, columns=["run", "pipeline", "mean", "ci_lo", "ci_hi"])


def fig_eye_story(runs, out):
    items = [(lab, runs[r].query("pipeline == @REF").acc.values) for r, lab in STORY
             if r in runs and (runs[r].pipeline == REF).any()]
    if not items:
        return False
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rng = np.random.RandomState(0)
    for i, (lab, a) in enumerate(items):
        ax.boxplot(a, positions=[i], widths=.5, showfliers=False)
        ax.plot(i + rng.uniform(-.18, .18, len(a)), a, ".", alpha=.35, ms=4)
        ax.text(i, 1.01, f"{a.mean():.1%}", ha="center", fontsize=8)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([l for l, _ in items], rotation=20, ha="right", fontsize=8)
    ax.axhline(.5, color="k", lw=.8, ls=":"); ax.axhline(THRESH, color="k", lw=.8, ls="--")
    ax.set_ylim(.2, 1.05); ax.set_ylabel("dokładność per osoba")
    ax.set_title("Ile wyniku L/R to oczy (EEGNet + zscore, wyobrażenie)", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return True


def fig_mi_vs_mm(runs, out, mi="mi_reg_mu_beta_nocue", mm="mm_reg_mu_beta_nocue"):
    if mi not in runs or mm not in runs:
        return None
    pipes = sorted(set(runs[mi].pipeline) & set(runs[mm].pipeline))
    if not pipes:
        return None
    fig, axes = plt.subplots(1, len(pipes), figsize=(3.2 * len(pipes), 3.4), squeeze=False)
    tests = {}
    for ax, p in zip(axes[0], pipes):
        a = runs[mi].query("pipeline == @p").set_index("subject").acc
        b = runs[mm].query("pipeline == @p").set_index("subject").acc
        common = a.index.intersection(b.index)
        w = paired_wilcoxon(b[common].values, a[common].values)
        tests[p] = (len(common), w["mean_diff"], w["p"])
        ax.plot(a[common], b[common], ".", alpha=.5); ax.plot([.3, 1], [.3, 1], "k:", lw=.8)
        ax.set_title(f"{p}\nruch - wyobr. = {w['mean_diff']:+.1%}, p={w['p']:.3g}", fontsize=7)
        ax.set_xlabel("wyobrażenie"); ax.set_ylabel("ruch")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    adj = holm({p: t[2] for p, t in tests.items()})
    return pd.DataFrame([(p, n, d, pv, adj[p]) for p, (n, d, pv) in tests.items()],
                        columns=["pipeline", "N", "mean_diff_mm_minus_mi", "p_wilcoxon", "p_holm"])


def fig_curves(hist, key, out, title):
    groups = sorted(hist[key].unique())
    fig, axes = plt.subplots(2, len(groups), figsize=(3.3 * len(groups), 5), squeeze=False)
    for j, g in enumerate(groups):
        h = hist[hist[key] == g]
        for i, (metric, lab) in enumerate([("loss", "strata"), ("acc", "dokładność")]):
            ax = axes[i, j]
            for split, c in (("train", "C0"), ("val", "C1")):
                s = h.groupby("epoch")[f"{split}_{metric}"]
                m, sd = s.mean(), s.std().fillna(0)
                ax.plot(m.index, m, color=c, label=split); ax.fill_between(m.index, m - sd, m + sd, color=c, alpha=.2)
            ax.axvline(h.groupby(["fold", "seed"]).best_epoch.first().median(), color="k", ls="--", lw=.8)
            ax.set_ylabel(lab, fontsize=8); ax.set_xlabel("epoka", fontsize=8)
            if i == 0:
                ax.set_title(g, fontsize=7)
        axes[0, j].legend(fontsize=7)
    fig.suptitle(f"{title}: średnia ± sd po foldach i seedach; linia = mediana najlepszej epoki", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_best_epochs(hists, out):
    rows = []
    for run, (h, key) in hists.items():
        for g, d in h.groupby(key):
            be = d.groupby(["fold", "seed"]).best_epoch.first()
            rows.append((f"{run}  {g}", be.values, int(d.epoch.max()) + 1))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(rows) + 1.2))
    for i, (lab, be, n_ep) in enumerate(rows):
        ax.plot(be, np.full(len(be), i) + np.random.RandomState(i).uniform(-.15, .15, len(be)), "o", ms=3, alpha=.7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=6)
    ax.set_xlabel("najlepsza epoka (wybrana na walidacji); 0 = model nie poprawił się po pierwszej epoce")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return pd.DataFrame([(lab, len(be), float(np.median(be)), int((be == 0).sum()), n_ep) for lab, be, n_ep in rows],
                        columns=["run_pipeline", "fits", "median_best_epoch", "n_best_epoch_0", "epochs"])


def fig_tuning(sel, out):
    pipes = sorted(sel.pipeline.unique())
    fig, axes = plt.subplots(1, len(pipes), figsize=(3.5 * len(pipes), 3.2), squeeze=False)
    for ax, p in zip(axes[0], pipes):
        s = sel[sel.pipeline == p]
        order = s.groupby("params").score.mean().sort_values(ascending=False).index
        x = {k: i for i, k in enumerate(order)}
        ax.plot(s.params.map(x), s.score, ".", alpha=.5, color="C0")
        t = s[s.selected.astype(bool)]
        ax.plot(t.params.map(x), t.score, "o", mfc="none", color="C3", label="wybrany w foldzie")
        ax.set_title(p, fontsize=7); ax.set_xlabel("kandydat (wg średniej)", fontsize=7); ax.set_ylabel("wynik wewn.", fontsize=7)
        ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def replication_table(rep):
    rows = []
    for f in sorted(rep.glob("*_trials.csv")):
        name = f.name.rsplit("_", 2)[0]
        df = pd.read_csv(f)
        k = int(df.y.max()) + 1
        for seed, g in df.groupby("seed") if "seed" in df else [(0, df)]:
            yy, pp = g.y.values, g.pred.values
            bal = np.mean([np.mean(pp[yy == c] == c) for c in range(k) if (yy == c).any()])
            rows.append((name, k, seed, np.mean(yy == pp), bal, ovr_accuracy(yy, pp, k)))
    return pd.DataFrame(rows, columns=["experiment", "n_classes", "seed", "acc", "bal", "ovr"])


def fig_replication(tab, out):
    agg = tab.groupby(["experiment", "n_classes"], sort=False)[["acc", "bal", "ovr"]].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(agg) + 2), 4.3))
    x = np.arange(len(agg))
    for i, (m, c) in enumerate((("acc", "C0"), ("bal", "C2"), ("ovr", "C3"))):
        ax.bar(x + (i - 1) * .27, agg[(m, "mean")], .27, yerr=agg[(m, "std")].fillna(0), color=c, label=m)
    for xi, k in zip(x, agg.n_classes):
        ax.plot([xi - .45, xi + .45], [1 / k, 1 / k], "k--", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(agg.experiment, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title("Replikacja Bouchane et al. 2025\nbal = dokładność zbalansowana, kreska = szansa, ovr = miara z pracy "
                 "(jeden-kontra-reszta); słupki błędu = sd po seedach", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/final")
    root = Path(ap.parse_args().root)
    figs = root / "figures"; figs.mkdir(parents=True, exist_ok=True)
    md = [f"# Raport: {root}", "", "Wygenerowane przez `scripts/make_figures.py`. Średnie po osobach, 95% CI bootstrap po osobach.", ""]
    runs = load_runs(root)

    ov = fig_overview(runs, figs / "overview.png")
    if ov is not None:
        md += ["## Wszystkie biegi", "", "![](figures/overview.png)", "",
               md_table(ov.assign(**{c: ov[c].map("{:.1%}".format) for c in ("mean", "ci_lo", "ci_hi")})), ""]
    if fig_eye_story(runs, figs / "eye_story.png"):
        md += ["## Ruchy oczu", "", "![](figures/eye_story.png)", ""]
    mm = fig_mi_vs_mm(runs, figs / "mi_vs_mm.png")
    if mm is not None:
        md += ["## Ruch vs wyobrażenie (te same osoby, Wilcoxon, Holm)", "", "![](figures/mi_vs_mm.png)", "",
               md_table(mm, ".4g"), ""]

    hists = {}
    for run in runs:
        if (root / run / "history.csv").exists():
            h = pd.read_csv(root / run / "history.csv"); hists[run] = (h, "pipeline")
            fig_curves(h, "pipeline", figs / f"curves_{run}.png", run)
    rep = root / "replication_bouchane"
    if (rep / "history.csv").exists():
        h = pd.read_csv(rep / "history.csv"); hists["replication_bouchane"] = (h, "experiment")
        fig_curves(h, "experiment", figs / "curves_replication_bouchane.png", "replication_bouchane")
    if hists:
        md += ["## Przebieg treningu", ""] + [f"![](figures/curves_{r}.png)" for r in hists] + [""]
        be = fig_best_epochs(hists, figs / "best_epochs.png")
        md += ["### Najlepsze epoki", "", "![](figures/best_epochs.png)", "", md_table(be), ""]

    for run in runs:
        if (root / run / "selection.csv").exists():
            fig_tuning(pd.read_csv(root / run / "selection.csv"), figs / f"tuning_{run}.png")
            md += [f"## Strojenie: {run}", "", f"![](figures/tuning_{run}.png)", ""]

    if rep.exists() and any(rep.glob("*_trials.csv")):
        tab = replication_table(rep)
        agg = fig_replication(tab, figs / "replication.png")
        agg.columns = ["_".join(c).strip("_") for c in agg.columns]
        md += ["## Replikacja Bouchane et al. 2025", "", "![](figures/replication.png)", "",
               md_table(agg, ".3f"), ""]
        if (rep / "summary.txt").exists():
            md += ["```", (rep / "summary.txt").read_text().rstrip(), "```", ""]
    (root / "REPORT.md").write_text("\n".join(md) + "\n")
    print(f"wrote {root / 'REPORT.md'} and {len(list(figs.glob('*.png')))} figures in {figs}")


if __name__ == "__main__":
    main()
