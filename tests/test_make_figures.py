"""scripts/make_figures.py on a synthetic results tree: every figure and the report index are produced."""
import subprocess
import sys

import numpy as np
import pandas as pd


def _per(d, pipes, base):
    rng = np.random.RandomState(len(str(d)))
    rows = [dict(pipeline=p, subject=s, n_trials=45, n_correct=0, acc=np.clip(base + rng.randn() * .08, 0, 1))
            for p in pipes for s in range(1, 21)]
    df = pd.DataFrame(rows); df["n_correct"] = (df.acc * 45).round()
    d.mkdir(parents=True); df.to_csv(d / "per_subject.csv", index=False)


def _hist(d, pipes, **tags):
    rows = [dict(pipeline=p, fold=f, seed=s, epoch=e, train_loss=1 / (e + 1), train_acc=.5 + e / 20,
                 val_loss=.7 - e / 100, val_acc=.5 + e / 40, lr=1e-3, best_epoch=4, **tags)
            for p in pipes for f in range(2) for s in range(2) for e in range(5)]
    pd.DataFrame(rows).to_csv(d / "history.csv", index=False)


def test_make_figures_on_synthetic_tree(tmp_path):
    root = tmp_path / "final"
    for run, base in [("legacy_wideband", .83), ("legacy_eye_heog", .81), ("mi_reg_mu_beta_nocue", .67),
                      ("mi_ctrl_lowfreq_raw", .8), ("mi_ctrl_lowfreq_reg", .62), ("mm_reg_mu_beta_nocue", .69)]:
        _per(root / run, ["eegnet|zscore_subject_channel", "ts_lr|euclidean_alignment"], base)
    _hist(root / "mi_reg_mu_beta_nocue", ["eegnet|zscore_subject_channel"])
    _per(root / "mi_tuned", ["eegnet_tuned|zscore_subject_channel"], .67)
    _hist(root / "mi_tuned", ["eegnet_tuned|zscore_subject_channel"], params='{"lr": 0.001}')
    pd.DataFrame([dict(pipeline="eegnet_tuned|zscore_subject_channel", fold=f, params=f'{{"lr": {lr}}}',
                       score=.6 + lr, selected=lr == .001) for f in range(2) for lr in (.001, .0005)]
                 ).to_csv(root / "mi_tuned" / "selection.csv", index=False)
    rep = root / "replication_bouchane"; rep.mkdir()
    rng = np.random.RandomState(0)
    for name, task in [("paper7_random", "5class"), ("all_subject_lr", "lr")]:
        k = 5 if task == "5class" else 2
        y = rng.randint(k, size=600)
        pd.DataFrame(dict(seed=np.repeat([0, 1], 300), fold=0, subject=np.tile(np.arange(1, 7), 100), trial=0,
                          pair=0, y=y, pred=rng.randint(k, size=600))).to_csv(rep / f"{name}_abc_trials.csv", index=False)
    hr = pd.read_csv(root / "mi_reg_mu_beta_nocue" / "history.csv").drop(columns="pipeline").assign(experiment="paper7_random")
    hr.to_csv(rep / "history.csv", index=False)
    (rep / "summary.txt").write_text("experiment ...\n")

    r = subprocess.run([sys.executable, "scripts/make_figures.py", "--root", str(root)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    figs = root / "figures"
    for f in ("overview.png", "eye_story.png", "mi_vs_mm.png", "curves_mi_reg_mu_beta_nocue.png",
              "curves_mi_tuned.png", "best_epochs.png", "tuning_mi_tuned.png", "replication.png",
              "curves_replication_bouchane.png"):
        assert (figs / f).stat().st_size > 1000, f
    rep_md = (root / "REPORT.md").read_text()
    assert "figures/eye_story.png" in rep_md and "mi_reg_mu_beta_nocue" in rep_md and "Wilcoxon" in rep_md
