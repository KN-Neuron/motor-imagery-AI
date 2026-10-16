import json
import subprocess
import sys

import numpy as np
import pandas as pd

from src.eval.smr import smr_peak


def test_smr_peak_detects_alpha_over_pink_noise():
    rng = np.random.RandomState(0)
    t = np.arange(120 * 160) / 160
    noise = np.cumsum(rng.randn(t.size)) * 0.01
    with_peak = noise + 0.5 * np.sin(2 * np.pi * 11 * t)
    assert smr_peak(with_peak, 160.0) > smr_peak(noise, 160.0) + 0.5


def test_make_report_end_to_end(tmp_path):
    rng = np.random.RandomState(0)
    rows = [(p, s, 45, int(45 * a), a) for p, mu in (("eegnet|euclidean_alignment", .8), ("csp_lda|none", .65))
            for s, a in enumerate(np.clip(rng.normal(mu, .1, 20), .3, 1))]
    m = tmp_path / "main"; m.mkdir()
    pd.DataFrame(rows, columns=["pipeline", "subject", "n_trials", "n_correct", "acc"]).to_csv(m / "per_subject.csv", index=False)
    json.dump({"commit": "abc123", "config": {"eval": {"reference": "eegnet|euclidean_alignment"}}}, open(m / "run_meta.json", "w"))
    out = tmp_path / "docs" / "results.md"
    subprocess.run([sys.executable, "scripts/make_report.py", "--main", str(m), "--sim", str(tmp_path / "none"),
                    "--out", str(out)], check=True)
    txt = out.read_text()
    assert "abc123" in txt and "csp_lda|none" in txt and "Holma" in txt
