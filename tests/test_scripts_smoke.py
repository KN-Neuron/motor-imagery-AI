"""End-to-end smoke test of benchmark -> simulations -> report on synthetic EDF files."""
import subprocess
import sys

import mne
import numpy as np
import yaml

from src.eval.montages import MAXI32, MIDI16

# PhysioNet-style names: 'Fc3.', 'C3..', 'Fp1.'
def _dotted(n):
    s = n[0] + n[1:].lower()
    return s + "." * (4 - len(s))

CH = [_dotted(c) for c in MAXI32]


def _write(path, seed):
    rng = np.random.RandomState(seed)
    sf, n = 160.0, 160 * 125
    data = rng.randn(len(CH), n) * 1e-5
    onsets = np.arange(4, 118, 8.0)
    labels = ["T1" if i % 2 == 0 else "T2" for i in range(len(onsets))]
    for on, l in zip(onsets, labels):  # class info in C3/C4 power
        i0, i1 = int(on * sf), int((on + 4) * sf)
        data[CH.index(_dotted("C3")) if l == "T2" else CH.index(_dotted("C4")), i0:i1] *= 3
    raw = mne.io.RawArray(data, mne.create_info(CH, sf, "eeg"), verbose=False)
    raw.set_annotations(mne.Annotations(onsets, 4.0, labels))
    mne.export.export_raw(path, raw, fmt="edf", overwrite=True, verbose=False)


def test_full_pipeline_on_synthetic_physionet(tmp_path):
    d = tmp_path / "physionet"
    for s in range(1, 11):
        (d / f"S{s:03d}").mkdir(parents=True)
        for r in ("R04", "R08", "R12"):
            _write(d / f"S{s:03d}" / f"S{s:03d}{r}.edf", s * 10 + int(r[1:]))
    cfg = yaml.safe_load(open("configs/benchmark.yaml"))
    cfg["data"].update(data_dir=str(d), cache_dir=str(tmp_path / "cache"), exclude="none")
    cfg["eval"].update(n_outer=2, seeds=[0], epochs=2,
                       grid=[["eegnet", "euclidean_alignment"], ["ts_lr", "euclidean_alignment"], ["csp_lda", "none"]],
                       reference="eegnet|euclidean_alignment")
    cfg["simulation"].update(montages=["midi16"], fewshot_ks=[0, 2], train_norms=["none", "euclidean_alignment"])
    cfg["preprocessing"].update(tmin=0.5, tmax=2.5)
    cf = tmp_path / "cfg.yaml"; yaml.safe_dump(cfg, open(cf, "w"))
    py = sys.executable
    run = lambda *a: subprocess.run([py, *a], check=True, capture_output=True, text=True)
    def run(*a):
        r = subprocess.run([py, *a], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
    run("scripts/run_benchmark.py", "--config", str(cf), "--out", str(tmp_path / "main"))
    run("scripts/run_simulations.py", "--config", str(cf), "--out", str(tmp_path / "sim"),
        "--trials", str(tmp_path / "main" / "per_trial.csv"))
    run("scripts/make_report.py", "--main", str(tmp_path / "main"), "--sim", str(tmp_path / "sim"),
        "--out", str(tmp_path / "docs" / "results.md"))
    txt = (tmp_path / "docs" / "results.md").read_text()
    for needle in ("Porównanie modeli", "pseudo-BrainAccess", "few-shot", "zeros_midi16", "eegnet|euclidean_alignment", "commit"):
        assert needle in txt, needle
