"""Replication of Bouchane et al. 2025 (Sensors 25:1399) under three split schemes."""
import subprocess
import sys

import mne
import numpy as np
import pytest
import yaml

from src.replication import bouchane as B

CH = ["Fc1.", "Fc2.", "Fc3.", "Fc4.", "C3..", "C4..", "C1..", "C2..", "Cp1.", "Cp2.", "Cp3.", "Cp4.", "Fz.."]


def _write(path, run, seed):
    rng = np.random.RandomState(seed)
    sf, n = 160.0, 160 * 70
    data = rng.randn(len(CH), n) * 1e-5
    onsets = np.arange(2, 62, 4.0)
    labels = ["T0" if i % 2 == 0 else ("T1" if i % 4 == 1 else "T2") for i in range(len(onsets))]
    for on, l in zip(onsets, labels):  # class info: amplitude of C3 (T1) / C4 (T2), stronger in hands-vs-feet runs
        i0 = int(on * sf)
        if l != "T0":
            data[CH.index("C3.." if l == "T1" else "C4.."), i0:i0 + 640] *= 4 if run in ("R04", "R08", "R12") else 8
    raw = mne.io.RawArray(data, mne.create_info(CH, sf, "eeg"), verbose=False)
    raw.set_annotations(mne.Annotations(onsets, 4.0, labels))
    mne.export.export_raw(path, raw, fmt="edf", overwrite=True, verbose=False)


def _edfs(root, n_sub):
    files = {}
    for s in range(1, n_sub + 1):
        d = root / f"S{s:03d}"; d.mkdir(parents=True)
        for r in B.RUNS:
            p = d / f"S{s:03d}{r}.edf"; _write(p, r, s * 100 + int(r[1:]))
            files.setdefault(f"{s:03d}", []).append(str(p))
    return files


def test_labels_windows_and_zscore(tmp_path):
    files = _edfs(tmp_path, 1)
    X, y, trial = B.load_subject(files["001"])
    assert X.shape[1:] == (12, 640)                      # 6 pairs x 2 channels, 4 s at 160 Hz
    assert set(y) == {0, 1, 2, 3, 4}
    n_rest = (y == 4).sum()
    assert n_rest > (y == 0).sum()                       # T0 = baseline = majority class (paper: SMOTE on the rest)
    assert len(set(trial)) == len(y)
    assert np.allclose(X.mean(axis=(0, 2)), 0, atol=1e-6) and np.allclose(X.std(axis=(0, 2)), 1, atol=1e-3)


def test_pair_instances_and_lr_task():
    X = np.arange(2 * 12 * 3, dtype=float).reshape(2, 12, 3)
    Xi, yi, ti, pi = B.to_pairs(X, np.array([0, 4]), np.array([7, 8]))
    assert Xi.shape == (12, 2, 3) and list(yi) == [0] * 6 + [4] * 6 and list(ti) == [7] * 6 + [8] * 6
    assert np.array_equal(Xi[1], X[0, 2:4]) and list(pi[:6]) == list(range(6))
    keep = B.task_mask(np.array([0, 1, 2, 3, 4]), "lr")
    assert list(keep) == [True, True, False, False, False]


def test_smote_balances_and_interpolates_within_class():
    rng = np.random.RandomState(0)
    X = np.concatenate([rng.randn(40, 2, 5), rng.randn(10, 2, 5) + 10])
    y = np.array([0] * 40 + [1] * 10)
    Xs, ys = B.smote(X, y, k=5, seed=0)
    assert (np.bincount(ys) == 40).all() and np.array_equal(Xs[:50], X)
    new = Xs[50:]
    assert (new.min(0) >= X[y == 1].min(0) - 1e-9).all() and (new.max(0) <= X[y == 1].max(0) + 1e-9).all()


def test_split_schemes_leak_exactly_what_they_should():
    subj = np.repeat(np.arange(6), 60)
    trial = np.repeat(np.arange(60), 6)                  # 6 pair-instances per trial
    y = np.tile(np.repeat(np.arange(5), 6), 12)
    for scheme in ("random_instance", "trial_grouped", "subject"):
        leak_trial = leak_subj = False
        for tr, va, te in B.splits(scheme, y, subj, trial, n_folds=3, seed=0):
            assert not (set(tr) & set(te)) and not (set(va) & set(te)) and not (set(tr) & set(va))
            assert len(tr) + len(va) + len(te) == len(y)
            leak_trial |= bool(set(trial[tr]) & set(trial[te]))
            leak_subj |= bool(set(subj[tr]) & set(subj[te]))
            if scheme != "random_instance":
                assert not set(trial[va]) & set(trial[te])
        assert leak_trial == (scheme == "random_instance")
        assert leak_subj == (scheme != "subject")


def test_ovr_accuracy_matches_paper_definition():
    y = np.repeat(np.arange(5), 20)
    p = y.copy(); p[::5] = (p[::5] + 1) % 5             # 80% multiclass accuracy
    assert abs((p == y).mean() - 0.8) < 1e-9
    assert abs(B.ovr_accuracy(y, p, 5) - (1 - 0.4 * 0.2)) < 1e-9   # mean (TP+TN)/N over classes


def test_model_shape():
    import torch
    m = B.CNNGRU(n_classes=5).eval()
    assert m(torch.randn(3, 2, 640)).shape == (3, 5)


def test_script_end_to_end_and_resume(tmp_path):
    _edfs(tmp_path / "physionet", 4)
    cfg = yaml.safe_load(open("configs/replication_bouchane.yaml"))
    cfg["data"].update(data_dir=str(tmp_path / "physionet"), exclude=[])
    cfg["train"].update(epochs=1)
    cfg["experiments"] = [dict(name="r_small", subjects=3, task="5class", scheme="random_instance", n_folds=2),
                          dict(name="s_small", subjects=4, task="lr", scheme="subject", n_folds=2)]
    cf = tmp_path / "c.yaml"; yaml.safe_dump(cfg, open(cf, "w"))
    out = tmp_path / "out"
    for _ in range(2):
        r = subprocess.run([sys.executable, "scripts/run_replication_bouchane.py", "--config", str(cf), "--out", str(out)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
    assert "done earlier" in r.stdout
    txt = (out / "summary.txt").read_text()
    assert "r_small" in txt and "s_small" in txt and "ovr" in txt
