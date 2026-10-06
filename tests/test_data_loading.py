"""Exclusion must come ONLY from the configurable preset + sfreq check, and every drop must be logged."""
import sys
import types

import mne
import numpy as np

from src.eval.data import load_raw


def _edf(path, sfreq, seed=0):
    rng = np.random.RandomState(seed)
    raw = mne.io.RawArray(rng.randn(2, int(sfreq * 20)) * 1e-5, mne.create_info(["C3..", "C4.."], sfreq, "eeg"), verbose=False)
    raw.set_annotations(mne.Annotations([2.0, 8.0], 4.0, ["T1", "T2"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(path, raw, fmt="edf", overwrite=True, verbose=False)


def _fake_kaggle(tmp_path, monkeypatch):
    for sid, sf in (("001", 160.0), ("038", 160.0), ("089", 160.0), ("088", 128.0)):
        for run in ("R04", "R08", "R12"):
            _edf(tmp_path / "files" / f"S{sid}" / f"S{sid}{run}.edf", sf)
    fake = types.ModuleType("kagglehub")
    fake.dataset_download = lambda *_a, **_k: str(tmp_path)
    monkeypatch.setitem(sys.modules, "kagglehub", fake)


def test_kagglehub_path_applies_only_configured_exclusion(tmp_path, monkeypatch, capsys):
    _fake_kaggle(tmp_path, monkeypatch)
    raw = load_raw({"data_dir": None, "exclude": "koellod2023", "sfreq": 160.0})
    # 038 is NOT on the koellod2023 list (the old hard-coded list silently removed it)
    assert set(raw) == {"001", "038"}
    out = capsys.readouterr().out
    assert "089" in out and "excluded by id" in out     # logged
    assert "088" in out and "sfreq" in out              # dropped by real sfreq, also logged


def test_exclude_none_keeps_everything_with_valid_sfreq(tmp_path, monkeypatch):
    _fake_kaggle(tmp_path, monkeypatch)
    assert set(load_raw({"data_dir": None, "exclude": "none", "sfreq": 160.0})) == {"001", "038", "089"}
