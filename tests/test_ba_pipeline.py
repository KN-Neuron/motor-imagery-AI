"""Synthetic-data tests for the BrainAccess module: mismatch detection, no padding, no random splits."""
import numpy as np
import pandas as pd
import pytest
import mne

from src.ba.pipeline import (TASK_KIND, block_folds, evaluate_methods, load_checkpoint, load_session,
                             read_events, save_checkpoint)
from src.data.normalization import PreprocMeta, PreprocessingMismatch
from src.models.eegnet import EEGNet

CH = ["C3", "Cz", "C4", "Pz"]
SF = 160.0


def _make_session(tmp_path, sfreq=SF, scale=1e-5, ch=CH, dur=80, name="s1", task="CLENCH"):
    d = tmp_path / name; d.mkdir()
    rng = np.random.RandomState(0)
    n = int(dur * sfreq)
    t = np.arange(n) / sfreq
    data = (np.sin(2 * np.pi * 12 * t) + .3 * rng.randn(len(ch), n)) * scale
    raw = mne.io.RawArray(data, mne.create_info(ch, sfreq, "eeg"), verbose=False)
    mne.export.export_raw(d / "session.edf", raw, fmt="edf", overwrite=True, verbose=False)
    onsets = np.arange(5, dur - 5, 4.0)
    pd.DataFrame({"onset": onsets, "duration": 3.0,
                  "trial_type": [f"{'LEFT' if i % 2 == 0 else 'RIGHT'}_HAND_{task}" for i in range(len(onsets))]}
                 ).to_csv(d / "events.tsv", sep="\t", index=False)
    return d / "session.edf", d / "events.tsv"


def _meta(ch=CH, norm="zscore_subject_channel", band=(8.0, 30.0)):
    ref = np.random.RandomState(1).randn(20, len(ch), 320).astype(np.float32)
    from scipy.signal import butter, sosfiltfilt
    ref = sosfiltfilt(butter(4, band, "band", fs=SF, output="sos"), ref, axis=-1)
    ref = (ref / ref.std()).astype(np.float32)
    return PreprocMeta.from_training_data(ref, bandpass=band, tmin=0.5, tmax=2.5, sfreq=SF, channels=ch, normalization=norm)


def test_clench_is_labelled_execution_not_imagery():
    assert TASK_KIND["LEFT_HAND_CLENCH"][1] == "ME" and TASK_KIND["RIGHT_HAND_CLENCH"][1] == "ME"


def test_read_events_filters_by_task_kind(tmp_path):
    _, tsv = _make_session(tmp_path)
    assert len(read_events(tsv, "ME")[0]) > 0
    assert len(read_events(tsv, "MI")[0]) == 0  # ME recordings are never silently used as MI


def test_session_loads_with_exact_window_and_no_padding(tmp_path):
    edf, tsv = _make_session(tmp_path)
    X, y, idx = load_session(edf, tsv, _meta())
    assert X.shape[1:] == (4, 320) and len(X) == len(y) == len(idx)


def test_missing_channel_is_an_error_not_zeros(tmp_path):
    edf, tsv = _make_session(tmp_path, ch=["C3", "Cz", "C4"])
    with pytest.raises(PreprocessingMismatch, match="channels"):
        load_session(edf, tsv, _meta())


def test_scale_mismatch_detected_when_normalization_is_none(tmp_path):
    edf, tsv = _make_session(tmp_path, scale=1.0)  # data in "volts" 1e5 times too large vs training scale
    ref = np.random.RandomState(1).randn(20, 4, 320).astype(np.float32) * 1e-5
    meta = PreprocMeta.from_training_data(ref, bandpass=(8.0, 30.0), tmin=0.5, tmax=2.5, sfreq=SF,
                                          channels=CH, normalization="none")
    with pytest.raises(PreprocessingMismatch, match="scale"):
        load_session(edf, tsv, meta)


def test_band_mismatch_detected(tmp_path):
    edf, tsv = _make_session(tmp_path)
    meta = _meta(band=(8.0, 30.0))
    meta.ref_out_of_band = 0.0
    X = np.random.RandomState(0).randn(10, 4, 320).astype(np.float32)  # broadband data
    from src.data.normalization import check_compatibility
    with pytest.raises(PreprocessingMismatch, match="band"):
        check_compatibility(meta, X / X.std(), CH, SF)


def test_short_recording_drops_trials_instead_of_padding(tmp_path):
    edf, tsv = _make_session(tmp_path, dur=20)
    # last onset leaves < 2 s of data after tmin: must be dropped
    pd.DataFrame({"onset": [5.0, 19.5], "duration": 3.0, "trial_type": ["LEFT_HAND_CLENCH", "RIGHT_HAND_CLENCH"]}
                 ).to_csv(tsv, sep="\t", index=False)
    X, y, idx = load_session(edf, tsv, _meta())
    assert len(X) == 1


def test_block_folds_are_contiguous_disjoint_no_random_epochs():
    groups = np.repeat(np.arange(10), 6)
    for tr, va, te in block_folds(groups, 5):
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
        assert set(groups[te]).isdisjoint(set(groups[tr]) | set(groups[va]))  # whole blocks only
        assert len(np.unique(groups[te])) == 2


def test_checkpoint_requires_metadata_and_roundtrips(tmp_path):
    net = EEGNet(chans=4, classes=2, time_points=320)
    save_checkpoint(net, tmp_path / "m.pth", _meta(), classes=2, f1=8, d=2)
    net2, meta = load_checkpoint(tmp_path / "m.pth")
    assert meta.channels == CH and net2.fc.in_features == net.fc.in_features
    import torch
    torch.save(net.state_dict(), tmp_path / "legacy.pth")
    with pytest.raises(PreprocessingMismatch, match="metadata"):
        load_checkpoint(tmp_path / "legacy.pth")


def test_evaluate_methods_runs_on_block_folds():
    rng = np.random.RandomState(0)
    X = rng.randn(60, 4, 320).astype(np.float32); y = np.tile([0, 1], 30)
    groups = np.repeat(np.arange(10), 6)
    net = EEGNet(chans=4, classes=2, time_points=320).eval()
    res = evaluate_methods(X, y, groups, net, n_folds=3)
    assert set(res) >= {"frozen", "bn_adapt", "finetune_last", "scratch", "csp_lda"}
    assert all(sum(n for n, _ in v) == 60 for v in res.values())  # each trial tested exactly once


def test_bids_export_writes_valid_structure(tmp_path):
    from src.ba.bids import export_bids
    edf, tsv = _make_session(tmp_path)
    p = export_bids(edf, tsv, tmp_path / "bids", subject="p001", session="01", kind="me")
    assert p.fpath.exists()
    assert (tmp_path / "bids" / "dataset_description.json").exists()
    assert any((tmp_path / "bids").rglob("*_events.tsv"))
    assert "handclenchexecution" in p.fpath.name
