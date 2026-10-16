import mne
import numpy as np
import torch

from src.data.subjects import PRESETS, filter_subjects, resolve_exclusions
from src.models.eegnet import EEGNet


def _raw(sfreq):
    info = mne.create_info(["C3", "C4"], sfreq, "eeg")
    return mne.io.RawArray(np.zeros((2, int(sfreq * 10))), info, verbose=False)


def test_presets_cover_literature_list():
    assert {88, 89, 92, 100} <= PRESETS["koellod2023"] <= PRESETS["extended"]
    assert resolve_exclusions([1, "2"]) == {1, 2}


def test_filter_rejects_by_id_and_sfreq_and_logs():
    logs = []
    raw = {"001": _raw(160.0), "088": _raw(128.0), "089": _raw(160.0), "002": _raw(160.0)}
    kept, rej = filter_subjects(raw, "koellod2023", 160.0, log=logs.append)
    assert set(kept) == {"001", "002"}
    assert "sfreq" in rej["088"] or "id" in rej["088"]
    assert rej["089"] == "excluded by id"
    assert any("rejected 089" in l for l in logs)
    # sfreq check works even when the id is not on the list
    _, rej2 = filter_subjects({"088": _raw(128.0)}, "none", 160.0, log=logs.append)
    assert "sfreq" in rej2["088"]


def test_max_norm_is_opt_in_and_enforced():
    net = EEGNet(chans=8, classes=2, time_points=128, use_max_norm=False)
    with torch.no_grad():
        net.fc.weight.mul_(100)
    before = net.fc.weight.clone()
    net.apply_max_norm()
    assert torch.equal(before, net.fc.weight)  # off: untouched

    net.use_max_norm = True
    net.apply_max_norm()
    assert net.fc.weight.norm(dim=1).max() <= 0.25 + 1e-5
