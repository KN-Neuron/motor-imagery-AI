"""epoch_subjects must honour `normalization` (the old code silently ignored it)."""
import mne
import numpy as np
import pytest

from src.data.preprocessing import epoch_subjects

CH = ["C3", "Cz", "C4", "Pz"]


def _raw(seed, scale):
    rng = np.random.RandomState(seed)
    sf, n = 160.0, 160 * 60
    mix = rng.randn(4, 4)
    raw = mne.io.RawArray(mix @ rng.randn(4, n) * scale, mne.create_info(CH, sf, "eeg"), verbose=False)
    onsets = np.arange(3, 55, 5.0)
    raw.set_annotations(mne.Annotations(onsets, 4.0, ["T1" if i % 2 == 0 else "T2" for i in range(len(onsets))]))
    return raw


EV = {"left_hand": 2, "right_hand": 3}


def test_normalization_modes_change_scale_and_are_per_subject():
    raws = {"001": _raw(0, 1e-5), "002": _raw(1, 3e-4)}
    base, *_ = epoch_subjects(raws, EV, tmin=0, tmax=2, normalization="none")
    assert base.std() < 1e-2                                # volts
    for norm in ("zscore_subject_channel", "euclidean_alignment", "exp_moving_standardization"):
        X, y, s, _ = epoch_subjects(raws, EV, tmin=0, tmax=2, normalization=norm)
        assert 0.3 < X.std() < 3.0, norm                    # unit scale for both subjects, despite 30x scale gap
    # a subject's normalized data is independent of the other subject
    a, _, _, _ = epoch_subjects({"001": raws["001"]}, EV, tmin=0, tmax=2, normalization="euclidean_alignment")
    b, _, s, _ = epoch_subjects(raws, EV, tmin=0, tmax=2, normalization="euclidean_alignment")
    assert np.allclose(a, b[s == 1], atol=1e-5)


def test_unknown_normalization_rejected():
    with pytest.raises(ValueError):
        epoch_subjects({"001": _raw(0, 1e-5)}, EV, normalization="bogus")
