import numpy as np

from src.eval.data import regress_out
from src.eval.pipelines import make_registry


def test_regress_out_removes_reference_leak_per_subject_and_drops_refs():
    rng = np.random.RandomState(0)
    n, T = 40, 200
    eog = rng.randn(n, 1, T)
    brain = rng.randn(n, 1, T)
    s = np.repeat([1, 2], n // 2)
    gain = np.where(s == 1, 0.5, 3.0)[:, None, None]  # different leak per subject
    X = np.concatenate([brain + gain * eog, eog], axis=1)
    Xc, ch = regress_out(X, s, ["C3..", "F7.."], ["F7.."])
    assert ch == ["C3.."] and Xc.shape == (n, 1, T)
    for sid in (1, 2):
        m = s == sid
        assert abs(np.corrcoef(Xc[m].ravel(), eog[m].ravel())[0, 1]) < 0.02
        assert np.corrcoef(Xc[m].ravel(), brain[m].ravel())[0, 1] > 0.95


def test_heog_lda_learns_lateral_offset():
    rng = np.random.RandomState(0)
    n, T = 200, 160
    y = np.tile([0, 1], n // 2)
    X = rng.randn(n, 2, T)
    X[:, 0, 40:] += np.where(y == 1, 1.0, -1.0)[:, None]  # F7 - F8 shifts with gaze side
    p = make_registry()["heog_lda"](0).fit(X[:150], y[:150])
    assert (p.predict_proba(X[150:]).argmax(1) == y[150:]).mean() > 0.9


def test_build_epochs_channel_names_follow_data_order():
    import mne
    from src.eval.data import build_epochs
    ch = ["F7..", "C3..", "F8.."]
    data = np.zeros((3, 160 * 40))
    data[0] = 1.0  # F7 marker
    raw = mne.io.RawArray(data, mne.create_info(ch, 160.0, "eeg"), verbose=False)
    raw.set_annotations(mne.Annotations([2, 10, 18, 26], 4.0, ["T1", "T2", "T1", "T2"]))
    X, y, s, names, _ = build_epochs({"001": raw}, (0.0, 79.0), 0.0, 4.0, channels=["F8..", "F7.."])
    assert names == ["F8..", "F7.."]
    assert np.abs(X[:, names.index("F7..")]).mean() > np.abs(X[:, names.index("F8..")]).mean()


def test_build_epochs_regresses_before_normalizing():
    import mne
    import pytest
    from src.eval.data import build_epochs
    rng = np.random.RandomState(0)
    n = 160 * 40
    eog = np.cumsum(rng.randn(n)) * 1e-6
    c3 = rng.randn(n) * 1e-6 + 5 * eog
    raw = mne.io.RawArray(np.stack([c3, eog]), mne.create_info(["C3..", "F7.."], 160.0, "eeg"), verbose=False)
    raw.set_annotations(mne.Annotations([2, 10, 18, 26], 4.0, ["T1", "T2", "T1", "T2"]))
    X, _, _, names, _ = build_epochs({"001": raw}, (0.5, 40.0), 0.0, 4.0, "zscore_subject_channel",
                                     channels=["C3..", "F7.."], eog_regress=["F7.."])
    assert names == ["C3.."]
    assert abs(X.std() - 1.0) < 0.05  # z-scored AFTER regression
    with pytest.raises(ValueError):
        build_epochs({"001": raw}, (0.5, 40.0), 0.0, 4.0, "exp_moving_standardization",
                     channels=["C3..", "F7.."], eog_regress=["F7.."])
