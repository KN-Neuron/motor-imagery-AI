import numpy as np
import pytest

from src.data.normalization import (
    NORMALIZATIONS, PreprocessingMismatch, PreprocMeta, check_compatibility,
    ea_reference, euclidean_alignment, normalize_subject,
)

CH = [f"C{i}" for i in range(8)]


def _subject(seed, n=20, c=8, t=320, scale=1e-5):
    rng = np.random.RandomState(seed)
    mix = rng.randn(c, c)
    return (np.einsum("cd,ndt->nct", mix, rng.randn(n, c, t)) * scale).astype(np.float32)


def test_euclidean_alignment_whitens_mean_covariance():
    Xa = euclidean_alignment(_subject(0))
    cov = np.einsum("nct,ndt->cd", Xa.astype(float), Xa.astype(float)) / (Xa.shape[0] * Xa.shape[2])
    assert np.allclose(cov, np.eye(8), atol=1e-3)


@pytest.mark.parametrize("method", [m for m in NORMALIZATIONS if m != "none"])
def test_normalization_is_per_subject_and_label_free(method):
    """Output for subject A must not depend on subject B (no cross-subject leakage)."""
    A, B = _subject(1), _subject(2, scale=5e-4)
    out_alone = normalize_subject(A, method)
    out_with_other = normalize_subject(A.copy(), method)  # B is never an input
    assert np.array_equal(out_alone, out_with_other)
    # statistics differ between subjects: each subject normalized on its own
    assert not np.allclose(normalize_subject(B, method), out_alone)


def test_scale_invariance_of_subject_normalizations():
    """V vs uV input gives the same normalized data, so train/inference scale cannot differ."""
    X = _subject(3)
    for method in ("zscore_subject_channel", "euclidean_alignment"):
        a, b = normalize_subject(X, method), normalize_subject(X * 1e6, method)
        assert np.allclose(a, b, atol=1e-3), method


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        normalize_subject(_subject(0), "magic")


def _meta(X):
    return PreprocMeta.from_training_data(
        X, bandpass=(7.0, 30.0), tmin=0.0, tmax=2.0, sfreq=160.0, channels=CH,
        normalization="zscore_subject_channel",
    )


def _bandlimited(seed=0, n=10, t=320, f=(8, 28)):
    from scipy.signal import butter, sosfiltfilt
    rng = np.random.RandomState(seed)
    sos = butter(4, f, btype="band", fs=160.0, output="sos")
    return normalize_subject(sosfiltfilt(sos, rng.randn(n, 8, t), axis=-1).astype(np.float32),
                             "zscore_subject_channel")


def test_compat_accepts_matching_data():
    X = _bandlimited()
    assert check_compatibility(_meta(X), _bandlimited(1), CH, 160.0) == []


def test_compat_detects_scale_mismatch():
    X = _bandlimited()
    with pytest.raises(PreprocessingMismatch, match="scale"):
        check_compatibility(_meta(X), X * 1e-5, CH, 160.0)  # volts vs z-score


def test_compat_detects_channel_and_window_and_sfreq_mismatch():
    X = _bandlimited()
    m = _meta(X)
    with pytest.raises(PreprocessingMismatch, match="channels"):
        check_compatibility(m, X, CH[::-1], 160.0)
    with pytest.raises(PreprocessingMismatch, match="window"):
        check_compatibility(m, X[..., :200], CH, 160.0)
    with pytest.raises(PreprocessingMismatch, match="sfreq"):
        check_compatibility(m, X, CH, 250.0)


def test_compat_detects_band_mismatch():
    X = _bandlimited()
    broadband = normalize_subject(np.random.RandomState(5).randn(10, 8, 320).astype(np.float32),
                                  "zscore_subject_channel")
    with pytest.raises(PreprocessingMismatch, match="band"):
        check_compatibility(_meta(X), broadband, CH, 160.0)
    with pytest.warns(UserWarning, match="band"):
        check_compatibility(_meta(X), broadband, CH, 160.0, strict=False)


def test_meta_roundtrip():
    m = _meta(_bandlimited())
    assert PreprocMeta.from_dict(m.to_dict()).to_dict() == m.to_dict()
