import glob

import yaml

from src.eval.montages import PHYSIONET64


def test_physionet64_has_64_unique_names():
    assert len(PHYSIONET64) == 64 == len(set(PHYSIONET64))


def test_legacy_config_channels_exist_in_physionet():
    files = sorted(glob.glob("configs/legacy_*.yaml"))
    assert any("wb_frontal" in f for f in files)
    for f in files:
        pp = yaml.safe_load(open(f))["preprocessing"]
        ch = pp["channels"]
        assert ch is None or set(ch) <= set(PHYSIONET64), f
        assert set(pp.get("eog_regress") or []) <= set(ch or PHYSIONET64), f
