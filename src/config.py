"""
Configuration loading and merging.

Loads a base YAML config, optionally deep-merges with an override file,
and exposes a plain dict (or OmegaConf DictConfig if available).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load configuration from YAML.

    Parameters
    ----------
    path : str or Path, optional
        Path to a YAML config file.  If *None*, loads ``configs/default.yaml``.
    overrides : dict, optional
        Programmatic overrides applied **on top** of the loaded file.

    Returns
    -------
    dict
        Fully merged configuration dictionary.
    """
    base_path = Path(path) if path else _DEFAULT_CONFIG

    with open(base_path, "r") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    # Auto-compute derived values
    eegnet = cfg.get("eegnet", {})
    if "f2" not in eegnet:
        eegnet["f2"] = eegnet.get("f1", 8) * eegnet.get("d", 2)
        cfg["eegnet"] = eegnet

    training = cfg.get("training", {})
    if training.get("scheduler_T_max") is None:
        training["scheduler_T_max"] = training.get("epochs", 50)
        cfg["training"] = training

    return cfg
