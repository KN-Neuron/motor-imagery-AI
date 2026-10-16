"""Per-pipeline checkpointing so a long benchmark can be resumed after a crash."""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def cfg_hash(cfg: dict) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]


def part_path(out_dir, label: str, h: str, suffix: str) -> Path:
    """Path of an extra per-label file (e.g. training history) next to the resumable parts."""
    d = Path(out_dir) / "parts" / h
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', label)}_{suffix}.csv"


def run_or_load(out_dir, label: str, h: str, fn, log=print):
    """Return fn()'s (per_subject, per_trial); reuse the saved result if this
    label already finished under the same config hash."""
    d = Path(out_dir) / "parts" / h
    d.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
    p_per, p_tr = d / f"{stem}_per.csv", d / f"{stem}_trials.csv"
    if p_per.exists() and p_tr.exists():
        log(f"[{label}] done earlier, loading {p_per}")
        return pd.read_csv(p_per), pd.read_csv(p_tr)
    per, tr = fn()
    tr.to_csv(p_tr, index=False)
    per.to_csv(p_per, index=False)  # written last: marks the label as complete
    return per, tr
