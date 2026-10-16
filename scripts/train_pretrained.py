"""Train an EEGNet on PhysioNet (optionally on a common-channel subset) and save it WITH preprocessing metadata.
Validation subjects (from the training pool) pick the checkpoint. No test subjects are involved.

    python scripts/train_pretrained.py --config configs/benchmark.yaml --norm euclidean_alignment --montage midi16 --out models/eegnet_midi16.pth
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ba.pipeline import save_checkpoint  # noqa: E402
from src.data.normalization import PreprocMeta  # noqa: E402
from src.eval.data import build_epochs, load_raw  # noqa: E402
from src.eval.montages import MONTAGES, norm_name  # noqa: E402
from src.eval.nlnso import nlnso_splits  # noqa: E402
from src.eval.pipelines import make_registry  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--norm", required=True)
    ap.add_argument("--montage", default=None, choices=[None, *MONTAGES])
    ap.add_argument("--max-norm", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); pp = cfg["preprocessing"]
    raw = load_raw(cfg["data"])
    channels = pp["channels"]
    if a.montage:  # normalization is computed on the SAME channels as the target device
        want = {norm_name(c) for c in MONTAGES[a.montage]}
        first = next(iter(raw.values()))
        channels = [c for c in first.ch_names if norm_name(c) in want]
    X, y, s, ch, _ = build_epochs(raw, pp["band"], pp["tmin"], pp["tmax"], a.norm, channels, cfg["data"].get("cache_dir"))
    subj = np.unique(s)
    rng = np.random.RandomState(cfg["seed"]); val = rng.choice(subj, max(1, int(.15 * len(subj))), replace=False)
    tr = ~np.isin(s, val)
    pipe = make_registry(epochs=cfg["eval"]["epochs"])["eegnet_maxnorm" if a.max_norm else "eegnet"](cfg["seed"])
    pipe.fit(X[tr], y[tr], X[~tr], y[~tr])
    meta = PreprocMeta.from_training_data(X[tr], bandpass=tuple(pp["band"]), tmin=pp["tmin"], tmax=pp["tmax"],
                                          sfreq=cfg["data"]["sfreq"], channels=list(ch), normalization=a.norm)
    save_checkpoint(pipe.model, a.out, meta, classes=2, f1=8, d=2, val_acc=pipe.val_acc_, val_subjects=[int(v) for v in val])


if __name__ == "__main__":
    main()
