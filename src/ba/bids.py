"""Export own BrainAccess recordings to BIDS-EEG (Pernet et al., Sci Data 2019, 6:103)."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
from mne_bids import BIDSPath, write_raw_bids

TASKS = {"me": "handclenchexecution", "mi": "handimagery", "rest": "restingeyesopen"}  # BIDS labels: alphanumeric only


def export_bids(edf: str | Path, events_tsv: str | Path | None, root: str | Path, subject: str,
                session: str, kind: str, run: int = 1, line_freq: float = 50.0) -> BIDSPath:
    """
    ``subject`` must be a pseudonym (e.g. 'p001'), never a name; keep the key
    file mapping pseudonyms to persons separately and off the dataset (RODO).
    ``kind``: 'me' (executed movement), 'mi' (imagery), 'rest' (2 min eyes open).
    """
    import pandas as pd
    raw = mne.io.read_raw_edf(edf, preload=False, verbose=False)
    raw.info["line_freq"] = line_freq
    event_id = None
    if events_tsv is not None:
        df = pd.read_csv(events_tsv, sep="\t")
        names = sorted(df.trial_type.unique())
        event_id = {n: i + 1 for i, n in enumerate(names)}
        raw.set_annotations(mne.Annotations(df.onset.values, df.get("duration", pd.Series(np.zeros(len(df)))).values,
                                            df.trial_type.values))
    path = BIDSPath(subject=subject, session=session, task=TASKS[kind], run=run, datatype="eeg", root=root)
    write_raw_bids(raw, path, event_id=event_id, allow_preload=True, format="EDF", overwrite=True, verbose=False)
    return path
