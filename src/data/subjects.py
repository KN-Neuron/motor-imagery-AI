"""
Subject exclusion for PhysioNet EEGMMIDB, configurable and logged.

Presets (sources are from docs/literature_review.md; items marked
"do weryfikacji" were NOT independently verified against the primary papers):

- ``koellod2023``: {88, 89, 92, 100}. Kollod et al. 2023 (Electronics 12:2743)
  exclude them: 88/92/100 have different timing (5.125 s task, 1.375 s rest,
  88 also 128 Hz instead of 160 Hz) and 89 has wrong labels; MOABB docs also
  note "Subject 88 was recorded at 128 Hz instead of 160 Hz". (do weryfikacji)
- ``extended``: koellod2023 + {38, 104, 106} (Frontiers 2025,
  doi:10.3389/fnins.2025.1689647 excludes 38, 88, 89, 92, 100, 104; 106 after
  Shuqfa et al. 2024). (do weryfikacji)
- ``legacy``: the old project list {38, 82, 89, 104}. Subject 82 is not flagged
  in any source found; kept only to reproduce old runs.
- ``none``: nothing excluded by ID (the sfreq/trial checks still run).
"""

from __future__ import annotations

PRESETS: dict[str, set[int]] = {
    "none": set(),
    "koellod2023": {88, 89, 92, 100},
    "extended": {88, 89, 92, 100, 38, 104, 106},
    "legacy": {38, 82, 89, 104},
}


def resolve_exclusions(spec) -> set[int]:
    """spec: preset name, list of ints/strs, or None."""
    if spec is None:
        return set()
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise ValueError(f"Unknown exclusion preset '{spec}', choose from {list(PRESETS)}")
        return set(PRESETS[spec])
    return {int(s) for s in spec}


def filter_subjects(
    raw_data: dict, exclude=None, expected_sfreq: float = 160.0,
    expected_trial_s: float | None = None, log=print,
) -> tuple[dict, dict[str, str]]:
    """
    Drop subjects by ID and by actual sfreq (and optional mean annotation
    duration). Returns (kept_raw, {subject: reason}) and logs every rejection.
    """
    excl = resolve_exclusions(exclude)
    kept, rejected = {}, {}
    for sid, raw in raw_data.items():
        if int(sid) in excl:
            rejected[sid] = "excluded by id"
        elif float(raw.info["sfreq"]) != float(expected_sfreq):
            rejected[sid] = f"sfreq {raw.info['sfreq']} != {expected_sfreq}"
        elif expected_trial_s is not None and len(raw.annotations):
            dur = float(raw.annotations.duration[raw.annotations.duration > 0].mean())
            if abs(dur - expected_trial_s) > 0.2:
                rejected[sid] = f"mean trial duration {dur:.3f}s != {expected_trial_s}s"
        if sid not in rejected:
            kept[sid] = raw
    for sid, why in sorted(rejected.items()):
        log(f"[subjects] rejected {sid}: {why}")
    log(f"[subjects] kept {len(kept)}, rejected {len(rejected)}")
    return kept, rejected
