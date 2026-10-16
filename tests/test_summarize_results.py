import subprocess
import sys

import pandas as pd


def test_summarize_results_prints_one_row_per_run_and_pipeline(tmp_path):
    for run, acc in (("a", 0.6), ("b", 0.8)):
        d = tmp_path / run
        d.mkdir()
        pd.DataFrame({"pipeline": ["eegnet|none"] * 4, "subject": range(4), "n_trials": 45,
                      "n_correct": acc * 45, "acc": acc}).to_csv(d / "per_subject.csv", index=False)
    out = subprocess.run([sys.executable, "scripts/summarize_results.py", str(tmp_path / "a"), str(tmp_path / "b")],
                         capture_output=True, text=True, check=True).stdout
    lines = [l for l in out.splitlines() if "eegnet|none" in l]
    assert len(lines) == 2 and "60.0" in lines[0] and "80.0" in lines[1]
