import subprocess
import sys

import pandas as pd


def test_inspect_history_reports_learning_per_group(tmp_path):
    root = tmp_path / "final"
    rows = []
    for g, va in (("learns", [.5, .6, .7]), ("flat", [.5, .5, .49])):
        for f in range(2):
            for e, a in enumerate(va):
                rows.append(dict(pipeline=g, fold=f, seed=0, epoch=e, train_loss=1 - e / 10, train_acc=.5 + e / 10,
                                 val_loss=.7 - (e / 10 if g == "learns" else -e / 10), val_acc=a, lr=1e-3,
                                 best_epoch=2 if g == "learns" else 0))
    (root / "run_a").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(root / "run_a" / "history.csv", index=False)
    (root / "replication_bouchane").mkdir()
    pd.DataFrame(rows).rename(columns={"pipeline": "experiment"}).to_csv(root / "replication_bouchane" / "history.csv", index=False)
    r = subprocess.run([sys.executable, "scripts/inspect_history.py", str(root)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert out.count("learns") == 2 and out.count("flat") == 2
    line = next(l for l in out.splitlines() if l.startswith("run_a") and "flat" in l)
    assert "50.0" in line and "0.0" not in line.split()[0]
    assert (root / "history_summary.csv").exists()
