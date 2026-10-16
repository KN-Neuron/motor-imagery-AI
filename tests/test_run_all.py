"""scripts/run_all.sh in DRY_RUN mode: every experiment config is scheduled once, outputs go under ROOT."""
import glob
import os
import subprocess


def test_run_all_dry_run_schedules_every_config(tmp_path):
    env = {**os.environ, "DRY_RUN": "1", "ROOT": str(tmp_path / "final")}
    r = subprocess.run(["bash", "scripts/run_all.sh"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    names = [os.path.basename(f)[:-5] for f in glob.glob("configs/legacy_*.yaml") + glob.glob("configs/mi_reg_*.yaml")
             + glob.glob("configs/mi_ctrl_*.yaml") + glob.glob("configs/mm_*.yaml")]
    for n in names:
        assert out.count(f"--config configs/{n}.yaml --out {tmp_path}/final/{n}") == 1, n
    for needle in ("configs/benchmark.yaml", "run_simulations.py", "make_report.py", "run_tuned.py",
                   "run_replication_bouchane.py", "make_figures.py --root", "summarize_results.py", "pip freeze"):
        assert needle in out, needle
    assert not (tmp_path / "final" / "legacy_replica").exists()   # dry run computes nothing


def test_run_all_phase_selection(tmp_path):
    env = {**os.environ, "DRY_RUN": "1", "ROOT": str(tmp_path / "f"), "PHASES": "replication"}
    out = subprocess.run(["bash", "scripts/run_all.sh"], capture_output=True, text=True, env=env).stdout
    assert "run_replication_bouchane.py" in out and "run_benchmark.py" not in out
