#!/usr/bin/env bash
# Explains the gap between the old 84% holdout and the main N-LNSO benchmark.
# One command (from the repo root):  nohup bash scripts/run_legacy_check.sh > legacy_check.log 2>&1 &
# Finished pipelines are reused (results/<run>/parts), so re-running after a crash continues.
# Optional: DATA_DIR=/path/to/edf bash scripts/run_legacy_check.sh
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
unset CUDA_VISIBLE_DEVICES
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
RUNS="${RUNS:-legacy_replica legacy_window_only legacy_no_cue legacy_motor21 legacy_wideband}"
for r in $RUNS; do
  echo "##### $r  $(date '+%F %T')"
  python scripts/run_benchmark.py --config "configs/$r.yaml" --out "results/$r" ${DATA_DIR:+--data-dir "$DATA_DIR"} \
    || echo "##### $r FAILED (continuing)"
done
python scripts/summarize_results.py results/main $(for r in $RUNS; do echo "results/$r"; done) | tee results/legacy_summary.txt
echo "##### ALL DONE $(date '+%F %T')"
