#!/usr/bin/env bash
# Bouchane et al. 2025 replication (configs/replication_bouchane.yaml).
# One command (from the repo root):  nohup bash scripts/run_replication.sh > replication.log 2>&1 &
# Interrupted? Run it again: finished experiments are reused.
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
unset CUDA_VISIBLE_DEVICES
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python scripts/run_replication_bouchane.py --config configs/replication_bouchane.yaml --out results/replication_bouchane \
  ${DATA_DIR:+--data-dir "$DATA_DIR"} || echo "##### replication FAILED"
echo "##### ALL DONE $(date '+%F %T')"
