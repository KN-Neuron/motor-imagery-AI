#!/usr/bin/env bash
# Overnight run: hyperparameters tuned inside N-LNSO on the MI benchmark (configs/mi_tuned.yaml).
# One command (from the repo root):  nohup bash scripts/run_night.sh > night.log 2>&1 &
# Interrupted? Run the same command again: finished models and inner scores are reused (results/mi_tuned/parts).
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
unset CUDA_VISIBLE_DEVICES
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "##### mi_tuned  $(date '+%F %T')"
python scripts/run_tuned.py --config configs/mi_tuned.yaml --out results/mi_tuned ${DATA_DIR:+--data-dir "$DATA_DIR"} \
  || echo "##### mi_tuned FAILED"
{
  python scripts/summarize_results.py results/mi_reg_mu_beta_nocue results/mi_tuned
  echo
  echo "selected hyperparameters per outer fold (inner acc):"
  python -c "import pandas as pd; s = pd.read_csv('results/mi_tuned/selection.csv'); print(s[s.selected][['pipeline', 'fold', 'params', 'score']].to_string(index=False))"
} | tee results/night_summary.txt
echo "##### ALL DONE $(date '+%F %T')"
