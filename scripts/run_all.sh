#!/usr/bin/env bash
# Everything from scratch into one results tree, with training curves, figures and a report.
# One command (from the repo root):  nohup bash scripts/run_all.sh > run_all.log 2>&1 &
#
# Output: $ROOT/<run>/ (per_subject, per_trial, history, run_meta), $ROOT/logs/<step>.log,
#         $ROOT/status.tsv (start, end, exit code per step), $ROOT/env/ (git, GPU, pip freeze),
#         $ROOT/summary_all.txt, $ROOT/figures/*.png, $ROOT/REPORT.md (refreshed after every phase).
# Options (env): ROOT=results/final  PHASES="mi mm legacy main replication tuned"  DATA_DIR=/path  DRY_RUN=1
# Interrupted? Run the same command again: finished models are reused from $ROOT/*/parts.
cd "$(dirname "$0")/.."
ROOT="${ROOT:-results/final}"
PHASES="${PHASES:-mi mm legacy main replication tuned}"
DD=${DATA_DIR:+--data-dir "$DATA_DIR"}

run() {  # run <step-name> <command...>: logged, timed, never stops the whole pipeline
  local name=$1; shift
  if [ -n "$DRY_RUN" ]; then echo "$*"; return; fi
  echo "##### $name  $(date '+%F %T')"
  local t0; t0=$(date '+%F %T')
  "$@" 2>&1 | tee "$ROOT/logs/$name.log"
  local rc=${PIPESTATUS[0]}
  printf '%s\t%s\t%s\t%s\n' "$name" "$t0" "$(date '+%F %T')" "$rc" >> "$ROOT/status.tsv"
  [ "$rc" -eq 0 ] || echo "##### $name FAILED (exit $rc, continuing)"
}

bench() {  # bench <config-name>
  run "$1" python scripts/run_benchmark.py --config "configs/$1.yaml" --out "$ROOT/$1" $DD
}

report() {
  run summary python scripts/summarize_results.py $(for d in "$ROOT"/*/; do [ -f "$d/per_subject.csv" ] && echo "${d%/}"; done)
  [ -n "$DRY_RUN" ] || cp "$ROOT/logs/summary.log" "$ROOT/summary_all.txt"
  run figures python scripts/make_figures.py --root "$ROOT"
}

names() { for f in "$@"; do [ -f "$f" ] && basename "$f" .yaml; done; }

if [ -z "$DRY_RUN" ]; then
  mkdir -p "$ROOT/logs" "$ROOT/env"
  [ -f .venv/bin/activate ] && source .venv/bin/activate
  unset CUDA_VISIBLE_DEVICES
  { git rev-parse HEAD; git status --porcelain; } > "$ROOT/env/git.txt"
  nvidia-smi > "$ROOT/env/gpu.txt" 2>&1
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" | tee "$ROOT/env/torch.txt"
fi
run env python -m pip freeze --all
[ -n "$DRY_RUN" ] || cp "$ROOT/logs/env.log" "$ROOT/env/pip_freeze.txt"

for phase in $PHASES; do
  case $phase in
    mi)     for c in $(names configs/mi_reg_*.yaml configs/mi_ctrl_*.yaml); do bench "$c"; done ;;
    mm)     for c in $(names configs/mm_*.yaml); do bench "$c"; done ;;
    legacy) for c in $(names configs/legacy_*.yaml); do bench "$c"; done ;;
    main)
      run main python scripts/run_benchmark.py --config configs/benchmark.yaml --out "$ROOT/main" $DD
      run sim python scripts/run_simulations.py --config configs/benchmark.yaml --out "$ROOT/sim" \
        --trials "$ROOT/main/per_trial.csv" $DD
      run main_report python scripts/make_report.py --main "$ROOT/main" --sim "$ROOT/sim" --out "$ROOT/results_main.md" ;;
    tuned)  run mi_tuned python scripts/run_tuned.py --config configs/mi_tuned.yaml --out "$ROOT/mi_tuned" $DD ;;
    replication)
      run replication python scripts/run_replication_bouchane.py --config configs/replication_bouchane.yaml \
        --out "$ROOT/replication_bouchane" $DD ;;
    *) echo "unknown phase: $phase" >&2 ;;
  esac
  report
done
echo "##### ALL DONE $(date '+%F %T')"
