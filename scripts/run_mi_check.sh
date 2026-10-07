#!/usr/bin/env bash
# MI benchmark on motor channels after EOG regression + residual-eye control.
# One command (from the repo root):  nohup bash scripts/run_mi_check.sh > mi_check.log 2>&1 &
# Result table: results/mi_summary.txt (with reference rows from earlier runs).
cd "$(dirname "$0")/.."
RUNS="mi_reg_mu_beta mi_reg_mu_beta_nocue mi_reg_wb_nocue mi_ctrl_lowfreq_raw mi_ctrl_lowfreq_reg" \
SUMMARY=results/mi_summary.txt \
COMPARE="results/main results/legacy_wideband results/legacy_motor21 results/legacy_eye_heog results/legacy_eye_motor_regressed" \
exec bash scripts/run_legacy_check.sh
