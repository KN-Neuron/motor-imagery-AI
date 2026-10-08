#!/usr/bin/env bash
# Motor EXECUTION (R03/R07/R11) mirror of the imagery checks: old setting, eye proxy, MI benchmark, low-frequency control.
# One command (from the repo root):  nohup bash scripts/run_movement_check.sh > movement_check.log 2>&1 &
cd "$(dirname "$0")/.."
RUNS="mm_legacy_wideband mm_eye_heog mm_reg_mu_beta_nocue mm_reg_wb_nocue mm_ctrl_lowfreq_raw mm_ctrl_lowfreq_reg" \
SUMMARY=results/movement_summary.txt \
COMPARE="results/legacy_wideband results/legacy_eye_heog results/mi_reg_mu_beta_nocue results/mi_reg_wb_nocue results/mi_ctrl_lowfreq_raw results/mi_ctrl_lowfreq_reg" \
exec bash scripts/run_legacy_check.sh
