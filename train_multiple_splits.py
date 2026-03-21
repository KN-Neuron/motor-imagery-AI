#!/usr/bin/env python3
"""
MindStride — Repeated Cross-Validation Script.

This wrapper script runs `train.py` multiple times with different random seeds.
It helps verify that our results are not overly dependent on a "lucky" or "unlucky" 
subject split by applying a "Repeated Random Sub-sampling" strategy on the 4-way split.
"""

import argparse
import numpy as np

from train import main as run_train

def main():
    parser = argparse.ArgumentParser(description="MindStride — Repeated Random Sub-sampling across splits")
    parser.add_argument("--config", type=str, default="configs/full_binary_all_channels.yaml", 
                        help="Path to YAML config")
    parser.add_argument("--runs", type=int, default=5, 
                        help="Number of random splits to test")
    parser.add_argument("--start-seed", type=int, default=42, 
                        help="Starting random seed")
    args = parser.parse_args()

    results = []

    print(f"============================================================")
    print(f" Starting {args.runs} runs to evaluate split stability.")
    print(f" Config:   {args.config}")
    print(f"============================================================")

    for i in range(args.runs):
        seed = args.start_seed + i
        print(f"\n\n{'#'*60}")
        print(f" RUN {i+1}/{args.runs} — SEED: {seed}")
        print(f"{'#'*60}\n")
        
        try:
            # Overriding the seed completely changes the train/val/dev/holdout subject assignment
            logger = run_train(config_path=args.config, overrides={"seed": seed})
        except Exception as e:
            print(f"Run {i+1} failed with error: {e}")
            continue
            
        stage_metrics = {s["stage"]: s["metrics"] for s in logger.data["stages"]}
        
        dev_acc = None
        holdout_acc = None
        best_model = "Unknown"
        
        # Extract metrics based on the stages that ran
        if "holdout_final" in stage_metrics:
            m = stage_metrics["holdout_final"]
            dev_acc = m.get("dev_acc")
            holdout_acc = m.get("holdout_acc")
            best_model = m.get("model_name", "Unknown")
        elif "final_retrain" in stage_metrics:
            m = stage_metrics["final_retrain"]
            dev_acc = m.get("dev_acc") or m.get("test_acc")
            holdout_acc = dev_acc
            best_model = m.get("model_name", "Unknown")
        elif "single_run" in stage_metrics:
            m = stage_metrics["single_run"]
            # Fallback to single_run values if the grid search was disabled
            dev_acc = m.get("dev_acc") or m.get("test_acc")
            best_model = "EEGNet_Single"

        res = {
            "seed": seed,
            "dev_acc": dev_acc,
            "holdout_acc": holdout_acc,
            "best_model": best_model,
        }
        results.append(res)
        
    print(f"\n\n{'='*70}")
    print(f" ALL RUNS COMPLETED ")
    print(f"{'='*70}")
    print(f"{'Seed':^6} | {'Dev Acc':^10} | {'Holdout Acc':^12} | {'Best Model'}")
    print(f"-"*70)
    
    valid_devs = []
    valid_holdouts = []
    
    for r in results:
        d_acc = f"{r['dev_acc']:.4f}" if r["dev_acc"] is not None else "N/A"
        h_acc = f"{r['holdout_acc']:.4f}" if r["holdout_acc"] is not None else "N/A"
        
        if r["dev_acc"] is not None: valid_devs.append(r["dev_acc"])
        if r["holdout_acc"] is not None: valid_holdouts.append(r["holdout_acc"])
        
        print(f"{r['seed']:^6d} | {d_acc:^10} | {h_acc:^12} | {r['best_model']}")
        
    print(f"-"*70)
    if valid_devs:
        print(f"MEAN Dev Acc:     {np.mean(valid_devs):.4f} ± {np.std(valid_devs):.4f}")
    if valid_holdouts:
        print(f"MEAN Holdout Acc: {np.mean(valid_holdouts):.4f} ± {np.std(valid_holdouts):.4f}")
    print(f"============================================================")

if __name__ == "__main__":
    main()
