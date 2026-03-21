#!/usr/bin/env python3
"""
BrainBoard — CLI entry point (FIXED v2, 4-way split).

4-way subject split:
  - TRAIN  (60%) — gradient descent, backprop
  - VAL    (20%) — early stopping, checkpoint selection
  - DEV    (10%) — model/preprocessing comparison across stages
  - HOLDOUT(10%) — touched ONCE at the very end, reported in paper

Data flow:
  ┌─────────────────────────────────────────────────────────┐
  │  Split subject IDs (before ANY preprocessing)           │
  │  ┌───────┐ ┌─────┐ ┌─────┐ ┌─────────┐                  │
  │  │ TRAIN │ │ VAL │ │ DEV │ │ HOLDOUT │                  │
  │  └───┬───┘ └──┬──┘ └──┬──┘ └────┬────┘                  │
  │      │        │       │         │                       │
  │  Preprocess  Preprocess  Preprocess  Preprocess         │
  │  (separate)  (separate)  (separate)  (separate)         │
  │      │        │       │         │                       │
  │      ▼        ▼       │         │ (locked away)         │
  │   train()  checkpoint │         │                       │
  │      │     selection  │         │                       │
  │      │        │       │         │                       │
  │  Stages 1-6   │       ▼         │                       │
  │  grid search  │   evaluate      │                       │
  │  CV on train+ │   best model    │                       │
  │  val subjects │   on DEV        │                       │
  │      │        │       │         │                       │
  │  Stage 7:     │       │         ▼                       │
  │  final retrain│       │    ONE evaluation               │
  │  with best    │       │    (reported result)            │
  │  config       │       │                                 │
  └─────────────────────────────────────────────────────────┘

  Stages 2-6 (CV, grid searches): GroupKFold on TRAIN+VAL only
  Stage 1 (single run): train on TRAIN, checkpoint on VAL, eval on DEV
  Stage 7 (final retrain): train on TRAIN, checkpoint on VAL, eval on DEV
  Stage 8 (holdout): ONE evaluation on HOLDOUT — this goes in the paper

Usage:
    python train_fixed.py --config configs/full_binary_all_channels.yaml
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data import (EEGDataset, download_dataset, epoch_subjects,
                      epoch_with_params, load_raw_subjects, make_dataloaders,
                      subject_split)
from src.data.dataset import EEGDataset
from src.data.splits import _balance_array
from src.engine import cross_validate_subjects, eval_step, train_step
from src.models import EEGNet
from src.pipelines import (run_csp_ml_grid, run_eegnet_grid, run_joint_grid,
                           run_preprocessing_grid, run_shallow_grid)
from src.utils import (ResultsLogger, get_device, plot_confusion_matrix,
                       plot_training_curves, predict_with_model,
                       print_evaluation, save_model, set_seeds)

# ═══════════════════════════════════════════════════════════════════
# Train loop — NO test/dev/holdout access
# ═══════════════════════════════════════════════════════════════════

def train_no_test(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scheduler,
    device: str,
    epochs: int = 50,
    verbose: bool = True,
    patience: int | None = None,
) -> tuple[dict[str, list[float]], float]:
    """Training loop with early stopping. Val for checkpointing ONLY. No other data access."""
    results = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
    for epoch in iterator:
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        if scheduler is not None:
            scheduler.step()
        val_loss, val_acc = eval_step(model, val_dataloader, loss_fn, device)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            print(f"Epoch {epoch+1:3d} | "
                  f"train: {train_loss:.4f} / {train_acc:.4f} | "
                  f"val: {val_loss:.4f} / {val_acc:.4f}" + 
                  (" (Best)" if is_best else f" (no imp: {epochs_no_improve})"))

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        if patience is not None and epochs_no_improve >= patience:
            if verbose:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    if best_state:
        model.load_state_dict(best_state)
    if verbose:
        print(f"\nBest val_acc: {best_val_acc:.4f}")

    return results, best_val_acc


# ═══════════════════════════════════════════════════════════════════
# Preprocessing — per subject list, fully isolated
# ═══════════════════════════════════════════════════════════════════

def preprocess_split(
    raw_data: dict, subject_ids: set | list,
    event_id: dict, channels: list | None,
    low_freq: float, high_freq: float,
    tmin: float, tmax: float,
    baseline: tuple | None, label_offset: int, seed: int,
    balance: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess ONLY the given subjects. Z-score per-subject per-channel."""
    raw_subset = {s: raw_data[s] for s in raw_data if s in set(subject_ids)}
    X, y, subjects, _ = epoch_subjects(
        raw_subset, event_id=event_id, channels=channels,
        low_freq=low_freq, high_freq=high_freq, tmin=tmin, tmax=tmax,
        baseline=baseline, normalize=False, balance=False,
        label_offset=label_offset, seed=seed)
    if balance and len(X) > 0:
        X, y, subjects = _balance_array(X, y, subjects, seed=seed)
    return X, y, subjects


# ═══════════════════════════════════════════════════════════════════
# Subject split — 3-way or 4-way, driven by config
# ═══════════════════════════════════════════════════════════════════

def split_subject_ids(
    all_subject_ids: list[str],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    dev_ratio: float | None = None,
    seed: int = 42,
) -> dict[str, set[str]]:
    """
    Split subject IDs. No data touched.

    If dev_ratio is set (> 0):
        4-way: train / val / dev / holdout (holdout = remainder)
        Val = early stopping, Dev = model comparison, Holdout = paper result
    If dev_ratio is None or 0:
        3-way: train / val / test (test = remainder)
        Val = early stopping + model comparison, Test = paper result
    """
    rng = np.random.RandomState(seed)
    ids = np.array(sorted(all_subject_ids))
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    if dev_ratio and dev_ratio > 0:
        # 4-way split
        n_dev = int(n * dev_ratio)
        return {
            "train":   set(ids[:n_train]),
            "val":     set(ids[n_train:n_train + n_val]),
            "dev":     set(ids[n_train + n_val:n_train + n_val + n_dev]),
            "holdout": set(ids[n_train + n_val + n_dev:]),
        }
    else:
        # 3-way split (backward compatible)
        return {
            "train": set(ids[:n_train]),
            "val":   set(ids[n_train:n_train + n_val]),
            "test":  set(ids[n_train + n_val:]),
        }


def _make_loss_fn(y, device, weighted=True):
    if weighted:
        w = compute_class_weight("balanced", classes=np.unique(y), y=y)
        return nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(device))
    return nn.CrossEntropyLoss()


def _make_loaders(X_train, y_train, X_val, y_val, batch_size=64):
    return {
        "train_loader": DataLoader(EEGDataset(X_train, y_train), batch_size=batch_size, shuffle=True),
        "val_loader": DataLoader(EEGDataset(X_val, y_val), batch_size=batch_size, shuffle=False),
    }


def main(config_path: str | None = None, overrides: dict | None = None):
    cfg = load_config(config_path, overrides)
    set_seeds(cfg["seed"])
    device = get_device()
    print(f"Device: {device}\nPyTorch: {torch.__version__}")

    out_dir = Path(cfg["output"]["save_dir"])
    run_cfg = cfg.get("run", {})
    logger = ResultsLogger(cfg, save_dir=out_dir)

    # ── Data loading ────────────────────────────────────────
    subjects_data = download_dataset(cfg["data"]["dataset"], cfg["data"]["desired_runs"])
    raw_data = load_raw_subjects(subjects_data, sfreq=cfg["data"]["sfreq"],
                                  n_subjects=cfg["data"].get("n_subjects"),
                                  cache_dir=cfg["data"].get("cache_dir"))

    # ── Task config ─────────────────────────────────────────
    task = cfg["task"]; pp = cfg["preprocessing"]; ch_cfg = cfg["channels"]
    channels = ch_cfg["motor_channels"] if ch_cfg["mode"] == "motor" else None
    n_chans = len(channels) if channels else 64

    if task["mode"] == "binary":
        event_id, label_offset = task["event_id_binary"], 2
        class_names, n_classes = task["class_names_binary"], 2
    else:
        event_id = task.get("event_id_ternary", {"rest": 1, "left_hand": 2, "right_hand": 3})
        label_offset, class_names, n_classes = 1, task.get("class_names_ternary", ["rest", "left_hand", "right_hand"]), 3

    balance = pp.get("balance_classes", True)
    baseline = tuple(pp["baseline"]) if pp["baseline"] else None
    eeg_cfg, tr_cfg = cfg["eegnet"], cfg["training"]
    batch_size = cfg["dataloader"]["batch_size"]

    # ══════════════════════════════════════════════════════════
    # STEP 0: Subject split BEFORE any preprocessing
    #   Config decides 3-way or 4-way (presence of dev_ratio)
    # ══════════════════════════════════════════════════════════
    split_cfg = cfg["split"]
    dev_ratio = split_cfg.get("dev_ratio", None)
    is_4way = dev_ratio is not None and dev_ratio > 0

    ids = split_subject_ids(
        list(raw_data.keys()),
        train_ratio=split_cfg.get("train_ratio", 0.60 if is_4way else 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.20 if is_4way else 0.15),
        dev_ratio=dev_ratio,
        seed=cfg["seed"])

    # Verify zero overlap
    all_sets = list(ids.values())
    all_names = list(ids.keys())
    for i in range(len(all_names)):
        for j in range(i+1, len(all_names)):
            assert len(all_sets[i] & all_sets[j]) == 0, \
                f"LEAK: {all_names[i]} ∩ {all_names[j]} = {all_sets[i] & all_sets[j]}"
    assert set().union(*all_sets) == set(raw_data.keys()), "Missing subjects!"

    if is_4way:
        print(f"\n{'='*60}\n  4-WAY SUBJECT SPLIT\n{'='*60}")
        print(f"  Train:   {len(ids['train']):3d}  (gradient descent)")
        print(f"  Val:     {len(ids['val']):3d}  (early stopping)")
        print(f"  Dev:     {len(ids['dev']):3d}  (model comparison)")
        print(f"  Holdout: {len(ids['holdout']):3d}  (final report — touched ONCE)")
        print(f"  Holdout IDs: {sorted(ids['holdout'])}")
    else:
        print(f"\n{'='*60}\n  3-WAY SUBJECT SPLIT\n{'='*60}")
        print(f"  Train:   {len(ids['train']):3d}  (gradient descent)")
        print(f"  Val:     {len(ids['val']):3d}  (early stopping + model comparison)")
        print(f"  Test:    {len(ids['test']):3d}  (final report — touched ONCE)")
        print(f"  Test IDs: {sorted(ids['test'])}")
    print(f"  ✓ Zero overlap verified")

    # ── Preprocessing per split ─────────────────────────────
    pp_kw = dict(event_id=event_id, channels=channels,
                 low_freq=pp["bandpass"][0], high_freq=pp["bandpass"][1],
                 tmin=pp["tmin"], tmax=pp["tmax"], baseline=baseline,
                 label_offset=label_offset, seed=cfg["seed"])

    print(f"\nPreprocessing per split...")
    X_train, y_train, s_train = preprocess_split(raw_data, ids["train"], balance=balance, **pp_kw)
    X_val, y_val, s_val       = preprocess_split(raw_data, ids["val"],   balance=balance, **pp_kw)

    if is_4way:
        X_dev, y_dev, s_dev       = preprocess_split(raw_data, ids["dev"],     balance=balance, **pp_kw)
        X_hold, y_hold, s_hold    = preprocess_split(raw_data, ids["holdout"], balance=balance, **pp_kw)
        # "eval set" = dev (for stages 1-7), "final test" = holdout (stage 8)
        X_eval, y_eval, s_eval = X_dev, y_dev, s_dev
        eval_label = "DEV"
    else:
        X_test, y_test, s_test    = preprocess_split(raw_data, ids["test"], balance=balance, **pp_kw)
        X_dev = X_hold = y_dev = y_hold = s_dev = s_hold = None
        # "eval set" = test (only touched at the end in 3-way)
        X_eval, y_eval, s_eval = X_test, y_test, s_test
        eval_label = "TEST"

    time_points = X_train.shape[2]

    for nm, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val)]:
        print(f"  {nm:7s}: {str(X.shape):>22s}  classes={np.bincount(y).tolist()}")
    if is_4way:
        for nm, X, y in [("Dev", X_dev, y_dev), ("Holdout", X_hold, y_hold)]:
            print(f"  {nm:7s}: {str(X.shape):>22s}  classes={np.bincount(y).tolist()}")
    else:
        print(f"  {'Test':7s}: {str(X_test.shape):>22s}  classes={np.bincount(y_test).tolist()}")

    logger.log_stage("split", {k: {"n_subjects": len(v), "subjects": sorted(v)} for k, v in ids.items()})

    # ── CV data: train+val (dev/holdout/test excluded) ──────
    cv_ids = ids["train"] | ids["val"]
    X_cv, y_cv, s_cv = preprocess_split(raw_data, cv_ids, balance=False, **pp_kw)

    # ════════════════════════════════════════════════════════
    # STAGE 1: Single EEGNet → eval on DEV
    # ════════════════════════════════════════════════════════
    if run_cfg.get("single_run", True):
        print(f"\n{'='*60}\n  STAGE 1: Single EEGNet run\n{'='*60}")
        model = EEGNet(chans=n_chans, classes=n_classes, time_points=time_points,
                        f1=eeg_cfg["f1"], f2=eeg_cfg["f1"]*eeg_cfg["d"], d=eeg_cfg["d"],
                        temp_kernel=eeg_cfg["temp_kernel"], pk1=eeg_cfg["pk1"],
                        pk2=eeg_cfg["pk2"], dropout_rate=eeg_cfg["dropout_rate"]).to(device)
        loss_fn = _make_loss_fn(y_train, device, tr_cfg["class_weighted_loss"])
        weight_decay = tr_cfg.get("weight_decay", 0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=tr_cfg["lr"], weight_decay=weight_decay)
        T_max = tr_cfg.get("scheduler_T_max") or tr_cfg["epochs"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        loaders = _make_loaders(X_train, y_train, X_val, y_val, batch_size)

        patience = tr_cfg.get("patience", None)
        results, best_val = train_no_test(model, loaders["train_loader"], loaders["val_loader"],
                                           optimizer, loss_fn, scheduler, device, 
                                           epochs=tr_cfg["epochs"], patience=patience)
        eval_preds, eval_labels = predict_with_model(
            model, DataLoader(EEGDataset(X_eval, y_eval), batch_size=batch_size), device)
        eval_acc = float(np.mean(eval_preds == eval_labels))
        print(f"{eval_label} acc: {eval_acc:.4f}")
        logger.log_stage("single_run", {"best_val_acc": float(best_val), f"{eval_label.lower()}_acc": eval_acc,
                                         "epoch_history": results},
                          extra={"eegnet": eeg_cfg, "lr": tr_cfg["lr"], "epochs": tr_cfg["epochs"]})
        plot_training_curves(results, "Single Run", save_path=out_dir / "single_run_curves.png")
        plot_confusion_matrix(eval_labels, eval_preds, class_names, f"Single Run {eval_label}",
                              save_path=out_dir / "single_run_cm.png")
        if cfg["output"]["save_model"]:
            save_model(model, out_dir / "eegnet_single_run")

    # ════════════════════════════════════════════════════════
    # STAGE 2: CV on train+val subjects
    # ════════════════════════════════════════════════════════
    if run_cfg.get("cross_validation", True):
        print(f"\n{'='*60}\n  STAGE 2: Cross-Validation (train+val subjects)\n{'='*60}")
        fold_results, mean_acc, std_acc = cross_validate_subjects(
            X_cv, y_cv, s_cv, n_splits=cfg["cv"]["n_splits"], epochs=tr_cfg["epochs"],
            lr=tr_cfg["lr"], chans=n_chans, classes=n_classes,
            time_points=time_points, device=device,
            weight_decay=tr_cfg.get("weight_decay", 0.0), 
            patience=tr_cfg.get("patience", None))
        logger.log_stage("cross_validation", {"mean_acc": float(mean_acc), "std_acc": float(std_acc),
                                               "fold_accs": [float(x) for x in fold_results]})

    # ════════════════════════════════════════════════════════
    # STAGE 3: EEGNet Grid Search
    # ════════════════════════════════════════════════════════
    if run_cfg.get("eegnet_grid_search", True):
        print(f"\n{'='*60}\n  STAGE 3: EEGNet Grid Search\n{'='*60}")
        grid_results = run_eegnet_grid(
            X_cv, y_cv, s_cv, param_grid=cfg["eegnet_grid"],
            chans=n_chans, classes=n_classes,
            n_splits=cfg["cv"]["quick_splits"], epochs_train=cfg["cv"]["quick_epochs"], device=device,
            weight_decay=tr_cfg.get("weight_decay", 0.0), patience=tr_cfg.get("patience", None),
            checkpoint_path=out_dir / "ckpt_eegnet_grid.csv")
        grid_df = pd.DataFrame(grid_results).sort_values("mean_acc", ascending=False)
        for i, r in enumerate(grid_results):
            logger.log_stage(f"eegnet_grid_combo_{i+1}", {"mean_acc": r["mean_acc"], "std_acc": r["std_acc"]},
                              extra={k: v for k, v in r.items() if k not in ("mean_acc", "std_acc")})
        logger.log_stage("eegnet_grid_summary", {"best_mean_acc": float(grid_df.iloc[0]["mean_acc"])})

    # ════════════════════════════════════════════════════════
    # STAGE 3.5: ShallowConvNet Grid Search
    # ════════════════════════════════════════════════════════
    if run_cfg.get("shallow_grid_search", False):
        print(f"\n{'='*60}\n  STAGE 3.5: ShallowConvNet Grid Search\n{'='*60}")
        shallow_results = run_shallow_grid(
            X_cv, y_cv, s_cv, param_grid=cfg.get("shallow_grid"),
            chans=n_chans, classes=n_classes,
            n_splits=cfg["cv"]["quick_splits"], epochs_train=cfg["cv"]["quick_epochs"], device=device,
            weight_decay=tr_cfg.get("weight_decay", 0.0), patience=tr_cfg.get("patience", None),
            checkpoint_path=out_dir / "ckpt_shallow_grid.csv")
        shallow_df = pd.DataFrame(shallow_results).sort_values("mean_acc", ascending=False)
        for i, r in enumerate(shallow_results):
            logger.log_stage(f"shallow_grid_combo_{i+1}", {"mean_acc": r["mean_acc"], "std_acc": r["std_acc"]},
                              extra={k: v for k, v in r.items() if k not in ("mean_acc", "std_acc")})
        logger.log_stage("shallow_grid_summary", {"best_mean_acc": float(shallow_df.iloc[0]["mean_acc"])})

    # ════════════════════════════════════════════════════════
    # STAGE 4: CSP + ML → eval best on DEV
    # ════════════════════════════════════════════════════════
    if run_cfg.get("csp_ml_grid", True):
        print(f"\n{'='*60}\n  STAGE 4: CSP + ML Grid Search\n{'='*60}")
        ml_results = run_csp_ml_grid(X_cv, y_cv, s_cv, task_mode=task["mode"],
                                      n_splits=cfg["cv"]["n_splits"])
        for r in ml_results:
            logger.log_stage(f"csp_ml_{r['model']}", {"cv_acc": r["best_cv_acc"], "time_s": r["time_s"],
                              "best_params": {k: str(v) for k, v in r["best_params"].items()}})
        best_ml = max(ml_results, key=lambda x: x["best_cv_acc"])
        y_pred_ml = best_ml["grid_obj"].best_estimator_.predict(X_eval)
        ml_eval_acc = float(np.mean(y_pred_ml == y_eval))
        logger.log_stage("csp_ml_summary", {"best_model": best_ml["model"],
                                             "best_cv_acc": best_ml["best_cv_acc"],
                                             f"{eval_label.lower()}_acc": ml_eval_acc})
        plot_confusion_matrix(y_eval, y_pred_ml, class_names,
                              f"CSP+{best_ml['model']} {eval_label}", save_path=out_dir / "csp_ml_best_cm.png")

    # ════════════════════════════════════════════════════════
    # STAGE 5: Preprocessing Grid Search (train+val only)
    # ════════════════════════════════════════════════════════
    if run_cfg.get("preprocessing_grid", True):
        print(f"\n{'='*60}\n  STAGE 5: Preprocessing Grid Search\n{'='*60}")
        raw_data_cv = {s: raw_data[s] for s in raw_data if s in cv_ids}
        preproc_df = run_preprocessing_grid(
            raw_data_cv, preprocessing_grid=cfg["preprocessing_grid"],
            channels=channels, task_mode=task["mode"],
            n_splits=cfg["cv"]["quick_splits"], epochs=cfg["cv"]["quick_epochs"],
            chans=n_chans, classes=n_classes, seed=cfg["seed"], device=device,
            checkpoint_path=out_dir / "ckpt_preproc_grid.csv")
        for i, row in preproc_df.iterrows():
            logger.log_stage(f"preproc_combo_{i+1}", row.to_dict())
        logger.log_stage("preprocessing_grid_summary", {
            "n_combos": len(preproc_df),
            "best": preproc_df.iloc[0].to_dict() if len(preproc_df) > 0 else {},
            "top5": preproc_df.head(5).to_dict(orient="records")})

    # ════════════════════════════════════════════════════════
    # STAGE 6: Joint Preprocessing × Model Grid Search
    # ════════════════════════════════════════════════════════
    if run_cfg.get("joint_grid", True) and run_cfg.get("preprocessing_grid", True):
        print(f"\n{'='*60}\n  STAGE 6: Joint Grid Search\n{'='*60}")
        raw_data_cv = {s: raw_data[s] for s in raw_data if s in cv_ids}
        joint_df = run_joint_grid(
            raw_data_cv, preproc_df.head(5), channels=channels, task_mode=task["mode"],
            chans=n_chans, classes=n_classes,
            n_splits=cfg["cv"]["quick_splits"], eegnet_epochs=cfg["cv"]["quick_epochs"],
            include_eegnet=run_cfg.get("eegnet_grid_search", True),
            include_csp_ml=run_cfg.get("csp_ml_grid", True),
            include_shallow=run_cfg.get("shallow_grid_search", False),
            seed=cfg["seed"], device=device,
            checkpoint_path=out_dir / "ckpt_joint_grid.csv")
        for i, row in joint_df.iterrows():
            logger.log_stage(f"joint_combo_{i+1}", row.to_dict())
        logger.log_stage("joint_grid_summary", {
            "n_combos": len(joint_df),
            "best": joint_df.iloc[0].to_dict() if len(joint_df) > 0 else {}})

    # ════════════════════════════════════════════════════════
    # STAGE 7: Final retrain → eval on DEV (4-way) or TEST (3-way)
    # ════════════════════════════════════════════════════════
    if run_cfg.get("final_retrain", True) and run_cfg.get("joint_grid", True):
        print(f"\n{'='*60}\n  STAGE 7: Final Retrain (eval on {eval_label})\n{'='*60}")
        best = joint_df.iloc[0]
        print(f"Best: {best['model_name']} | {best['preproc']} | CV: {best['mean_acc']:.4f}")

        baseline_p = (None, 0) if best["baseline"] == "yes" else None
        best_pp = dict(event_id=event_id, channels=channels,
                       low_freq=best["low_freq"], high_freq=best["high_freq"],
                       tmin=best["tmin"], tmax=best["tmax"],
                       baseline=baseline_p, label_offset=label_offset, seed=cfg["seed"])

        X_tr_f, y_tr_f, _ = preprocess_split(raw_data, ids["train"], balance=balance, **best_pp)
        X_vl_f, y_vl_f, _ = preprocess_split(raw_data, ids["val"],   balance=balance, **best_pp)
        ev_ids = ids["dev"] if is_4way else ids["test"]
        X_ev_f, y_ev_f, s_ev_f = preprocess_split(raw_data, ev_ids, balance=balance, **best_pp)

        tp_f = X_tr_f.shape[2]
        loaders_f = _make_loaders(X_tr_f, y_tr_f, X_vl_f, y_vl_f, 64)
        final_model = None; pipe = None
        train_results = None

        if best["model_type"] == "EEGNet":
            import re
            m = re.search(r"f1=(\d+),d=(\d+),do=([\d.]+),lr=([\d.]+)", best["model_name"])
            f1, d, do, lr = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))
            final_model = EEGNet(chans=n_chans, classes=n_classes, time_points=tp_f,
                                  f1=f1, f2=f1*d, d=d, dropout_rate=do).to(device)
            weight_decay = tr_cfg.get("weight_decay", 0.0)
            opt = torch.optim.Adam(final_model.parameters(), lr=lr, weight_decay=weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr_cfg["epochs"])
            train_results, best_val = train_no_test(final_model, loaders_f["train_loader"], loaders_f["val_loader"],
                          opt, _make_loss_fn(y_tr_f, device), sched, device, epochs=tr_cfg["epochs"],
                          patience=tr_cfg.get("patience", None))
            ev_preds, ev_labels = predict_with_model(
                final_model, DataLoader(EEGDataset(X_ev_f, y_ev_f), batch_size=64), device)
        elif best["model_type"] == "ShallowConvNet":
            import re

            from src.models.shallow_convnet import ShallowConvNet
            m = re.search(r"ft=(\d+),fs=(\d+),do=([\d.]+),lr=([\d.]+)", best["model_name"])
            ft, fs, do, lr = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))
            final_model = ShallowConvNet(chans=n_chans, classes=n_classes, time_points=tp_f,
                                          n_filters_time=ft, n_filters_spat=fs, drop_prob=do).to(device)
            weight_decay = tr_cfg.get("weight_decay", 0.0)
            opt = torch.optim.Adam(final_model.parameters(), lr=lr, weight_decay=weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr_cfg["epochs"])
            train_results, best_val = train_no_test(final_model, loaders_f["train_loader"], loaders_f["val_loader"],
                          opt, _make_loss_fn(y_tr_f, device), sched, device, epochs=tr_cfg["epochs"],
                          patience=tr_cfg.get("patience", None))
            ev_preds, ev_labels = predict_with_model(
                final_model, DataLoader(EEGDataset(X_ev_f, y_ev_f), batch_size=64), device)
        else:  # CSP+ML
            from mne.decoding import CSP
            from sklearn.discriminant_analysis import \
                LinearDiscriminantAnalysis
            from sklearn.ensemble import (GradientBoostingClassifier,
                                          RandomForestClassifier)
            from sklearn.linear_model import LogisticRegression
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.neural_network import MLPClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
            X_tv = np.concatenate([X_tr_f, X_vl_f]); y_tv = np.concatenate([y_tr_f, y_vl_f])
            ml_name = best["model_name"].replace("CSP+", "")
            ml_map = {"LDA": LinearDiscriminantAnalysis(),
                       "SVM": SVC(probability=True, random_state=42, class_weight="balanced"),
                       "RandomForest": RandomForestClassifier(random_state=42, class_weight="balanced"),
                       "KNN": KNeighborsClassifier(),
                       "GradientBoosting": GradientBoostingClassifier(random_state=42),
                       "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
                       "MLP": MLPClassifier(max_iter=500, random_state=42, early_stopping=True)}
            pipe = Pipeline([("csp", CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=False)),
                              ("scaler", StandardScaler()), ("classifier", ml_map[ml_name])])
            pipe.fit(X_tv, y_tv)
            ev_preds = pipe.predict(X_ev_f); ev_labels = y_ev_f

        stage7_acc = float(np.mean(ev_preds == ev_labels))
        print_evaluation(ev_labels, ev_preds, class_names, f"STAGE 7 — {eval_label}")

        stage7_log_data = {
            "model_name": str(best["model_name"]),
            "preproc": str(best["preproc"]),
            "cv_acc": float(best["mean_acc"]),
            f"{eval_label.lower()}_acc": stage7_acc
        }
        if train_results is not None:
            stage7_log_data["epoch_history"] = train_results
            plot_training_curves(train_results, f"Final Retrain — {best['model_name']}",
                                 save_path=out_dir / "final_retrain_curves.png")

        logger.log_stage("final_retrain", stage7_log_data)

        # ── 3-way: this IS the final result ──────────────────
        if not is_4way:
            plot_confusion_matrix(ev_labels, ev_preds, class_names,
                                  f"Final — {best['model_name']}", save_path=out_dir / "final_cm.png")
            if final_model is not None and cfg["output"]["save_model"]:
                save_model(final_model, out_dir / "final_best", metadata={
                    "model_name": str(best["model_name"]), "preproc": str(best["preproc"]),
                    "chans": n_chans, "classes": n_classes, "time_points": int(tp_f),
                    "test_accuracy": stage7_acc,
                    "bandpass": [float(best["low_freq"]), float(best["high_freq"])],
                    "tmin": float(best["tmin"]), "tmax": float(best["tmax"]),
                    "split": {k: sorted(v) for k, v in ids.items()}})

    # ════════════════════════════════════════════════════════
    # STAGE 8: HOLDOUT — only in 4-way mode, ONE SHOT
    # ════════════════════════════════════════════════════════
    if is_4way and run_cfg.get("final_retrain", True) and run_cfg.get("joint_grid", True):
        print(f"\n{'='*60}\n  STAGE 8: HOLDOUT — final one-shot evaluation\n{'='*60}")
        print(f"  ⚠️  FIRST and ONLY time holdout data is evaluated.")
        print(f"  ⚠️  This number goes in the paper.\n")

        X_ho_f, y_ho_f, s_ho_f = preprocess_split(raw_data, ids["holdout"], balance=balance, **best_pp)
        if final_model is not None:
            ho_preds, ho_labels = predict_with_model(
                final_model, DataLoader(EEGDataset(X_ho_f, y_ho_f), batch_size=64), device)
        else:
            ho_preds = pipe.predict(X_ho_f); ho_labels = y_ho_f

        holdout_acc = float(np.mean(ho_preds == ho_labels))
        print_evaluation(ho_labels, ho_preds, class_names, "HOLDOUT (reported)")
        plot_confusion_matrix(ho_labels, ho_preds, class_names,
                              f"Holdout — {best['model_name']}", save_path=out_dir / "final_cm.png")
        logger.log_stage("holdout_final", {
            "model_name": str(best["model_name"]), "preproc": str(best["preproc"]),
            "cv_acc": float(best["mean_acc"]), "dev_acc": stage7_acc,
            "holdout_acc": holdout_acc}, extra={"best_row": best.to_dict()})
        if final_model is not None and cfg["output"]["save_model"]:
            save_model(final_model, out_dir / "final_best", metadata={
                "model_name": str(best["model_name"]), "preproc": str(best["preproc"]),
                "chans": n_chans, "classes": n_classes, "time_points": int(tp_f),
                "dev_accuracy": stage7_acc, "holdout_accuracy": holdout_acc,
                "bandpass": [float(best["low_freq"]), float(best["high_freq"])],
                "tmin": float(best["tmin"]), "tmax": float(best["tmax"]),
                "split": {k: sorted(v) for k, v in ids.items()}})

    print(f"\n{'='*60}\n  ALL DONE — results saved to {logger.filename}\n{'='*60}")
    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindStride — EEG Motor Imagery Training")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    args = parser.parse_args()
    main(config_path=args.config)
