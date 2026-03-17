#!/usr/bin/env python3
"""
MindStride — CLI entry point.

Runs all enabled stages (single run, CV, grid searches, final retrain)
and saves results to JSON after every stage.

Fixes vs original:
  - Preprocessing grid receives only CV subjects (test excluded)
  - Final retrain uses val split for model selection (not test set)
  - Bad subjects excluded via loading module

Usage:
    python train.py                                    # default config
    python train.py --config configs/full_binary.yaml  # full pipeline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

from src.config import load_config
from src.data import (EEGDataset, download_dataset, epoch_subjects,
                      epoch_with_params, load_raw_subjects, make_dataloaders,
                      subject_split)
from src.engine import cross_validate_subjects, eval_step, train
from src.models import EEGNet
from src.pipelines import (run_csp_ml_grid, run_eegnet_grid, run_joint_grid,
                           run_preprocessing_grid, run_shallow_grid)
from src.utils import (ResultsLogger, get_device, plot_confusion_matrix,
                       plot_training_curves, predict_with_model,
                       print_evaluation, save_model, set_seeds)


def _make_loss_fn(y, device, weighted=True):
    if weighted:
        w = compute_class_weight("balanced", classes=np.unique(y), y=y)
        return nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(device))
    return nn.CrossEntropyLoss()


def main(config_path: str | None = None, overrides: dict | None = None):
    cfg = load_config(config_path, overrides)

    # ── Setup ───────────────────────────────────────────────
    set_seeds(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    out_dir = Path(cfg["output"]["save_dir"])
    run_cfg = cfg.get("run", {})
    logger = ResultsLogger(cfg, save_dir=out_dir)

    # ── Data loading ────────────────────────────────────────
    subjects_data = download_dataset(
        cfg["data"]["dataset"],
        cfg["data"]["desired_runs"],
    )
    raw_data = load_raw_subjects(
        subjects_data,
        sfreq=cfg["data"]["sfreq"],
        n_subjects=cfg["data"].get("n_subjects"),
        cache_dir=cfg["data"].get("cache_dir"),
    )

    # ── Preprocessing ───────────────────────────────────────
    task = cfg["task"]
    pp = cfg["preprocessing"]
    ch_cfg = cfg["channels"]

    channels = ch_cfg["motor_channels"] if ch_cfg["mode"] == "motor" else None
    n_chans = len(channels) if channels else 64

    if task["mode"] == "binary":
        event_id = task["event_id_binary"]
        label_offset = 2
        class_names = task["class_names_binary"]
        n_classes = 2
    else:
        event_id = task.get("event_id_ternary", {"rest": 1, "left_hand": 2, "right_hand": 3})
        label_offset = 1
        class_names = task.get("class_names_ternary", ["rest", "left_hand", "right_hand"])
        n_classes = 3

    X_all, y_all, subjects_all, skipped = epoch_subjects(
        raw_data,
        event_id=event_id,
        channels=channels,
        low_freq=pp["bandpass"][0],
        high_freq=pp["bandpass"][1],
        tmin=pp["tmin"],
        tmax=pp["tmax"],
        baseline=tuple(pp["baseline"]) if pp["baseline"] else None,
        normalize=pp["normalize"],
        balance=False,  # FIX: balance per-split, not globally
        label_offset=label_offset,
        seed=cfg["seed"],
    )

    logger.log_stage("data_loaded", {
        "n_epochs": int(X_all.shape[0]),
        "shape": list(X_all.shape),
        "n_subjects": int(len(np.unique(subjects_all))),
        "class_dist": np.bincount(y_all).tolist(),
        "skipped_subjects": skipped,
    })

    # ── Split (with per-split balancing) ────────────────────
    split = subject_split(
        X_all, y_all, subjects_all,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        seed=cfg["seed"],
        balance=pp.get("balance_classes", True),
    )
    loaders = make_dataloaders(split, batch_size=cfg["dataloader"]["batch_size"])
    cv_mask = split["train_mask"] | split["val_mask"]

    logger.log_stage("split", {
        "train": int(len(split["X_train"])),
        "val": int(len(split["X_val"])),
        "test": int(len(split["X_test"])),
        "train_class_dist": np.bincount(split["y_train"]).tolist(),
        "val_class_dist": np.bincount(split["y_val"]).tolist(),
        "test_class_dist": np.bincount(split["y_test"]).tolist(),
    })

    eeg_cfg = cfg["eegnet"]
    tr_cfg = cfg["training"]
    time_points = X_all.shape[2]

    # ════════════════════════════════════════════════════════
    # STAGE 1: Single EEGNet run
    # ════════════════════════════════════════════════════════
    if run_cfg.get("single_run", True):
        print("\n" + "=" * 60)
        print("  STAGE 1: Single EEGNet run")
        print("=" * 60)

        model = EEGNet(
            chans=n_chans, classes=n_classes, time_points=time_points,
            f1=eeg_cfg["f1"], f2=eeg_cfg["f1"] * eeg_cfg["d"], d=eeg_cfg["d"],
            temp_kernel=eeg_cfg["temp_kernel"], pk1=eeg_cfg["pk1"],
            pk2=eeg_cfg["pk2"], dropout_rate=eeg_cfg["dropout_rate"],
        ).to(device)

        loss_fn = _make_loss_fn(split["y_train"], device, tr_cfg["class_weighted_loss"])
        optimizer = torch.optim.Adam(model.parameters(), lr=tr_cfg["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tr_cfg["scheduler_T_max"])

        results, best_val = train(
            model, loaders["train_loader"], loaders["val_loader"], loaders["test_loader"],
            optimizer, loss_fn, scheduler, device, epochs=tr_cfg["epochs"],
        )

        preds, labels = predict_with_model(model, loaders["test_loader"], device)
        test_acc = float(np.mean(preds == labels))

        logger.log_stage("single_run", {
            "best_val_acc": float(best_val),
            "test_acc": test_acc,
            "final_train_loss": results["train_loss"][-1],
            "final_train_acc": results["train_acc"][-1],
            "final_val_loss": results["val_loss"][-1],
            "final_val_acc": results["val_acc"][-1],
            "epoch_history": {
                "train_loss": results["train_loss"],
                "train_acc": results["train_acc"],
                "val_loss": results["val_loss"],
                "val_acc": results["val_acc"],
            },
        }, extra={"eegnet": eeg_cfg, "lr": tr_cfg["lr"], "epochs": tr_cfg["epochs"]})

        plot_training_curves(results, "Single Run", save_path=out_dir / "single_run_curves.png")
        plot_confusion_matrix(labels, preds, class_names, "Single Run Test", save_path=out_dir / "single_run_cm.png")

        if cfg["output"]["save_model"]:
            save_model(model, out_dir / "eegnet_single_run")

    # ════════════════════════════════════════════════════════
    # STAGE 2: Subject-based Cross-Validation
    # ════════════════════════════════════════════════════════
    if run_cfg.get("cross_validation", True):
        print("\n" + "=" * 60)
        print("  STAGE 2: Cross-Validation")
        print("=" * 60)

        fold_results, mean_acc, std_acc = cross_validate_subjects(
            X_all[cv_mask], y_all[cv_mask], subjects_all[cv_mask],
            n_splits=cfg["cv"]["n_splits"], epochs=tr_cfg["epochs"], lr=tr_cfg["lr"],
            chans=n_chans, classes=n_classes, time_points=time_points, device=device,
        )

        logger.log_stage("cross_validation", {
            "mean_acc": float(mean_acc),
            "std_acc": float(std_acc),
            "fold_accs": [float(x) for x in fold_results],
            "n_splits": cfg["cv"]["n_splits"],
        })

    # ════════════════════════════════════════════════════════
    # STAGE 3: EEGNet Grid Search
    # ════════════════════════════════════════════════════════
    if run_cfg.get("eegnet_grid_search", True):
        print("\n" + "=" * 60)
        print("  STAGE 3: EEGNet Grid Search")
        print("=" * 60)

        grid_results = run_eegnet_grid(
            X_all[cv_mask], y_all[cv_mask], subjects_all[cv_mask],
            param_grid=cfg["eegnet_grid"], chans=n_chans, classes=n_classes,
            n_splits=cfg["cv"]["quick_splits"], epochs_train=cfg["cv"]["quick_epochs"],
            device=device,
        )

        for i, r in enumerate(grid_results):
            logger.log_stage(f"eegnet_grid_combo_{i+1}", {
                "mean_acc": r["mean_acc"], "std_acc": r["std_acc"],
            }, extra={k: v for k, v in r.items() if k not in ("mean_acc", "std_acc")})

        grid_df = pd.DataFrame(grid_results).sort_values("mean_acc", ascending=False)
        best_eegnet = grid_df.iloc[0].to_dict()

        logger.log_stage("eegnet_grid_summary", {
            "best_mean_acc": float(best_eegnet["mean_acc"]),
            "best_params": {k: v for k, v in best_eegnet.items() if k not in ("mean_acc", "std_acc")},
            "all_results": grid_df.to_dict(orient="records"),
        })

    # ════════════════════════════════════════════════════════
    # STAGE 3.5: ShallowConvNet Grid Search
    # ════════════════════════════════════════════════════════
    if run_cfg.get("shallow_grid_search", False):
        print("\n" + "=" * 60)
        print("  STAGE 3.5: ShallowConvNet Grid Search")
        print("=" * 60)

        shallow_results = run_shallow_grid(
            X_all[cv_mask], y_all[cv_mask], subjects_all[cv_mask],
            param_grid=cfg.get("shallow_grid"),
            chans=n_chans, classes=n_classes,
            n_splits=cfg["cv"]["quick_splits"],
            epochs_train=cfg["cv"]["quick_epochs"],
            device=device,
        )

        for i, r in enumerate(shallow_results):
            logger.log_stage(f"shallow_grid_combo_{i+1}", {
                "mean_acc": r["mean_acc"], "std_acc": r["std_acc"],
            }, extra={k: v for k, v in r.items() if k not in ("mean_acc", "std_acc")})

        shallow_df = pd.DataFrame(shallow_results).sort_values("mean_acc", ascending=False)
        best_shallow = shallow_df.iloc[0].to_dict()

        logger.log_stage("shallow_grid_summary", {
            "best_mean_acc": float(best_shallow["mean_acc"]),
            "best_params": {k: v for k, v in best_shallow.items() if k not in ("mean_acc", "std_acc")},
            "all_results": shallow_df.to_dict(orient="records"),
        })

    # ════════════════════════════════════════════════════════
    # STAGE 4: CSP + ML Grid Search
    # ════════════════════════════════════════════════════════
    if run_cfg.get("csp_ml_grid", True):
        print("\n" + "=" * 60)
        print("  STAGE 4: CSP + ML Grid Search")
        print("=" * 60)

        ml_results = run_csp_ml_grid(
            X_all[cv_mask], y_all[cv_mask], subjects_all[cv_mask],
            task_mode=task["mode"], n_splits=cfg["cv"]["n_splits"],
        )

        for r in ml_results:
            logger.log_stage(f"csp_ml_{r['model']}", {
                "cv_acc": r["best_cv_acc"], "time_s": r["time_s"],
                "best_params": {k: str(v) for k, v in r["best_params"].items()},
            })

        best_ml = max(ml_results, key=lambda x: x["best_cv_acc"])
        y_pred_ml = best_ml["grid_obj"].best_estimator_.predict(split["X_test"])
        ml_test_acc = float(np.mean(y_pred_ml == split["y_test"]))

        logger.log_stage("csp_ml_summary", {
            "best_model": best_ml["model"],
            "best_cv_acc": best_ml["best_cv_acc"],
            "test_acc": ml_test_acc,
            "all_models": [{"model": r["model"], "cv_acc": r["best_cv_acc"], "time_s": r["time_s"]} for r in ml_results],
        })

        plot_confusion_matrix(split["y_test"], y_pred_ml, class_names,
                              f"CSP+{best_ml['model']} Test", save_path=out_dir / "csp_ml_best_cm.png")

    # ════════════════════════════════════════════════════════
    # STAGE 5: Preprocessing Grid Search
    # ════════════════════════════════════════════════════════
    if run_cfg.get("preprocessing_grid", True):
        print("\n" + "=" * 60)
        print("  STAGE 5: Preprocessing Grid Search")
        print("=" * 60)

        # FIX: only CV subjects — test subjects excluded
        cv_subjects = split["train_subjects"] | split["val_subjects"]
        raw_data_cv = {s: raw_data[s] for s in raw_data if s in cv_subjects}

        preproc_df = run_preprocessing_grid(
            raw_data_cv, preprocessing_grid=cfg["preprocessing_grid"],
            channels=channels, task_mode=task["mode"],
            n_splits=cfg["cv"]["quick_splits"], epochs=cfg["cv"]["quick_epochs"],
            chans=n_chans, classes=n_classes, seed=cfg["seed"], device=device,
        )

        for i, row in preproc_df.iterrows():
            logger.log_stage(f"preproc_combo_{i+1}", row.to_dict())

        logger.log_stage("preprocessing_grid_summary", {
            "n_combos": len(preproc_df),
            "best": preproc_df.iloc[0].to_dict() if len(preproc_df) > 0 else {},
            "top5": preproc_df.head(5).to_dict(orient="records"),
        })

    # ════════════════════════════════════════════════════════
    # STAGE 6: Joint Preprocessing × Model Grid Search
    # ════════════════════════════════════════════════════════
    if run_cfg.get("joint_grid", True) and run_cfg.get("preprocessing_grid", True):
        print("\n" + "=" * 60)
        print("  STAGE 6: Joint Grid Search")
        print("=" * 60)

        top_preproc = preproc_df.head(5)
        joint_df = run_joint_grid(
            raw_data_cv, top_preproc, channels=channels, task_mode=task["mode"],
            chans=n_chans, classes=n_classes,
            n_splits=cfg["cv"]["quick_splits"], eegnet_epochs=cfg["cv"]["quick_epochs"],
            seed=cfg["seed"], device=device,
        )

        for i, row in joint_df.iterrows():
            logger.log_stage(f"joint_combo_{i+1}", row.to_dict())

        logger.log_stage("joint_grid_summary", {
            "n_combos": len(joint_df),
            "best": joint_df.iloc[0].to_dict() if len(joint_df) > 0 else {},
            "top10": joint_df.head(10).to_dict(orient="records"),
        })

    # ════════════════════════════════════════════════════════
    # STAGE 7: Final retrain with best config
    # ════════════════════════════════════════════════════════
    if run_cfg.get("final_retrain", True) and run_cfg.get("joint_grid", True):
        print("\n" + "=" * 60)
        print("  STAGE 7: Final Retrain")
        print("=" * 60)

        best = joint_df.iloc[0]
        print(f"Best: {best['model_name']} | {best['preproc']} | CV: {best['mean_acc']:.4f}")

        baseline_param = (None, 0) if best["baseline"] == "yes" else None
        X_final, y_final, subj_final = epoch_with_params(
            raw_data, best["low_freq"], best["high_freq"],
            best["tmin"], best["tmax"], baseline=baseline_param,
            channels=channels, task_mode=task["mode"], seed=cfg["seed"],
        )

        split_final = subject_split(X_final, y_final, subj_final, seed=cfg["seed"],
                                    balance=pp.get("balance_classes", True))
        loaders_final = make_dataloaders(split_final, batch_size=64)

        final_model = None

        if best["model_type"] == "EEGNet":
            import re
            m = re.search(r"f1=(\d+),d=(\d+),do=([\d.]+),lr=([\d.]+)", best["model_name"])
            f1, d, do, lr = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))

            final_model = EEGNet(
                chans=n_chans, classes=n_classes, time_points=X_final.shape[2],
                f1=f1, f2=f1 * d, d=d, dropout_rate=do,
            ).to(device)
            opt = torch.optim.Adam(final_model.parameters(), lr=lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
            loss_fn = _make_loss_fn(split_final["y_train"], device)

            # FIX: use val_loader for model selection, test_loader only for final eval
            train(final_model, loaders_final["train_loader"], loaders_final["val_loader"],
                  loaders_final["test_loader"], opt, loss_fn, sched, device, epochs=100)

            preds, labels = predict_with_model(final_model, loaders_final["test_loader"], device)
        elif best["model_type"] == "ShallowConvNet":
            import re
            m = re.search(r"ft=(\d+),fs=(\d+),do=([\d.]+),lr=([\d.]+)", best["model_name"])
            ft, fs, do, lr = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))

            from src.models.shallow_convnet import ShallowConvNet
            final_model = ShallowConvNet(
                chans=n_chans, classes=n_classes, time_points=X_final.shape[2],
                n_filters_time=ft, n_filters_spat=fs, drop_prob=do,
            ).to(device)
            opt = torch.optim.Adam(final_model.parameters(), lr=lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
            loss_fn = _make_loss_fn(split_final["y_train"], device)

            # FIX: use val_loader for model selection, test_loader only for final eval
            train(final_model, loaders_final["train_loader"], loaders_final["val_loader"],
                  loaders_final["test_loader"], opt, loss_fn, sched, device, epochs=100)

            preds, labels = predict_with_model(final_model, loaders_final["test_loader"], device)
        else:
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

            X_trainval = np.concatenate([split_final["X_train"], split_final["X_val"]])
            y_trainval = np.concatenate([split_final["y_train"], split_final["y_val"]])

            ml_name = best["model_name"].replace("CSP+", "")
            ml_map = {
                "LDA": LinearDiscriminantAnalysis(),
                "SVM": SVC(probability=True, random_state=42, class_weight="balanced"),
                "RandomForest": RandomForestClassifier(random_state=42, class_weight="balanced"),
                "KNN": KNeighborsClassifier(),
                "GradientBoosting": GradientBoostingClassifier(random_state=42),
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
                "MLP": MLPClassifier(max_iter=500, random_state=42, early_stopping=True),
            }
            pipe = Pipeline([
                ("csp", CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=False)),
                ("scaler", StandardScaler()),
                ("classifier", ml_map[ml_name]),
            ])
            pipe.fit(X_trainval, y_trainval)
            preds = pipe.predict(split_final["X_test"])
            labels = split_final["y_test"]

        test_acc = float(np.mean(preds == labels))
        print_evaluation(labels, preds, class_names, "FINAL TEST")
        plot_confusion_matrix(labels, preds, class_names, f"Final — {best['model_name']}",
                              save_path=out_dir / "final_cm.png")

        logger.log_stage("final_retrain", {
            "model_name": str(best["model_name"]),
            "preproc": str(best["preproc"]),
            "cv_acc": float(best["mean_acc"]),
            "test_acc": test_acc,
        }, extra={"best_row": best.to_dict()})

        if final_model is not None and cfg["output"]["save_model"]:
            save_model(final_model, out_dir / "final_best", metadata={
                "model_name": str(best["model_name"]),
                "preproc": str(best["preproc"]),
                "chans": n_chans, "classes": n_classes,
                "time_points": int(X_final.shape[2]),
                "test_accuracy": test_acc,
                "bandpass": [float(best["low_freq"]), float(best["high_freq"])],
                "tmin": float(best["tmin"]), "tmax": float(best["tmax"]),
            })

    print(f"\n{'=' * 60}")
    print(f"  ALL DONE — results saved to {logger.filename}")
    print(f"{'=' * 60}")
    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindStride — EEG Motor Imagery Training")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    args = parser.parse_args()
    main(config_path=args.config)