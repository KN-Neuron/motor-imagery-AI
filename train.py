#!/usr/bin/env python3
"""
MindStride — CLI entry point.

Usage:
    python train.py                          # use default config
    python train.py --config configs/my.yaml # custom config
    python train.py --config configs/my.yaml --task.mode binary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.data import (
    download_dataset,
    load_raw_subjects,
    epoch_subjects,
    subject_split,
    make_dataloaders,
)
from src.models import EEGNet
from src.engine import train, cross_validate_subjects
from src.utils import (
    set_seeds,
    get_device,
    plot_training_curves,
    plot_confusion_matrix,
    print_evaluation,
    save_model,
    predict_with_model,
)


def main(config_path: str | None = None, overrides: dict | None = None):
    cfg = load_config(config_path, overrides)

    # ── Setup ───────────────────────────────────────────────
    set_seeds(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

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
        event_id = task["event_id_ternary"]
        label_offset = 1
        class_names = task["class_names_ternary"]
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
        balance=pp["balance_classes"],
        label_offset=label_offset,
        seed=cfg["seed"],
    )

    # ── Split ───────────────────────────────────────────────
    split = subject_split(
        X_all, y_all, subjects_all,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        seed=cfg["seed"],
    )

    loaders = make_dataloaders(
        split,
        batch_size=cfg["dataloader"]["batch_size"],
        num_workers=cfg["dataloader"]["num_workers"],
    )

    print(f"\nTrain: {len(split['X_train'])} | Val: {len(split['X_val'])} | Test: {len(split['X_test'])}")

    # ── Model ───────────────────────────────────────────────
    eeg_cfg = cfg["eegnet"]
    model = EEGNet(
        chans=n_chans,
        classes=n_classes,
        time_points=X_all.shape[2],
        f1=eeg_cfg["f1"],
        f2=eeg_cfg["f1"] * eeg_cfg["d"],
        d=eeg_cfg["d"],
        temp_kernel=eeg_cfg["temp_kernel"],
        pk1=eeg_cfg["pk1"],
        pk2=eeg_cfg["pk2"],
        dropout_rate=eeg_cfg["dropout_rate"],
    ).to(device)

    # ── Loss & optimizer ────────────────────────────────────
    tr_cfg = cfg["training"]
    if tr_cfg["class_weighted_loss"]:
        from sklearn.utils.class_weight import compute_class_weight
        weights = compute_class_weight(
            "balanced", classes=np.unique(split["y_train"]), y=split["y_train"]
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32).to(device)
        )
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=tr_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tr_cfg["scheduler_T_max"]
    )

    # ── Train ───────────────────────────────────────────────
    results, best_val_acc = train(
        model=model,
        train_dataloader=loaders["train_loader"],
        val_dataloader=loaders["val_loader"],
        test_dataloader=loaders["test_loader"],
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        device=device,
        epochs=tr_cfg["epochs"],
    )

    # ── Evaluate ────────────────────────────────────────────
    plot_training_curves(results, title=f"EEGNet {task['mode']}")

    preds, labels = predict_with_model(model, loaders["test_loader"], device)
    test_acc = print_evaluation(labels, preds, class_names, "Test Results")
    plot_confusion_matrix(labels, preds, class_names, "Test Confusion Matrix")

    # ── Save ────────────────────────────────────────────────
    if cfg["output"]["save_model"]:
        out_dir = Path(cfg["output"]["save_dir"])
        meta = {
            "task_mode": task["mode"],
            "chans": n_chans,
            "classes": n_classes,
            "time_points": int(X_all.shape[2]),
            "bandpass": pp["bandpass"],
            "tmin": pp["tmin"],
            "tmax": pp["tmax"],
            "test_accuracy": float(test_acc),
            "best_val_accuracy": float(best_val_acc),
            **{f"eegnet_{k}": v for k, v in eeg_cfg.items()},
        }
        save_model(model, out_dir / "eegnet_best", metadata=meta)

    return results, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindStride — EEG Motor Imagery Training")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    args = parser.parse_args()
    main(config_path=args.config)
