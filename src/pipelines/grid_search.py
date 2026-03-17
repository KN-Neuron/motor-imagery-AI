"""
Grid search pipelines — EEGNet hyperparameters, preprocessing, and joint search.

Fixes vs original:
  - run_preprocessing_grid accepts cv_raw_data (caller must filter test subjects)
  - Removed self-import of run_shallow_grid in run_joint_grid
"""

from __future__ import annotations

import itertools
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mne.decoding import CSP
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.data.preprocessing import epoch_with_params
from src.engine import (cross_validate_subjects, cv_for_preprocessing,
                        eval_step, train_step)
from src.models.eegnet import EEGNet
from src.models.shallow_convnet import ShallowConvNet
from src.pipelines.csp_ml import get_ml_models


def run_shallow_grid(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    param_grid: dict[str, list] | None = None,
    chans: int = 21,
    classes: int = 2,
    n_splits: int = 3,
    epochs_train: int = 30,
    device: str | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """ShallowConvNet hyperparameter grid search with subject-based CV."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if param_grid is None:
        param_grid = {
            "n_filters_time": [40],
            "n_filters_spat": [40],
            "filter_time_length": [25],
            "pool_time_length": [75],
            "pool_time_stride": [15],
            "drop_prob": [0.5],
            "lr": [0.001],
        }

    time_points = X.shape[2]
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        lr = params.pop("lr")

        if verbose:
            print(f"\nShallow combo {i+1}/{len(combos)}: lr={lr}, {params}")

        gkf = GroupKFold(n_splits=n_splits)
        fold_accs = []

        for train_idx, val_idx in gkf.split(X, y, groups=subjects):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_vl, y_vl = X[val_idx], y[val_idx]

            train_dl = DataLoader(EEGDataset(X_tr, y_tr), batch_size=64, shuffle=True)
            val_dl = DataLoader(EEGDataset(X_vl, y_vl), batch_size=64, shuffle=False)

            model = ShallowConvNet(
                chans=chans, classes=classes, time_points=time_points, **params,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
            loss_fn = nn.CrossEntropyLoss(
                weight=torch.tensor(weights, dtype=torch.float32).to(device)
            )

            best_val_acc = 0.0
            for epoch in range(epochs_train):
                train_step(model, train_dl, loss_fn, optimizer, device)
                _, val_acc = eval_step(model, val_dl, loss_fn, device)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

            fold_accs.append(best_val_acc)

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)

        if verbose:
            print(f"  → {mean_acc:.4f} ± {std_acc:.4f}")

        results.append({**params, "lr": lr, "mean_acc": mean_acc, "std_acc": std_acc})

    return results


def run_eegnet_grid(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    param_grid: dict[str, list] | None = None,
    chans: int = 21,
    classes: int = 2,
    n_splits: int = 3,
    epochs_train: int = 30,
    device: str | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """EEGNet hyperparameter grid search with subject-based CV."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if param_grid is None:
        param_grid = {
            "lr": [0.001, 0.0005],
            "dropout_rate": [0.25, 0.5],
            "f1": [8, 16],
            "d": [2],
        }

    time_points = X.shape[2]
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        f2 = params["f1"] * params["d"]

        if verbose:
            print(f"\nCombo {i+1}/{len(combos)}: {params}, f2={f2}")

        gkf = GroupKFold(n_splits=n_splits)
        fold_accs = []

        for train_idx, val_idx in gkf.split(X, y, groups=subjects):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_vl, y_vl = X[val_idx], y[val_idx]

            train_dl = DataLoader(EEGDataset(X_tr, y_tr), batch_size=64, shuffle=True)
            val_dl = DataLoader(EEGDataset(X_vl, y_vl), batch_size=64, shuffle=False)

            model = EEGNet(
                chans=chans, classes=classes, time_points=time_points,
                f1=params["f1"], f2=f2, d=params["d"],
                dropout_rate=params["dropout_rate"],
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

            weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
            loss_fn = nn.CrossEntropyLoss(
                weight=torch.tensor(weights, dtype=torch.float32).to(device)
            )

            best_val_acc = 0.0
            for epoch in range(epochs_train):
                train_step(model, train_dl, loss_fn, optimizer, device)
                _, val_acc = eval_step(model, val_dl, loss_fn, device)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

            fold_accs.append(best_val_acc)

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)

        if verbose:
            print(f"  → {mean_acc:.4f} ± {std_acc:.4f}")

        results.append({**params, "f2": f2, "mean_acc": mean_acc, "std_acc": std_acc})

    return results


def run_preprocessing_grid(
    raw_data: dict,
    preprocessing_grid: dict,
    channels: list[str] | None = None,
    task_mode: str = "binary",
    n_splits: int = 3,
    epochs: int = 30,
    chans: int = 21,
    classes: int = 2,
    seed: int = 42,
    device: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run preprocessing hyperparameter grid search.

    IMPORTANT: Caller must pass only CV subjects in raw_data
    (test subjects must be filtered out before calling this).
    """
    results = []
    combos = list(itertools.product(
        preprocessing_grid["bandpass"],
        preprocessing_grid["time_window"],
        preprocessing_grid["baseline"],
    ))

    for i, ((low, high), (tmin, tmax), baseline) in enumerate(combos):
        baseline_tuple = tuple(baseline) if baseline is not None else None
        label = f"bp={low}-{high}Hz, t={tmin}-{tmax}s, bl={'yes' if baseline else 'no'}"

        if verbose:
            print(f"\n[{i+1}/{len(combos)}] {label}")

        start = time.time()
        try:
            X, y, subjects = epoch_with_params(
                raw_data, low, high, tmin, tmax,
                baseline=baseline_tuple,
                channels=channels,
                task_mode=task_mode,
                seed=seed,
            )

            mean_acc, std_acc = cv_for_preprocessing(
                X, y, subjects,
                n_splits=n_splits, epochs=epochs,
                chans=chans, classes=classes, device=device,
            )
            elapsed = time.time() - start

            if verbose:
                print(f"  → {mean_acc:.4f} ± {std_acc:.4f} ({elapsed:.0f}s)")

            results.append({
                "low_freq": low, "high_freq": high,
                "tmin": tmin, "tmax": tmax,
                "baseline": "yes" if baseline else "no",
                "time_points": X.shape[2],
                "n_epochs": X.shape[0],
                "mean_acc": mean_acc, "std_acc": std_acc,
                "time_s": elapsed,
            })

        except Exception as e:
            if verbose:
                print(f"  ✗ FAILED: {e}")
            results.append({
                "low_freq": low, "high_freq": high,
                "tmin": tmin, "tmax": tmax,
                "baseline": "yes" if baseline else "no",
                "time_points": None, "n_epochs": None,
                "mean_acc": None, "std_acc": None,
                "time_s": time.time() - start,
            })

    df = pd.DataFrame(results)
    df = df.dropna(subset=["mean_acc"])
    return df.sort_values("mean_acc", ascending=False).reset_index(drop=True)


def run_joint_grid(
    raw_data: dict,
    top_preproc: pd.DataFrame,
    channels: list[str] | None = None,
    task_mode: str = "binary",
    chans: int = 21,
    classes: int = 2,
    n_splits: int = 3,
    eegnet_epochs: int = 30,
    seed: int = 42,
    device: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Joint preprocessing × model grid search.

    IMPORTANT: Caller must pass only CV subjects in raw_data.
    """
    all_results = []

    for pp_idx, (_, pp_row) in enumerate(top_preproc.iterrows()):
        baseline_tuple = (None, 0) if pp_row["baseline"] == "yes" else None
        preproc_label = (
            f"bp={int(pp_row['low_freq'])}-{int(pp_row['high_freq'])}Hz, "
            f"t={pp_row['tmin']}-{pp_row['tmax']}s, "
            f"bl={pp_row['baseline']}"
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"Preprocessing [{pp_idx+1}/{len(top_preproc)}]: {preproc_label}")

        X, y, subjects = epoch_with_params(
            raw_data,
            pp_row["low_freq"], pp_row["high_freq"],
            pp_row["tmin"], pp_row["tmax"],
            baseline=baseline_tuple,
            channels=channels,
            task_mode=task_mode,
            seed=seed,
        )

        # --- EEGNet grid ---
        eegnet_results = run_eegnet_grid(
            X, y, subjects,
            chans=chans, classes=classes,
            n_splits=n_splits, epochs_train=eegnet_epochs,
            device=device, verbose=verbose,
        )
        for r in eegnet_results:
            all_results.append({
                "model_name": f"EEGNet(f1={r['f1']},d={r['d']},do={r['dropout_rate']},lr={r['lr']})",
                "model_type": "EEGNet",
                "preproc": preproc_label,
                "low_freq": pp_row["low_freq"],
                "high_freq": pp_row["high_freq"],
                "tmin": pp_row["tmin"],
                "tmax": pp_row["tmax"],
                "baseline": pp_row["baseline"],
                "mean_acc": r["mean_acc"],
                "std_acc": r["std_acc"],
            })

        # --- CSP + ML grid ---
        from src.pipelines.csp_ml import run_csp_ml_grid
        ml_results = run_csp_ml_grid(
            X, y, subjects,
            task_mode=task_mode,
            n_splits=n_splits,
            verbose=verbose,
        )
        for r in ml_results:
            all_results.append({
                "model_name": f"CSP+{r['model']}",
                "model_type": "CSP+ML",
                "preproc": preproc_label,
                "low_freq": pp_row["low_freq"],
                "high_freq": pp_row["high_freq"],
                "tmin": pp_row["tmin"],
                "tmax": pp_row["tmax"],
                "baseline": pp_row["baseline"],
                "mean_acc": r["best_cv_acc"],
                "std_acc": 0.0,
            })

        # --- ShallowConvNet grid ---
        shallow_results = run_shallow_grid(
            X, y, subjects,
            chans=chans, classes=classes,
            n_splits=n_splits, epochs_train=eegnet_epochs,
            device=device, verbose=verbose,
        )
        for r in shallow_results:
            all_results.append({
                "model_name": f"Shallow(ft={r['n_filters_time']},fs={r['n_filters_spat']},do={r['drop_prob']},lr={r['lr']})",
                "model_type": "ShallowConvNet",
                "preproc": preproc_label,
                "low_freq": pp_row["low_freq"],
                "high_freq": pp_row["high_freq"],
                "tmin": pp_row["tmin"],
                "tmax": pp_row["tmax"],
                "baseline": pp_row["baseline"],
                "mean_acc": r["mean_acc"],
                "std_acc": r["std_acc"],
            })

    df = pd.DataFrame(all_results)
    return df.sort_values("mean_acc", ascending=False).reset_index(drop=True)