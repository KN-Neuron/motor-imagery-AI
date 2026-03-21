"""
Training engine — train_step, eval_step, train loop, cross-validation.

Fixes vs original:
  - Per-sample accuracy (not per-batch average)
  - CosineAnnealingLR in cross_validate_subjects
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from src.data.dataset import EEGDataset
from src.models.eegnet import EEGNet


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """Single training epoch. Returns (loss, accuracy) per-sample."""
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        total_correct += (y_pred.argmax(dim=1) == y).sum().item()
        total_samples += len(y)

    return total_loss / total_samples, total_correct / total_samples


def eval_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Single evaluation pass. Returns (loss, accuracy) per-sample."""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)

            total_loss += loss.item() * len(y)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += len(y)

    return total_loss / total_samples, total_correct / total_samples


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scheduler: Any,
    device: str,
    epochs: int = 50,
    verbose: bool = True,
    patience: int | None = None,
) -> tuple[dict[str, list[float]], float]:
    """
    Full training loop with best-model checkpointing on val set.
    """
    results: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "test_loss": [], "test_acc": [],
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
            print(
                f"Epoch {epoch+1:3d} | "
                f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}" +
                (" (Best)" if is_best else f" (no imp: {epochs_no_improve})")
            )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        
        if patience is not None and epochs_no_improve >= patience:
            if verbose:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    # Restore best model and evaluate on test set
    if best_state:
        model.load_state_dict(best_state)
    test_loss, test_acc = eval_step(model, test_dataloader, loss_fn, device)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

    if verbose:
        print(f"\nBest val_acc: {best_val_acc:.4f}")
        print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

    return results, best_val_acc


def cross_validate_subjects(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    n_splits: int = 5,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 64,
    dropout_rate: float = 0.5,
    f1: int = 8,
    f2: int = 16,
    d: int = 2,
    temp_kernel: int = 80,
    chans: int = 64,
    classes: int = 3,
    time_points: int | None = None,
    device: str | None = None,
    verbose: bool = True,
    weight_decay: float = 0.0,
    patience: int | None = None,
) -> tuple[list[float], float, float]:
    """
    Subject-based K-Fold cross-validation for EEGNet.

    Now includes CosineAnnealingLR scheduler for consistency with training.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if time_points is None:
        time_points = X.shape[2]

    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=subjects)):
        if verbose:
            print(f"\nFold {fold+1}/{n_splits} — "
                  f"Train: {len(train_idx)}, Val: {len(val_idx)}")

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_vl, y_vl = X[val_idx], y[val_idx]

        train_dl = DataLoader(EEGDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(EEGDataset(X_vl, y_vl), batch_size=batch_size, shuffle=False)

        model = EEGNet(
            chans=chans, classes=classes, time_points=time_points,
            f1=f1, f2=f2, d=d, dropout_rate=dropout_rate, temp_kernel=temp_kernel,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        fold_weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(fold_weights, dtype=torch.float32).to(device)
        )

        best_val_acc = 0.0
        epochs_no_improve = 0
        for epoch in range(epochs):
            train_step(model, train_dl, loss_fn, optimizer, device)
            scheduler.step()
            _, val_acc = eval_step(model, val_dl, loss_fn, device)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if patience is not None and epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        if verbose:
            print(f"Fold {fold+1} best val_acc: {best_val_acc:.4f}")
        fold_results.append(best_val_acc)

    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)

    if verbose:
        print(f"\nCV: {mean_acc:.4f} ± {std_acc:.4f}")

    return fold_results, mean_acc, std_acc


def cv_for_preprocessing(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    n_splits: int = 3,
    epochs: int = 30,
    lr: float = 0.001,
    chans: int = 21,
    classes: int = 2,
    device: str | None = None,
) -> tuple[float, float]:
    """Lightweight CV for preprocessing grid search."""
    _, mean_acc, std_acc = cross_validate_subjects(
        X, y, subjects,
        n_splits=n_splits, epochs=epochs, lr=lr,
        chans=chans, classes=classes,
        time_points=X.shape[2],
        device=device, verbose=False,
    )
    return mean_acc, std_acc