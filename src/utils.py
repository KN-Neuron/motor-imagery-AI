"""
Utility functions — plotting, model saving/loading, seeding.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ── Reproducibility ──────────────────────────────────────────

def set_seeds(seed: int = 42) -> None:
    """Set seeds for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── Plotting ─────────────────────────────────────────────────

def plot_training_curves(
    results: dict[str, list[float]],
    title: str = "",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot loss and accuracy curves from a training results dict.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(results["train_loss"], label="Train")
    axes[0].plot(results["val_loss"], label="Val")
    axes[0].set_title(f"Loss{' — ' + title if title else ''}")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(results["train_acc"], label="Train")
    axes[1].plot(results["val_acc"], label="Val")
    axes[1].set_title(f"Accuracy{' — ' + title if title else ''}")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    save_path: str | Path | None = None,
) -> None:
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig_size = (6, 5) if len(class_names) <= 2 else (8, 6)
    plt.figure(figsize=fig_size)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_comparison(
    results_df,
    metric_col: str = "CV Accuracy",
    title: str = "Model Comparison",
    save_path: str | Path | None = None,
) -> None:
    """Horizontal bar chart comparing model accuracies."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_df)))
    bars = ax.barh(results_df["Model"], results_df[metric_col], color=colors)
    ax.set_xlabel(metric_col)
    ax.set_title(title)
    ax.set_xlim(0.3, max(results_df[metric_col]) + 0.05)
    for bar, acc in zip(bars, results_df[metric_col]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{acc:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str = "",
) -> float:
    """Print classification report and return accuracy."""
    acc = accuracy_score(y_true, y_pred)
    if title:
        print(f"\n{'='*50}\n  {title}\n{'='*50}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    return acc


# ── Model saving / loading ───────────────────────────────────

def save_model(
    model: torch.nn.Module,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save model state_dict and optional metadata JSON.

    Saves:
      - ``{path}.pth`` — model weights
      - ``{path}_meta.json`` — metadata (if provided)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path.with_suffix(".pth"))
    print(f"Model saved to {path.with_suffix('.pth')}")

    if metadata:
        meta_path = path.with_name(path.stem + "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {meta_path}")


def load_model(
    model_class: type,
    path: str | Path,
    **model_kwargs,
) -> torch.nn.Module:
    """Load a model from a .pth file."""
    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def predict_with_model(
    model: torch.nn.Module,
    dataloader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference and return (predictions, labels) as numpy arrays.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for X_batch, y_batch in dataloader:
            logits = model(X_batch.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    return np.array(all_preds), np.array(all_labels)
