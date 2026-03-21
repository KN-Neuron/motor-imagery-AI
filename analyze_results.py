#!/usr/bin/env python3
"""
Generate analysis plots from MindStride results JSONs.

Usage:
    python analyze_results.py outputs/20260308_085439_binary_motor.json
    python analyze_results.py outputs/*.json          # all runs
    python analyze_results.py outputs/*.json --save    # save PNGs instead of showing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_results(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_grid_results(data: dict, prefix: str) -> pd.DataFrame:
    """Extract grid search combos from stages matching a prefix."""
    rows = []
    for stage in data["stages"]:
        if stage["stage"].startswith(prefix) and "combo" in stage["stage"]:
            row = {}
            row.update(stage.get("metrics", {}))
            row.update(stage.get("details", {}))
            rows.append(row)
    return pd.DataFrame(rows)


def extract_summary(data: dict, stage_name: str) -> dict | None:
    for stage in data["stages"]:
        if stage["stage"] == stage_name:
            return stage.get("metrics", {})
    return None


# ═══════════════════════════════════════════════════════════════
# Plot functions
# ═══════════════════════════════════════════════════════════════

def plot_eegnet_grid(data: dict, save_dir: Path | None = None):
    """EEGNet grid search — accuracy by hyperparameter."""
    df = extract_grid_results(data, "eegnet_grid_combo")
    if df.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("EEGNet Grid Search — Hyperparameter Analysis", fontsize=14, fontweight="bold")

    for ax, param in zip(axes.flat, ["f1", "d", "lr", "dropout_rate", "temp_kernel", "f2"]):
        if param not in df.columns:
            ax.set_visible(False)
            continue
        groups = df.groupby(param)["mean_acc"]
        vals = [g.values for _, g in groups]
        labels = [str(k) for k, _ in groups]
        bp = ax.boxplot(vals, labels=labels, patch_artist=True)
        colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(vals)))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.set_title(param, fontweight="bold")
        ax.set_ylabel("CV Accuracy")

    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "eegnet_grid_params.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_shallow_grid(data: dict, save_dir: Path | None = None):
    """ShallowConvNet grid search — accuracy by hyperparameter."""
    df = extract_grid_results(data, "shallow_grid_combo")
    if df.empty:
        return

    params = ["n_filters_time", "n_filters_spat", "filter_time_length",
              "pool_time_length", "drop_prob", "lr"]
    available = [p for p in params if p in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("ShallowConvNet Grid Search — Hyperparameter Analysis", fontsize=14, fontweight="bold")

    for ax, param in zip(axes.flat, available):
        groups = df.groupby(param)["mean_acc"]
        vals = [g.values for _, g in groups]
        labels = [str(k) for k, _ in groups]
        bp = ax.boxplot(vals, labels=labels, patch_artist=True)
        colors = plt.cm.Oranges(np.linspace(0.4, 0.85, len(vals)))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.set_title(param, fontweight="bold")
        ax.set_ylabel("CV Accuracy")

    for ax in axes.flat[len(available):]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "shallow_grid_params.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_csp_ml_comparison(data: dict, save_dir: Path | None = None):
    """CSP+ML model comparison bar chart."""
    rows = []
    for stage in data["stages"]:
        if stage["stage"].startswith("csp_ml_") and stage["stage"] != "csp_ml_summary":
            name = stage["stage"].replace("csp_ml_", "")
            metrics = stage.get("metrics", {})
            rows.append({"Model": f"CSP+{name}", "CV Accuracy": metrics.get("cv_acc", 0)})

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("CV Accuracy", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
    bars = ax.barh(df["Model"], df["CV Accuracy"], color=colors)
    ax.set_xlabel("CV Accuracy")
    ax.set_title("CSP + ML Model Comparison", fontweight="bold")
    ax.set_xlim(max(0.4, df["CV Accuracy"].min() - 0.05), df["CV Accuracy"].max() + 0.03)
    for bar, acc in zip(bars, df["CV Accuracy"]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{acc:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "csp_ml_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_preprocessing_grid(data: dict, save_dir: Path | None = None):
    """Preprocessing grid — accuracy by bandpass, time window, baseline."""
    df = extract_grid_results(data, "preproc_combo")
    if df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Preprocessing Grid Search", fontsize=14, fontweight="bold")

    # Bandpass
    if "low_freq" in df.columns and "high_freq" in df.columns:
        df["bandpass"] = df["low_freq"].astype(int).astype(str) + "–" + df["high_freq"].astype(int).astype(str) + " Hz"
        bp_order = df.groupby("bandpass")["mean_acc"].median().sort_values(ascending=False).index
        bp_data = [df[df["bandpass"] == bp]["mean_acc"].values for bp in bp_order]
        bp_plot = axes[0].boxplot(bp_data, labels=bp_order, patch_artist=True, vert=True)
        for patch, c in zip(bp_plot["boxes"], plt.cm.Blues(np.linspace(0.4, 0.85, len(bp_order)))):
            patch.set_facecolor(c)
        axes[0].set_title("By Bandpass", fontweight="bold")
        axes[0].tick_params(axis="x", rotation=25)

    # Time window
    if "tmin" in df.columns and "tmax" in df.columns:
        df["window"] = df["tmin"].astype(str) + "–" + df["tmax"].astype(str) + "s"
        tw_order = df.groupby("window")["mean_acc"].median().sort_values(ascending=False).index
        tw_data = [df[df["window"] == tw]["mean_acc"].values for tw in tw_order]
        tw_plot = axes[1].boxplot(tw_data, labels=tw_order, patch_artist=True, vert=True)
        for patch, c in zip(tw_plot["boxes"], plt.cm.Oranges(np.linspace(0.4, 0.85, len(tw_order)))):
            patch.set_facecolor(c)
        axes[1].set_title("By Time Window", fontweight="bold")
        axes[1].tick_params(axis="x", rotation=25)

    # Baseline
    if "baseline" in df.columns:
        bl_order = df.groupby("baseline")["mean_acc"].median().sort_values(ascending=False).index
        bl_data = [df[df["baseline"] == bl]["mean_acc"].values for bl in bl_order]
        bl_plot = axes[2].boxplot(bl_data, labels=bl_order, patch_artist=True, vert=True)
        bl_colors = ["#9b59b6", "#e67e22"]
        for patch, c in zip(bl_plot["boxes"], bl_colors[:len(bl_order)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        axes[2].set_title("By Baseline", fontweight="bold")

    for ax in axes:
        ax.set_ylabel("CV Accuracy")

    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "preprocessing_grid.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Heatmap: bandpass × time window
    if "bandpass" in df.columns and "window" in df.columns:
        pivot = df.pivot_table(values="mean_acc", index="bandpass", columns="window", aggfunc="mean")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        pivot = pivot[pivot.mean(axis=0).sort_values(ascending=False).index]

        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                    linewidths=1, linecolor="white", cbar_kws={"label": "Mean CV Accuracy"})
        plt.title("Bandpass × Time Window — Mean Accuracy", fontsize=13, fontweight="bold")
        plt.xlabel("Time Window")
        plt.ylabel("Bandpass Filter")
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / "preprocessing_heatmap.png", dpi=150, bbox_inches="tight")
        plt.show()


def plot_joint_grid(data: dict, save_dir: Path | None = None):
    """Joint grid — model vs preprocessing heatmap + boxplots."""
    df = extract_grid_results(data, "joint_combo")
    if df.empty:
        return

    # ── Accuracy by Model (boxplot) ──
    if "model_name" in df.columns:
        model_order = df.groupby("model_name")["mean_acc"].median().sort_values(ascending=False).index

        fig, axes = plt.subplots(1, 2, figsize=(20, 7))
        fig.suptitle("Joint Preprocessing × Model Grid Search", fontsize=14, fontweight="bold")

        # By model
        model_data = [df[df["model_name"] == m]["mean_acc"].values for m in model_order]
        bp1 = axes[0].boxplot(model_data, labels=[m[:30] for m in model_order],
                              patch_artist=True, vert=True)
        colors = plt.cm.tab20(np.linspace(0, 1, len(model_order)))
        for patch, c in zip(bp1["boxes"], colors):
            patch.set_facecolor(c)
        axes[0].set_title("Accuracy by Model (across all preprocessing)", fontweight="bold")
        axes[0].set_ylabel("CV Accuracy")
        axes[0].tick_params(axis="x", rotation=45, labelsize=7)

        # By preprocessing
        if "preproc" in df.columns:
            pp_order = df.groupby("preproc")["mean_acc"].median().sort_values(ascending=False).index
            pp_data = [df[df["preproc"] == p]["mean_acc"].values for p in pp_order]
            bp2 = axes[1].boxplot(pp_data, labels=[p[:25] for p in pp_order],
                                  patch_artist=True, vert=True)
            pp_colors = plt.cm.Set2(np.linspace(0, 1, len(pp_order)))
            for patch, c in zip(bp2["boxes"], pp_colors):
                patch.set_facecolor(c)
            axes[1].set_title("Accuracy by Preprocessing (across all models)", fontweight="bold")
            axes[1].set_ylabel("CV Accuracy")
            axes[1].tick_params(axis="x", rotation=25, labelsize=8)

        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / "joint_grid_boxplots.png", dpi=150, bbox_inches="tight")
        plt.show()

    # ── EEGNet vs CSP+ML comparison ──
    if "model_type" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        types = df.groupby("model_type")["mean_acc"]
        type_data = [g.values for _, g in types]
        type_labels = [k for k, _ in types]
        bp = ax.boxplot(type_data, labels=type_labels, patch_artist=True)
        for patch, c in zip(bp["boxes"], ["#3498db", "#e74c3c", "#2ecc71"][:len(type_labels)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_title("EEGNet vs CSP+ML vs ShallowConvNet", fontweight="bold")
        ax.set_ylabel("CV Accuracy")
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / "joint_model_type_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()

    # ── Heatmap: model × preprocessing ──
    if "model_name" in df.columns and "preproc" in df.columns:
        # Take top 15 models by median
        top_models = df.groupby("model_name")["mean_acc"].median().sort_values(ascending=False).head(15).index
        df_top = df[df["model_name"].isin(top_models)]

        pivot = df_top.pivot_table(values="mean_acc", index="model_name", columns="preproc", aggfunc="mean")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        pivot = pivot[pivot.mean(axis=0).sort_values(ascending=False).index]

        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                    linewidths=0.5, linecolor="white", cbar_kws={"label": "CV Accuracy"})
        plt.title("Model × Preprocessing Heatmap (top 15 models)", fontsize=13, fontweight="bold")
        plt.xlabel("Preprocessing")
        plt.ylabel("Model")
        plt.xticks(rotation=25, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / "joint_heatmap.png", dpi=150, bbox_inches="tight")
        plt.show()


def plot_top_results(data: dict, save_dir: Path | None = None):
    """Top N results bar chart across all grid searches."""
    df = extract_grid_results(data, "joint_combo")
    if df.empty:
        # Fallback to eegnet grid
        df = extract_grid_results(data, "eegnet_grid_combo")
    if df.empty:
        return

    top = df.nlargest(15, "mean_acc").copy()

    if "model_name" in top.columns and "preproc" in top.columns:
        top["label"] = top["model_name"].str[:30] + " | " + top["preproc"].str[:25]
    elif "model_name" in top.columns:
        top["label"] = top["model_name"].str[:40]
    else:
        top["label"] = [f"combo_{i}" for i in range(len(top))]

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = []
    for _, row in top.iterrows():
        mt = row.get("model_type", "")
        if mt == "EEGNet":
            colors.append("#3498db")
        elif mt == "ShallowConvNet":
            colors.append("#2ecc71")
        else:
            colors.append("#e74c3c")

    bars = ax.barh(top["label"][::-1], top["mean_acc"][::-1], color=colors[::-1])

    if "std_acc" in top.columns:
        ax.errorbar(top["mean_acc"][::-1], range(len(top)),
                    xerr=top["std_acc"][::-1], fmt="none", color="black", capsize=3)

    for i, (acc, std) in enumerate(zip(top["mean_acc"][::-1],
                                        top.get("std_acc", pd.Series([0]*len(top)))[::-1])):
        ax.text(acc + 0.005, i, f"{acc:.3f}±{std:.3f}", va="center", fontsize=9)

    ax.set_xlabel("CV Accuracy")
    ax.set_title("Top 15 Configurations", fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color="#3498db", label="EEGNet"),
               Patch(color="#2ecc71", label="ShallowConvNet"),
               Patch(color="#e74c3c", label="CSP+ML")]
    ax.legend(handles=handles, loc="lower right")

    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "top15.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_history(data: dict, save_dir: Path | None = None):
    """Plot single run training curves if available."""
    for stage in data["stages"]:
        if stage["stage"] == "single_run":
            metrics = stage.get("metrics", {})
            history = metrics.get("epoch_history", {})
            if not history:
                return

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("Single Run Training Curves", fontsize=14, fontweight="bold")

            axes[0].plot(history["train_loss"], label="Train")
            axes[0].plot(history["val_loss"], label="Val")
            axes[0].set_title("Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].legend()

            axes[1].plot(history["train_acc"], label="Train")
            axes[1].plot(history["val_acc"], label="Val")
            axes[1].set_title("Accuracy")
            axes[1].set_xlabel("Epoch")
            axes[1].legend()

            plt.tight_layout()
            if save_dir:
                plt.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches="tight")
            plt.show()
            return


def plot_summary_table(data: dict, save_dir: Path | None = None):
    """Print a summary of all completed stages."""
    run_name = data.get("run_name", "unknown")
    config = data.get("config", {})
    ch_mode = config.get("channels", {}).get("mode", "?")
    task_mode = config.get("task", {}).get("mode", "?")

    print(f"\n{'='*60}")
    print(f"  Run: {run_name} | {task_mode} | {ch_mode} channels")
    print(f"  Stages completed: {len(data['stages'])}")
    print(f"{'='*60}")

    key_stages = [
        ("single_run", "Single EEGNet", "test_acc"),
        ("cross_validation", "Cross-Validation", "mean_acc"),
        ("eegnet_grid_summary", "EEGNet Grid Best", "best_mean_acc"),
        ("shallow_grid_summary", "ShallowConvNet Grid Best", "best_mean_acc"),
        ("csp_ml_summary", "CSP+ML Best", "best_cv_acc"),
        ("preprocessing_grid_summary", "Best Preprocessing", "best"),
        ("joint_grid_summary", "Joint Grid Best", "best"),
        ("final_retrain", "Final Test", "test_acc"),
    ]

    for stage_name, label, metric_key in key_stages:
        s = extract_summary(data, stage_name)
        if s is None:
            continue

        val = s.get(metric_key, {})
        if isinstance(val, dict):
            acc = val.get("mean_acc", "?")
            name = val.get("model_name", val.get("preproc", ""))
            print(f"  {label:30s} → {acc:.4f}  {name}" if isinstance(acc, float) else f"  {label:30s} → {acc}  {name}")
        else:
            print(f"  {label:30s} → {val:.4f}" if isinstance(val, float) else f"  {label:30s} → {val}")

    print()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def analyze(path: str, save: bool = False):
    data = load_results(path)
    save_dir = Path(path).parent / (Path(path).stem + "_plots") if save else None
    if save_dir:
        save_dir.mkdir(exist_ok=True)
        print(f"Saving plots to {save_dir}/")

    plot_summary_table(data)
    plot_training_history(data, save_dir)
    plot_eegnet_grid(data, save_dir)
    plot_shallow_grid(data, save_dir)
    plot_csp_ml_comparison(data, save_dir)
    plot_preprocessing_grid(data, save_dir)
    plot_joint_grid(data, save_dir)
    plot_top_results(data, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MindStride results")
    parser.add_argument("files", nargs="+", help="JSON result files")
    parser.add_argument("--save", action="store_true", help="Save PNGs instead of showing")
    args = parser.parse_args()

    for f in args.files:
        print(f"\n{'#'*60}")
        print(f"  Analyzing: {f}")
        print(f"{'#'*60}")
        analyze(f, save=args.save)