"""
Visualisation utilities: training curves, confusion matrix, and model comparison charts.
All figures are saved to the outputs/visualizations/ directory.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from utils.logger import get_logger
from config import VIZ_DIR

logger = get_logger(__name__)

# Consistent colour palette
_PALETTE = "viridis"
_FIG_DPI = 150


# Training curves
def plot_training_curves(
    trainer_state_history: list[dict],
    output_dir: str | Path = VIZ_DIR,
    experiment_name: str = "experiment",
) -> Path:
    """
    Plot train/eval loss and metric curves from Trainer log history.

    Args:
        trainer_state_history: trainer.state.log_history list.
        output_dir: Directory to save the figure.
        experiment_name: Used in the file name.

    Returns:
        Path to the saved figure.
    """
    history = pd.DataFrame(trainer_state_history)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_hist = history[history["loss"].notna()] if "loss" in history else pd.DataFrame()
    eval_hist = history[history.get("eval_loss", pd.Series()).notna()] if "eval_loss" in history else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves — {experiment_name}", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    if not train_hist.empty and "step" in train_hist:
        ax.plot(train_hist["step"], train_hist["loss"], label="Train Loss", linewidth=1.5)
    if not eval_hist.empty and "eval_loss" in eval_hist:
        ax.plot(eval_hist["step"], eval_hist["eval_loss"], label="Val Loss", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # F1 Macro
    ax = axes[1]
    metric_col = "eval_f1_macro"
    if not eval_hist.empty and metric_col in eval_hist:
        ax.plot(eval_hist["step"], eval_hist[metric_col], label="Val F1 (macro)", color="darkorange", linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("F1 Macro")
        ax.set_title("Macro F1 Score")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No eval metrics found", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    save_path = output_dir / f"{experiment_name}_training_curves.png"
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved to '%s'", save_path)
    return save_path


# Confusion matrix
def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: list[str],
    output_dir: str | Path = VIZ_DIR,
    experiment_name: str = "experiment",
    normalise: bool = True,
) -> Path:
    """
    Plot and save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array (num_classes × num_classes).
        label_names: Class name list.
        output_dir: Directory to save the figure.
        experiment_name: Used in the file name.
        normalise: If True, display row-normalised (recall) matrix.

    Returns:
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True).clip(1)
        cm_display = cm.astype(float) / row_sums
        fmt = ".2f"
        title = f"Normalised Confusion Matrix — {experiment_name}"
    else:
        cm_display = cm
        fmt = "d"
        title = f"Confusion Matrix — {experiment_name}"

    n = len(label_names)
    fig_size = max(8, n * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=_PALETTE,
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    fname = "normalised_" if normalise else ""
    save_path = output_dir / f"{experiment_name}_{fname}confusion_matrix.png"
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to '%s'", save_path)
    return save_path


# Per-class performance bar chart
def plot_per_class_metrics(
    per_class: list[dict],
    output_dir: str | Path = VIZ_DIR,
    experiment_name: str = "experiment",
) -> Path:
    """
    Bar chart of per-class precision, recall, and F1.

    Args:
        per_class: List of dicts from full_evaluation_report().
        output_dir: Directory to save the figure.
        experiment_name: Used in the file name.

    Returns:
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(per_class)
    x = np.arange(len(df))
    width = 0.25
    colours = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 1.2), 5))
    for i, (metric, colour) in enumerate(zip(["precision", "recall", "f1"], colours)):
        ax.bar(x + i * width, df[metric], width, label=metric.capitalize(), color=colour, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(df["class"], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-Class Metrics — {experiment_name}", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    save_path = output_dir / f"{experiment_name}_per_class_metrics.png"
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Per-class metrics chart saved to '%s'", save_path)
    return save_path


# Class distribution
def plot_class_distribution(
    distribution: pd.DataFrame,
    output_dir: str | Path = VIZ_DIR,
) -> Path:
    """
    Bar chart showing class sample counts.

    Args:
        distribution: DataFrame from analyse_class_distribution().
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(distribution["label"].astype(str), distribution["count"], color=sns.color_palette(_PALETTE, len(distribution)))
    ax.set_xlabel("Class")
    ax.set_ylabel("Sample Count")
    ax.set_title("Class Distribution", fontweight="bold")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    save_path = output_dir / "class_distribution.png"
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Class distribution chart saved to '%s'", save_path)
    return save_path


# Model comparison

def plot_model_comparison(
    experiments_df: pd.DataFrame,
    metric: str = "f1_macro",
    output_dir: str | Path = VIZ_DIR,
) -> Path:
    """
    Horizontal bar chart comparing multiple experiment runs.

    Args:
        experiments_df: DataFrame with columns ['experiment_name', metric].
        metric: Metric column to compare.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = experiments_df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.5)))
    colours = sns.color_palette(_PALETTE, len(df))
    bars = ax.barh(df["experiment_name"], df[metric], color=colours)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}", fontweight="bold")
    ax.set_xlim(0, 1.05)
    for bar in bars:
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    save_path = output_dir / f"model_comparison_{metric}.png"
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Model comparison chart saved to '%s'", save_path)
    return save_path
