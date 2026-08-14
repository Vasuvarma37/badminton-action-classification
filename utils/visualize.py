"""
utils/visualize.py
──────────────────
Plotting utilities: training curves and confusion matrix.
All figures are saved to results/ and optionally displayed.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for servers/Colab)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, ACTION_CLASSES

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "primary":    "#6C63FF",
    "secondary":  "#FF6584",
    "success":    "#43D787",
    "bg":         "#1A1A2E",
    "panel":      "#16213E",
    "text":       "#E0E0E0",
    "grid":       "#2A2A4A",
}


def _apply_dark_style(ax, fig):
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["text"])
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.grid(color=PALETTE["grid"], linewidth=0.6, linestyle="--", alpha=0.7)


# ─────────────────────────────────────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    save_path: str | None = None,
    show: bool = False,
) -> str:
    """
    Plot loss and accuracy curves side-by-side.

    Returns
    -------
    str : path to the saved figure
    """
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "training_curves.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("BiLSTM Training History", color=PALETTE["text"], fontsize=15, fontweight="bold")

    epochs = range(1, len(train_losses) + 1)

    # Loss
    ax1.plot(epochs, train_losses, color=PALETTE["primary"],   label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses,   color=PALETTE["secondary"], label="Val Loss",   linewidth=2, linestyle="--")
    ax1.set_title("Loss",     fontsize=13)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"])
    _apply_dark_style(ax1, fig)

    # Accuracy
    ax2.plot(epochs, train_accs, color=PALETTE["success"],   label="Train Acc", linewidth=2)
    ax2.plot(epochs, val_accs,   color=PALETTE["secondary"], label="Val Acc",   linewidth=2, linestyle="--")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Accuracy", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"])
    _apply_dark_style(ax2, fig)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] Training curves saved → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] = ACTION_CLASSES,
    save_path: str | None = None,
    show: bool = False,
    title: str = "BiLSTM Confusion Matrix (Test Set)",
) -> str:
    """
    Generate and save a styled confusion matrix heatmap.

    Returns
    -------
    str : path to the saved figure
    """
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalised

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, color=PALETTE["text"], fontsize=14, fontweight="bold")
    fig.patch.set_facecolor(PALETTE["bg"])

    short_names = [n.replace("_", "\n") for n in class_names]

    for ax, data, fmt, subtitle in [
        (axes[0], cm,      "d",    "Raw counts"),
        (axes[1], cm_norm, ".2f",  "Row-normalised"),
    ]:
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="RdPu",
            xticklabels=short_names,
            yticklabels=short_names,
            linewidths=0.5,
            linecolor=PALETTE["bg"],
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(subtitle, color=PALETTE["text"], fontsize=11)
        ax.set_xlabel("Predicted", color=PALETTE["text"])
        ax.set_ylabel("Actual",    color=PALETTE["text"])
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["text"], labelsize=8)
        ax.figure.axes[-1].tick_params(colors=PALETTE["text"])  # colorbar

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] Confusion matrix saved → {save_path}")

    # Also print a text report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("\n── Classification Report ─────────────────────────────────\n")
    print(report)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# Per-class accuracy bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_accuracy(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] = ACTION_CLASSES,
    save_path: str | None = None,
) -> str:
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "per_class_accuracy.png")

    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["success"], "#FFD166"]
    bars = ax.barh(class_names, per_class_acc, color=colors, edgecolor=PALETTE["bg"], height=0.5)

    for bar, val in zip(bars, per_class_acc):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", color=PALETTE["text"], fontsize=11
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Accuracy", color=PALETTE["text"])
    ax.set_title("Per-Class Accuracy — BiLSTM", color=PALETTE["text"], fontsize=13, fontweight="bold")
    _apply_dark_style(ax, fig)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] Per-class accuracy saved → {save_path}")
    return save_path
