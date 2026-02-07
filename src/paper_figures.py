"""
Figure generation module for AI4Pain Paper 2.

Generates publication-quality figures as PDF for inclusion in the
Springer Nature LaTeX template.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from paper_analysis import (
    compute_dunn_index,
    compute_silhouette,
    filter_baseline_only,
    load_features,
    make_binary_labels,
    remove_outliers_iqr,
)

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_figures"

# Color scheme for pain classes
CLASS_COLORS = {
    "baseline": "#2196F3",  # blue
    "low": "#FF9800",       # orange
    "high": "#F44336",      # red
}

CLASS_LABELS = {
    "baseline": "No Pain",
    "low": "Low Pain",
    "high": "High Pain",
}


def prepare_ch_plane_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe for C-H plane plotting (exclude rest segments).

    Parameters
    ----------
    df : pd.DataFrame
        Raw feature dataframe.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe without rest segments.
    """
    return filter_baseline_only(df)


def plot_ch_plane_grid(
    signal: str,
    d_values: List[int],
    tau_values: List[int],
    output_path: str,
    split: str = "train",
) -> None:
    """Generate a grid of C-H plane scatter plots across D and tau values.

    Creates a len(d_values) x len(tau_values) grid where each subplot shows
    Permutation Entropy (X) vs Statistical Complexity (Y), colored by class.

    Parameters
    ----------
    signal : str
        Signal type (eda, bvp, resp, spo2).
    d_values : list of int
        Embedding dimensions for rows.
    tau_values : list of int
        Embedding delays for columns.
    output_path : str
        Path to save PDF.
    split : str
        Data split to use.
    """
    n_rows = len(d_values)
    n_cols = len(tau_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, d in enumerate(d_values):
        for j, tau in enumerate(tau_values):
            ax = axes[i, j]
            try:
                df = load_features(split, signal, d=d, tau=tau)
                df = filter_baseline_only(df)
                df = remove_outliers_iqr(df, columns=["pe", "comp"])

                for state in ["baseline", "low", "high"]:
                    mask = df["state"] == state
                    if mask.any():
                        ax.scatter(
                            df.loc[mask, "pe"],
                            df.loc[mask, "comp"],
                            c=CLASS_COLORS[state],
                            label=CLASS_LABELS[state],
                            s=15,
                            alpha=0.7,
                            edgecolors="none",
                        )

                sil = compute_silhouette(
                    df[["pe", "comp"]].values,
                    make_binary_labels(df),
                )
                ax.text(
                    0.02, 0.98,
                    f"SI={sil:.3f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    va="top",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )
            except (FileNotFoundError, ValueError):
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")

            ax.set_title(f"d={d}, tau={tau}", fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel("Permutation Entropy (H)")
            if j == 0:
                ax.set_ylabel("Statistical Complexity (C)")

    # Single legend for entire figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.suptitle(f"C-H Plane: {signal.upper()}", y=1.05, fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def plot_fisher_shannon_grid(
    signal: str,
    d_values: List[int],
    tau_values: List[int],
    output_path: str,
    split: str = "train",
) -> None:
    """Generate a grid of Fisher-Shannon plane scatter plots.

    Similar to C-H plane but uses Entropy (X) vs Fisher Information (Y).

    Parameters
    ----------
    signal : str
        Signal type.
    d_values : list of int
        Embedding dimensions for rows.
    tau_values : list of int
        Embedding delays for columns.
    output_path : str
        Path to save PDF.
    split : str
        Data split to use.
    """
    n_rows = len(d_values)
    n_cols = len(tau_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, d in enumerate(d_values):
        for j, tau in enumerate(tau_values):
            ax = axes[i, j]
            try:
                df = load_features(split, signal, d=d, tau=tau)
                df = filter_baseline_only(df)
                df = remove_outliers_iqr(df, columns=["fisher_shannon", "fisher_info"])

                for state in ["baseline", "low", "high"]:
                    mask = df["state"] == state
                    if mask.any():
                        ax.scatter(
                            df.loc[mask, "fisher_shannon"],
                            df.loc[mask, "fisher_info"],
                            c=CLASS_COLORS[state],
                            label=CLASS_LABELS[state],
                            s=15,
                            alpha=0.7,
                            edgecolors="none",
                        )
            except (FileNotFoundError, ValueError):
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")

            ax.set_title(f"d={d}, tau={tau}", fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel("Permutation Entropy (H)")
            if j == 0:
                ax.set_ylabel("Fisher Information (F)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.suptitle(f"Fisher-Shannon Plane: {signal.upper()}", y=1.05, fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    output_path: str,
    title: str = "",
) -> None:
    """Generate a publication-quality confusion matrix figure.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    labels : list of str
        Class label names.
    output_path : str
        Path to save PDF.
    title : str
        Optional title.
    """
    cm = sklearn_confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

    # Annotate cells with count and percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                    ha="center", va="center", color=color, fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def plot_parameter_sensitivity(
    sweep_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Generate heatmaps of Silhouette and Dunn indices across D x tau.

    Creates one row per signal, two columns (Silhouette, Dunn).

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Output from generate_parameter_sweep_table.
    output_path : str
        Path to save PDF.
    """
    signals = sweep_df["signal"].unique()
    n_signals = len(signals)

    fig, axes = plt.subplots(n_signals, 2, figsize=(8, 2.5 * n_signals))
    if n_signals == 1:
        axes = axes[np.newaxis, :]

    for idx, signal in enumerate(signals):
        sig_data = sweep_df[sweep_df["signal"] == signal]

        for col_idx, (metric, label) in enumerate([("silhouette", "Silhouette Index"), ("dunn", "Dunn Index")]):
            ax = axes[idx, col_idx]
            pivot = sig_data.pivot_table(index="d", columns="tau", values=metric)

            if pivot.empty:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
                continue

            im = ax.imshow(
                pivot.values,
                cmap="YlOrRd",
                aspect="auto",
                interpolation="nearest",
            )

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([str(c) for c in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(r) for r in pivot.index])

            # Annotate
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    color = "white" if val > (pivot.values.max() + pivot.values.min()) / 2 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            color=color, fontsize=8)

            ax.set_xlabel("tau")
            ax.set_ylabel("d")
            ax.set_title(f"{signal.upper()} - {label}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def plot_experiment_pipeline(output_path: str) -> None:
    """Generate a methodology pipeline diagram.

    Shows the flow: Raw Signals -> Bandt-Pompe -> Entropy/Complexity Features
    -> ML Models -> Classification.

    Parameters
    ----------
    output_path : str
        Path to save PDF.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")

    boxes = [
        (0.5, 1.0, "Raw Physiological\nSignals\n(BVP, EDA, RESP, SpO2)"),
        (2.5, 1.0, "Bandt-Pompe\nSymbolization\n(d, tau)"),
        (4.5, 1.0, "Entropy &\nComplexity\nFeatures"),
        (6.5, 1.0, "Global\nZ-Score\nNormalization"),
        (8.5, 1.0, "Optuna-Tuned\nML Models\n(RF, XGB, LGBM)"),
    ]

    box_w, box_h = 1.6, 1.4
    for x, y, text in boxes:
        rect = plt.Rectangle(
            (x - box_w / 2, y - box_h / 2),
            box_w, box_h,
            linewidth=1.5,
            edgecolor="#333333",
            facecolor="#E3F2FD",
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=8,
                fontweight="bold", zorder=3)

    # Arrows
    for i in range(len(boxes) - 1):
        x_start = boxes[i][0] + box_w / 2
        x_end = boxes[i + 1][0] - box_w / 2
        ax.annotate(
            "",
            xy=(x_end, 1.0),
            xytext=(x_start, 1.0),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5),
            zorder=1,
        )

    # Validation box below
    val_x, val_y = 5.5, -0.1
    rect = plt.Rectangle(
        (val_x - 1.2, val_y - 0.35),
        2.4, 0.7,
        linewidth=1.5,
        edgecolor="#333333",
        facecolor="#FFF3E0",
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(val_x, val_y, "LOSO Cross-Validation\n(53 subjects)", ha="center",
            va="center", fontsize=8, fontweight="bold", zorder=3)
    ax.set_ylim(-0.6, 2)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
