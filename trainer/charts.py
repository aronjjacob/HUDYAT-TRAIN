"""Matplotlib figures in the whitepaper style (white, thin lines, accent = train, grey = validation)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ACCENT = "#1F4E99"
GREY = "#8A8A8A"
TEXT = "#1A1A1A"
MUTED = "#6B6B6B"
GRID = "#E6E6E3"
FONT_DIR = Path(__file__).resolve().parent.parent / "static" / "fonts"


def _style() -> None:
    from matplotlib import font_manager
    family = "DejaVu Sans"
    for f in ("Poppins-Regular.ttf", "Poppins-Medium.ttf", "Poppins-SemiBold.ttf"):
        p = FONT_DIR / f
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            family = "Poppins"
    plt.rcParams.update({
        "font.family": family, "font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
        "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlecolor": TEXT,
    })


def training_plot(history: dict, path: Path, title: str = "") -> None:
    _style()
    epochs = np.arange(1, len(history.get("loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), dpi=150)
    for ax, key, name in ((axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")):
        ax.plot(epochs, history.get(key, []), color=ACCENT, lw=1.4, label="Train")
        ax.plot(epochs, history.get("val_" + key, []), color=GREY, lw=1.4, label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name, loc="left")
        ax.legend(loc="best", fontsize=8)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=TEXT)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def confusion_plot(cm: np.ndarray, labels: list[str], path: Path, title: str = "") -> None:
    _style()
    n = len(labels)
    size = max(4.0, 0.42 * n + 2.0)
    fig, ax = plt.subplots(figsize=(size, size * 0.9), dpi=150)
    ax.grid(False)
    rows = cm.sum(axis=1, keepdims=True)
    frac = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows > 0)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if frac[i, j] > 0.55 else TEXT)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
