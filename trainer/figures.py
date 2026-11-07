"""Figures in one style for the app and the paper: white background, one accent colour plus greys, Poppins,
axis labels with units, no titles (captions belong in the paper).

save(fig, path) writes path.png (dpi) and, for the paper, path.pdf (vector) as well.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ACCENT = "#1F4E99"
ACCENT_LIGHT = "#A9BFE3"
GREY = "#8A8A8A"
GREY_LIGHT = "#CFCFCB"
TEXT = "#1A1A1A"
MUTED = "#6B6B6B"
GRID = "#E6E6E3"
OK, ERR = "#2E7D32", "#B3261E"
FONT_DIR = Path(__file__).resolve().parent.parent / "static" / "fonts"
WIDTH = 6.3                       # inches: one column of an A4 / letter page with margins
FONT_SIZE = 9


def style() -> None:
    from matplotlib import font_manager
    family = "DejaVu Sans"
    for f in ("Poppins-Regular.ttf", "Poppins-Medium.ttf", "Poppins-SemiBold.ttf"):
        p = FONT_DIR / f
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            family = "Poppins"
    plt.rcParams.update({
        "font.family": family, "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1, "legend.fontsize": FONT_SIZE - 1, "axes.edgecolor": GREY_LIGHT,
        "axes.linewidth": 0.8, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
        "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    })


def save(fig, path: Path, dpi: int = 150, pdf: bool = False) -> list[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out = [path.with_suffix(".png")]
    fig.savefig(out[0], dpi=dpi)
    if pdf:
        out.append(path.with_suffix(".pdf"))
        fig.savefig(out[1])
    plt.close(fig)
    return out


def _pct_axis(ax, axis: str = "y") -> None:
    from matplotlib.ticker import PercentFormatter
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(PercentFormatter(1.0, decimals=0))


# ----------------------------------------------------------------------------
def confusion(cm, labels: list[str], path: Path, normalized: bool = False, both: bool = False, **kw) -> list[Path]:
    """Rows = true sign, columns = predicted sign. normalized: share of each true sign (row) in %.
    both: the count with the row % under it (count only when there are more than 12 signs, to stay readable)."""
    style()
    cm = np.asarray(cm, dtype=float)
    rows = cm.sum(axis=1, keepdims=True)
    frac = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    n = len(labels)
    size = min(WIDTH, max(3.2, 0.36 * n + 1.6))
    fig, ax = plt.subplots(figsize=(size, size * 0.92))
    ax.grid(False)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("hudyat", ["#FFFFFF", ACCENT])
    ax.imshow(frac, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(n), labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("Predicted sign")
    ax.set_ylabel("True sign")
    fs = FONT_SIZE - 1 if n <= 12 else FONT_SIZE - 3
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                if both:
                    txt = f"{int(cm[i, j])}\n{frac[i, j] * 100:.0f}%" if n <= 12 else f"{int(cm[i, j])}"
                else:
                    txt = f"{frac[i, j] * 100:.0f}" if normalized else f"{int(cm[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=fs,
                        color="white" if frac[i, j] > 0.55 else TEXT)
    for s in ax.spines.values():
        s.set_visible(False)
    return save(fig, path, **kw)


def per_sign_recall(recall: dict, path: Path, rounds: dict | None = None, line: float = 0.70, **kw) -> list[Path]:
    """Bars = pooled recall per sign; dots = each round (Test 2); dashed line = the 70 % target."""
    style()
    labels = list(recall)
    vals = [recall[k] if recall[k] is not None else 0.0 for k in labels]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    x = np.arange(len(labels))
    colors = [ACCENT if v >= line else GREY_LIGHT for v in vals]
    ax.bar(x, vals, color=colors, width=0.65, zorder=2)
    handles = []
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles += [Patch(color=ACCENT, label=f"Recall ≥ {line:.0%}"), Patch(color=GREY_LIGHT, label=f"Recall < {line:.0%}")]
    if rounds and len(rounds) > 1:
        for name, rec in rounds.items():
            ax.scatter(x, [rec.get(k) if rec.get(k) is not None else np.nan for k in labels], s=10, color=TEXT,
                       alpha=0.55, zorder=3, linewidths=0)
        handles.append(Line2D([], [], marker="o", ls="", ms=3.5, color=TEXT, alpha=0.55, label="Each round"))
    ax.axhline(line, color=TEXT, lw=0.9, ls=(0, (4, 3)), zorder=4)
    handles.append(Line2D([], [], color=TEXT, lw=0.9, ls=(0, (4, 3)), label=f"{line:.0%} target"))
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(handles), borderaxespad=0.2)
    ax.set_xticks(x, labels, rotation=45 if len(labels) > 6 else 0, ha="right" if len(labels) > 6 else "center")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall (%)")
    ax.set_xlabel("Sign")
    _pct_axis(ax)
    ax.grid(axis="x", visible=False)
    return save(fig, path, **kw)


def curves(histories: dict, path: Path, **kw) -> list[Path]:
    """Accuracy and loss per epoch, train vs validation. One round: its lines. Several rounds: thin lines per round
    and the mean +- SD band."""
    from .experiments import mean_curves
    style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5))
    for ax, key, name in ((axes[0], "accuracy", "Accuracy (%)"), (axes[1], "loss", "Loss (cross-entropy)")):
        for k, color, lab in ((key, ACCENT, "Training"), (f"val_{key}", GREY, "Validation")):
            if len(histories) > 1:
                for h in histories.values():
                    if h.get(k):
                        ax.plot(np.arange(1, len(h[k]) + 1), h[k], color=color, lw=0.5, alpha=0.35)
                e, m, sd = mean_curves(histories, k)
                ax.fill_between(e, m - sd, m + sd, color=color, alpha=0.18, lw=0)
                ax.plot(e, m, color=color, lw=1.5, label=f"{lab} (mean ± SD)")
            else:
                h = next(iter(histories.values()), {})
                if h.get(k):
                    ax.plot(np.arange(1, len(h[k]) + 1), h[k], color=color, lw=1.5, label=lab)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        if key == "accuracy":
            _pct_axis(ax)
    axes[0].legend(loc="lower right")
    return save(fig, path, **kw)


def round_accuracy(names: list[str], values: list[float], mean: float, sd: float | None, path: Path,
                   ylabel: str = "Test accuracy (%)", xlabel: str = "Held-out signer", **kw) -> list[Path]:
    style()
    fig, ax = plt.subplots(figsize=(min(WIDTH, 1.6 + 0.6 * len(names)), 2.5))
    x = np.arange(len(names))
    ax.bar(x, values, color=ACCENT, width=0.6, zorder=2)
    ax.axhline(mean, color=TEXT, lw=1.0, zorder=3)
    if sd is not None:
        ax.axhspan(mean - sd, mean + sd, color=GREY_LIGHT, alpha=0.5, lw=0, zorder=1)
    ax.text(len(names) - 0.5, mean + 0.01, f"mean {mean:.1%}" + (f" ± {sd:.1%}" if sd is not None else ""),
            ha="right", va="bottom", fontsize=FONT_SIZE - 1, color=TEXT)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    _pct_axis(ax)
    ax.grid(axis="x", visible=False)
    return save(fig, path, **kw)


def grouped_bars(groups: list[str], series: dict[str, list[float]], path: Path, ylabel: str,
                 errors: dict[str, list[float | None]] | None = None, xlabel: str = "", **kw) -> list[Path]:
    """Grouped bars (e.g. recorded vs mirrored x without / with the canonical mirror). First series = accent."""
    style()
    palette = [ACCENT, GREY, ACCENT_LIGHT, GREY_LIGHT]
    fig, ax = plt.subplots(figsize=(min(WIDTH, 2.2 + 1.2 * len(groups)), 2.6))
    k = len(series)
    w = 0.8 / k
    x = np.arange(len(groups))
    for i, (name, vals) in enumerate(series.items()):
        err = None
        if errors and errors.get(name):
            err = [e if e is not None else 0 for e in errors[name]]
        pos = x - 0.4 + w * (i + 0.5)
        ax.bar(pos, [v if v is not None else 0 for v in vals], width=w * 0.92, color=palette[i % 4], label=name,
               yerr=err, capsize=2, error_kw={"elinewidth": 0.8, "ecolor": TEXT}, zorder=2)
        for j, (p, v) in enumerate(zip(pos, vals)):
            if v is not None:
                top = v + (err[j] if err else 0)
                ax.text(p, top + 0.02, f"{v:.0%}", ha="center", va="bottom", fontsize=FONT_SIZE - 2, color=MUTED)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    _pct_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18 if xlabel else -0.1), ncol=min(4, k))
    return save(fig, path, **kw)


def handedness(h: dict, path: Path, **kw) -> list[Path]:
    """Test clips as recorded vs mirrored (simulated left-handed), without / with the canonical mirror (Auto)."""
    groups = ["As recorded", "Mirrored (simulated left-handed)"]
    series = {"Without canonical mirror": [h["recorded"]["none"]["accuracy"]["mean"], h["mirrored"]["none"]["accuracy"]["mean"]],
              "With canonical mirror (hand: Auto)": [h["recorded"]["auto"]["accuracy"]["mean"],
                                                    h["mirrored"]["auto"]["accuracy"]["mean"]]}
    errors = {"Without canonical mirror": [h["recorded"]["none"]["accuracy"]["sd"], h["mirrored"]["none"]["accuracy"]["sd"]],
              "With canonical mirror (hand: Auto)": [h["recorded"]["auto"]["accuracy"]["sd"],
                                                    h["mirrored"]["auto"]["accuracy"]["sd"]]}
    return grouped_bars(groups, series, path, "Test accuracy (%)", errors, **kw)


def run_figures(d: Path, results: dict) -> list[Path]:
    """The figures the Test page shows, saved in <run>/figures/ (150 dpi)."""
    out = []
    f = Path(d) / "figures"
    labels = results["labels"]
    out += confusion(results["pooled"]["confusion"], labels, f / "confusion_summed")
    out += confusion(results["pooled"]["confusion"], labels, f / "confusion_summed_normalized", normalized=True)
    for r in results["rounds"]:
        if len(results["rounds"]) > 1:
            out += confusion(r["confusion"], labels, f / f"confusion_{r['name']}")
            out += confusion(r["confusion"], labels, f / f"confusion_{r['name']}_normalized", normalized=True)
    rounds = {r["name"]: r["per_sign_recall"] for r in results["rounds"]}
    out += per_sign_recall(results["pooled"]["per_sign_recall"], f / "per_sign_recall", rounds)
    if any(h.get("loss") for h in results["histories"].values()):
        out += curves(results["histories"], f / "curves")
    if len(results["rounds"]) > 1:
        s = results["summary"]["accuracy"]
        out += round_accuracy([r["name"] for r in results["rounds"]], [r["accuracy"] for r in results["rounds"]],
                              s["mean"], s["sd"], f / "round_accuracy")
    out += handedness(results["handedness"], f / "handedness")
    return out


def stacked_bars(categories: list[str], stacks: dict[str, list[float]], path: Path, ylabel: str, xlabel: str = "",
                 **kw) -> list[Path]:
    """Stacked bars (e.g. clips per sign, one stack per signer): accent first, then greys."""
    style()
    palette = [ACCENT, "#5B7FBF", ACCENT_LIGHT, GREY, "#ABABAB", GREY_LIGHT, "#E2E2DE"]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    x = np.arange(len(categories))
    bottom = np.zeros(len(categories))
    for i, (name, vals) in enumerate(stacks.items()):
        ax.bar(x, vals, bottom=bottom, color=palette[i % len(palette)], width=0.7, label=name, zorder=2)
        bottom += np.asarray(vals, float)
    ax.set_xticks(x, categories, rotation=45 if len(categories) > 6 else 0,
                  ha="right" if len(categories) > 6 else "center")
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=min(7, len(stacks)), borderaxespad=0.2)
    return save(fig, path, **kw)


def histogram(values, path: Path, xlabel: str, line: float | None = None, line_label: str = "",
              bins: int = 20, ylabel: str = "Count", percent_x: bool = False, **kw) -> list[Path]:
    style()
    fig, ax = plt.subplots(figsize=(WIDTH * 0.75, 2.5))
    ax.hist(np.asarray(values, float), bins=bins, color=ACCENT, edgecolor="white", linewidth=0.6, zorder=2)
    if line is not None:
        ax.axvline(line, color=TEXT, lw=0.9, ls=(0, (4, 3)), zorder=3, label=line_label or None)
        if line_label:
            ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percent_x:
        _pct_axis(ax, "x")
    ax.grid(axis="x", visible=False)
    return save(fig, path, **kw)


def bars(names: list[str], values: list[float], path: Path, ylabel: str, xlabel: str = "", line: float | None = None,
         line_label: str = "", errors: list | None = None, highlight: list[bool] | None = None, **kw) -> list[Path]:
    """Simple bars in %, optional error bars, reference line and highlighted bars (accent; the rest grey)."""
    style()
    fig, ax = plt.subplots(figsize=(min(WIDTH, 1.8 + 0.55 * len(names)), 2.6))
    x = np.arange(len(names))
    colors = [ACCENT if (highlight is None or h) else GREY for h in (highlight or [True] * len(names))]
    err = [e if e is not None else 0 for e in errors] if errors else None
    ax.bar(x, values, color=colors, width=0.6, zorder=2)
    if errors:                                      # error bars only where there is an SD
        ex = [i for i, e in enumerate(errors) if e is not None]
        if ex:
            ax.errorbar(ex, [values[i] for i in ex], yerr=[errors[i] for i in ex], fmt="none", ecolor=TEXT,
                        elinewidth=0.8, capsize=3, zorder=3)
    for i, v in enumerate(values):
        top = v + (err[i] if err else 0)
        ax.text(i, top + 0.02, f"{v:.0%}", ha="center", va="bottom", fontsize=FONT_SIZE - 2, color=MUTED)
    if line is not None:
        ax.axhline(line, color=TEXT, lw=0.9, ls=(0, (4, 3)), zorder=3, label=line_label or None)
        if line_label:
            ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
    ax.set_xticks(x, names, rotation=45 if len(names) > 6 else 0, ha="right" if len(names) > 6 else "center")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    _pct_axis(ax)
    ax.grid(axis="x", visible=False)
    return save(fig, path, **kw)
