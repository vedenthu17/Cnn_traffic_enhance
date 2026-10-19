"""Figures (matplotlib, vector PDF).

Colour rules: model / method identity uses a fixed categorical palette that was checked for
colour-vision-deficiency separation, and every series also gets its own marker, so identity never
depends on colour alone. Magnitudes (confusion, Fourier error, Grad-CAM) use one hue from light to
dark. Every figure has its numbers in a CSV next to it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .metrics import ece_from_bins

# categorical slots, in fixed order (never cycled)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "roadlens_blue", ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def style() -> None:
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8, "font.family": "sans-serif",
        "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
        "lines.linewidth": 1.5, "lines.markersize": 4.5, "legend.frameon": False,
        "figure.dpi": 120, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


def series_style(names: list[str]) -> dict[str, dict]:
    """Fixed colour + marker per entity, assigned in order."""
    if len(names) > len(CATEGORICAL):
        raise ValueError(f"at most {len(CATEGORICAL)} series per figure (got {len(names)}); split the figure instead")
    return {n: {"color": CATEGORICAL[i], "marker": MARKERS[i], "markeredgecolor": "white", "markeredgewidth": 0.6}
            for i, n in enumerate(names)}


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------------------
def plot_severity_curves(per_seed: pd.DataFrame, runs: list[str], labels: dict, corruptions: list[str], path: Path) -> None:
    """Accuracy vs. severity, one panel per corruption plus the mean; mean over seeds with a +-1 std band.

    per_seed: columns run, seed, corruption, severity, acc (clean test set = corruption 'clean', severity 0).
    """
    style()
    st = series_style(runs)
    panels = corruptions + ["mean of all corruptions"]
    ncols = 5
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 1.55 * nrows + 0.5), sharex=True, sharey=True, squeeze=False)
    sev = sorted(per_seed.loc[per_seed.corruption != "clean", "severity"].unique())
    for ax, name in zip(axes.ravel(), panels):
        for run in runs:
            d = per_seed[per_seed.run == run]
            clean = 100 * d[d.corruption == "clean"].set_index("seed").acc
            if name in corruptions:
                c = 100 * d[d.corruption == name].pivot(index="seed", columns="severity", values="acc")
            else:
                c = 100 * d[d.corruption != "clean"].groupby(["seed", "severity"]).acc.mean().unstack("severity")
            vals = pd.concat([clean.rename(0), c[sev]], axis=1)
            mean, std = vals.mean(0).values, vals.std(0, ddof=1).fillna(0).values
            xs = [0] + list(sev)
            ax.fill_between(xs, mean - std, mean + std, color=st[run]["color"], alpha=0.12, linewidth=0)
            ax.plot(xs, mean, label=labels.get(run, run), **st[run])
        ax.set_title(name.replace("_", " "), color=INK)
        ax.set_xticks([0] + list(sev))
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("severity (0 = clean)")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (%)")
    handles, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="upper center", ncol=min(len(runs), 3), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.035 * int(np.ceil(len(runs) / 3)) - 0.02))
    _save(fig, path)


def plot_confusion(conf: np.ndarray, class_labels: list[str], title: str, path: Path, annotate_min: float = 0.10,
                   max_annotations: int = 30) -> None:
    """Row-normalised confusion matrix (rows = true class). Off-diagonal cells holding at least
    `annotate_min` of a class are labelled with their percentage (largest first, capped)."""
    style()
    conf = np.asarray(conf, dtype=np.float64)
    k = conf.shape[0]
    frac = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    mesh = ax.pcolormesh(np.arange(k + 1), np.arange(k + 1), frac, cmap=SEQUENTIAL, vmin=0, vmax=1,
                         edgecolors="white", linewidth=0.15)
    ax.set_xlim(0, k)
    ax.set_ylim(k, 0)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks(np.arange(k) + 0.5)
    ax.set_yticks(np.arange(k) + 0.5)
    ax.set_xticklabels(class_labels, rotation=90, fontsize=5.5)
    ax.set_yticklabels(class_labels, fontsize=5.5)
    ax.tick_params(length=0)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    acc = np.trace(conf) / max(conf.sum(), 1)
    ax.set_title(f"{title}  (accuracy {100 * acc:.2f}%)", color=INK)
    off = frac.copy()
    np.fill_diagonal(off, 0)
    cells = np.argwhere(off >= annotate_min)
    cells = sorted(cells, key=lambda rc: -off[rc[0], rc[1]])[:max_annotations]
    for r, c in cells:
        ax.text(c + 0.5, r + 0.5, f"{100 * off[r, c]:.0f}", ha="center", va="center", fontsize=4.5,
                color="white" if off[r, c] > 0.5 else INK)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("fraction of the true class")
    cb.outline.set_visible(False)
    _save(fig, path)


def plot_reliability(bins: dict, conditions: list[str], runs: list[str], labels: dict, path: Path) -> None:
    """Reliability diagrams. bins[(run, condition)] = [15, 3] array (count, sum confidence, n correct).
    Bars: accuracy of each confidence bin. Diagonal: perfect calibration."""
    style()
    fig, axes = plt.subplots(len(conditions), len(runs), figsize=(1.75 * len(runs) + 0.4, 1.8 * len(conditions) + 0.3),
                             sharex=True, sharey=True, squeeze=False)
    for i, cond in enumerate(conditions):
        for j, run in enumerate(runs):
            ax, b = axes[i, j], np.asarray(bins[(run, cond)], dtype=np.float64)
            n_bins = len(b)
            edges = np.linspace(0, 1, n_bins + 1)
            used = b[:, 0] > 0
            acc = np.divide(b[:, 2], b[:, 0], out=np.zeros(n_bins), where=used)
            ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, zorder=1)
            ax.bar(edges[:-1][used] + 0.5 / n_bins, acc[used], width=0.82 / n_bins, color=CATEGORICAL[0], zorder=2)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.text(0.04, 0.95, f"ECE {100 * ece_from_bins(b):.2f}%", transform=ax.transAxes, va="top", fontsize=7,
                    color=INK, zorder=3, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})
            if i == 0:
                ax.set_title(labels.get(run, run), color=INK)
            if i == len(conditions) - 1:
                ax.set_xlabel("confidence")
            if j == 0:
                ax.set_ylabel(f"{cond}\naccuracy")
    fig.tight_layout()
    _save(fig, path)


def plot_fourier(maps: dict, clean_err: dict, labels: dict, eps: float, path: Path) -> None:
    """Fourier heat maps (Yin et al. 2019): error rate under a perturbation along each Fourier basis.
    maps[run] = [h, w] array, zero frequency in the centre."""
    style()
    runs = list(maps)
    fig, axes = plt.subplots(1, len(runs), figsize=(2.3 * len(runs) + 0.6, 2.95), sharey=True, squeeze=False)
    mesh = None
    for ax, run in zip(axes[0], runs):
        m = np.asarray(maps[run])
        h, w = m.shape
        mesh = ax.pcolormesh(np.linspace(-0.5, 0.5, w + 1), np.linspace(-0.5, 0.5, h + 1), m, cmap=SEQUENTIAL,
                             vmin=0, vmax=1, rasterized=False)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_yticks([-0.5, 0, 0.5])
        ax.set_xlabel("horizontal frequency\n(cycles / pixel)")
        ax.set_title(f"{labels.get(run, run)}\nclean error {100 * clean_err[run]:.1f}%, "
                     f"mean {100 * np.nanmean(m):.1f}%", color=INK)
    axes[0, 0].set_ylabel("vertical frequency\n(cycles / pixel)")
    cb = fig.colorbar(mesh, ax=list(axes[0]), fraction=0.025, pad=0.02)
    cb.set_label("error rate")
    cb.outline.set_visible(False)
    fig.suptitle(f"Error under a Fourier-basis perturbation of L2 norm {eps:g} per channel", y=1.04, fontsize=8, color=INK)
    _save(fig, path)


def cam_overlay(image: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Blend a single-hue heat layer (opacity = Grad-CAM value) over the desaturated image."""
    gray = image.mean(axis=2, keepdims=True).repeat(3, axis=2) * 0.85 + 0.1
    hue = np.array(matplotlib.colors.to_rgb(CATEGORICAL[1]))
    a = (0.85 * np.clip(cam, 0, 1))[..., None]
    return np.clip((1 - a) * gray + a * hue, 0, 1)


def plot_gradcam(columns: list[dict], row_labels: list[str], path: Path) -> None:
    """columns: one dict per sample with 'title' and 'cells' = list (one per row) of (image HxWx3 in [0,1], caption)."""
    style()
    n_rows, n_cols = len(row_labels), len(columns)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.12 * n_cols + 1.1, 1.22 * n_rows + 0.2), squeeze=False)
    for j, col in enumerate(columns):
        for i, (img, caption) in enumerate(col["cells"]):
            ax = axes[i, j]
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xlabel(caption, fontsize=5.5, labelpad=1.5, color=INK2)
            if i == 0:
                ax.set_title(col["title"], fontsize=6.5, color=INK)
            if j == 0:
                ax.set_ylabel(row_labels[i], rotation=0, ha="right", va="center", fontsize=6.5, color=INK)
    fig.tight_layout(h_pad=0.25, w_pad=0.2)
    _save(fig, path)


def plot_tta_sweep(summary: pd.DataFrame, sizes: list[int], method_labels: dict, path: Path, title: str = "") -> None:
    """Accuracy vs. test batch size, one panel per model, one line per adaptation method.
    summary: columns model, method, batch_size, acc_mean, acc_std."""
    style()
    models = list(dict.fromkeys(summary.model))
    methods = list(dict.fromkeys(summary.method))
    st = series_style(methods)
    fig, axes = plt.subplots(1, len(models), figsize=(1.9 * len(models) + 0.5, 2.3), sharey=True, squeeze=False)
    xs = np.arange(len(sizes))
    for ax, model in zip(axes[0], models):
        for method in methods:
            d = summary[(summary.model == model) & (summary.method == method)].set_index("batch_size").reindex(sizes)
            if d.acc_mean.isna().all():
                continue
            ax.errorbar(xs, d.acc_mean, yerr=d.acc_std.fillna(0), label=method_labels.get(method, method),
                        capsize=2, elinewidth=0.8, **st[method])
        ax.set_xticks(xs)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("test batch size")
        ax.set_title(model, color=INK)
    axes[0, 0].set_ylabel("mean corruption accuracy (%)")
    handles, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1.0))
    if title:
        fig.suptitle(title, y=1.1, fontsize=8, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, path)


# ----------------------------------------------------------------------------------------
# accuracy heatmap: corruption x severity (one model per figure)
# ----------------------------------------------------------------------------------------
HEATMAP_ORDER = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "motion_blur", "zoom_blur", "snow",
                 "frost", "fog", "brightness", "contrast", "pixelate", "elastic_transform", "jpeg_compression"]


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()


def plot_corruption_heatmaps(long: pd.DataFrame, key: str, out_dir: Path, dataset: str = "GTSRB") -> list[Path]:
    """One heatmap PNG per value of `key` (e.g. "step" or "model"), averaged over any other columns (seeds)."""
    out = []
    for name, g in long.groupby(key, sort=False):
        d = g.groupby(["corruption", "severity"], as_index=False).acc.mean()
        out.append(plot_corruption_heatmap(d, f"{dataset} - {name} accuracy (%)", Path(out_dir) / f"{slug(str(name))}.png"))
    return out


def plot_corruption_heatmap(df: pd.DataFrame, title: str, path: Path, order: list[str] | None = None,
                            dpi: int = 200) -> Path:
    """Accuracy of one model on every corruption (rows) at every severity (columns), annotated in %.

    df: one evaluation table with columns corruption, severity, acc (the clean row, corruption == 'clean',
    gives the subtitle). Viridis from 0 to 1, so figures of different models are directly comparable.
    Written as PNG (dpi) or whatever suffix `path` has.
    """
    order = [c for c in (order or HEATMAP_ORDER) if c in set(df.corruption)]
    corr = df[df.corruption != "clean"]
    grid = corr.pivot_table(index="corruption", columns="severity", values="acc").reindex(order)
    sev = list(grid.columns)
    clean = df[df.corruption == "clean"]
    subtitle = f"clean test accuracy {100 * float(clean.acc.iloc[0]):.1f}%" if len(clean) else ""

    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    mesh = ax.imshow(grid.values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = float(grid.values[i, j])
            ax.text(j, i, f"{100 * v:.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v < 0.62 else "black")
    ax.set_xticks(range(len(sev)), [str(int(s)) for s in sev], fontsize=7)
    ax.set_yticks(range(len(order)), [c.replace("_", " ") for c in order], fontsize=7)
    ax.set_xlabel("severity", fontsize=8)
    ax.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("accuracy", fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.outline.set_visible(False)
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title, fontsize=8.5, color=INK)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
