"""Draw the evaluation charts from evaluation/results/*.csv.

Usage:  python scripts/make_charts.py [extra_output_dir]
Charts are always written to evaluation/results/charts/; if an extra
directory is given they are copied there too (used for the report).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evaluation" / "results"
OUT = RES / "charts"
OUT.mkdir(parents=True, exist_ok=True)
EXTRA = Path(sys.argv[1]) if len(sys.argv) > 1 else None

# validated categorical palette (fixed order) and text tokens
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1B2A3A", "#5B6B7C", "#E3E8EE"
PROPOSED = "Proposed pipeline (Tesseract)"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#B8C2CC", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False,
})


def style(ax, grid_axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if EXTRA:
        EXTRA.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, EXTRA / name)
    print(f"  wrote {path}")


def short(c):
    return c.replace("Tesseract - ", "Tesseract: ").replace("EasyOCR - ", "EasyOCR: ")


# ------------------------------------------------------------ 1 OCR accuracy per configuration
def ocr_accuracy():
    s = pd.read_csv(RES / "ocr_summary.csv")
    y = np.arange(len(s))[::-1]
    h = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.barh(y + h / 2 + 0.02, s.char_accuracy * 100, h, color=S1, label="Character accuracy")
    ax.barh(y - h / 2 - 0.02, s.word_accuracy * 100, h, color=S2, label="Word accuracy")
    for yi, c, w in zip(y, s.char_accuracy, s.word_accuracy):
        ax.text(c * 100 + 0.6, yi + h / 2 + 0.02, f"{c * 100:.1f}", va="center", fontsize=8.5, color=INK)
        ax.text(w * 100 + 0.6, yi - h / 2 - 0.02, f"{w * 100:.1f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([short(c) for c in s.config])
    for lab in ax.get_yticklabels():
        if lab.get_text().startswith("Proposed"):
            lab.set_fontweight("bold")
            lab.set_color(INK)
    ax.set_xlim(50, 102)
    ax.set_xlabel("Accuracy (%) - mean over 10 images")
    ax.set_title("OCR accuracy by engine and preprocessing")
    ax.legend(loc="lower right", ncols=2, bbox_to_anchor=(1, -0.2))
    style(ax)
    save(fig, "fig_6_1_ocr_accuracy.png")


# ------------------------------------------------------------ 2 accuracy per image category (heatmap)
def category_heatmap():
    d = pd.read_csv(RES / "char_accuracy_by_category.csv", index_col=0) * 100
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", ["#F2F7FD", "#9CC3EE", "#2a78d6", "#123F78"])
    im = ax.imshow(d.values, cmap=cmap, vmin=20, vmax=100, aspect="auto")
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 70 else INK)
    ax.set_xticks(range(d.shape[1]))
    ax.set_xticklabels([short(c).replace(": ", ":\n").replace(" (Tesseract)", "\n(Tesseract)") for c in d.columns],
                       fontsize=8.4)
    ax.set_yticks(range(d.shape[0]))
    ax.set_yticklabels(d.index)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Character accuracy (%)", color=MUTED)
    cb.outline.set_visible(False)
    ax.set_title("Character accuracy per image type")
    save(fig, "fig_6_2_accuracy_by_category.png")


# ------------------------------------------------------------ 3 exact vs fuzzy
def matching_modes():
    m = pd.read_csv(RES / "matching_modes.csv")
    panels = [(PROPOSED, "Proposed pipeline (clean OCR)"), ("Tesseract - full + Otsu", "Tesseract + Otsu (noisier OCR)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    metrics = ["precision", "recall", "f1"]
    x = np.arange(3)
    w = 0.36
    for ax, (cfg, title) in zip(axes, panels):
        sub = m[m.config == cfg].set_index("mode")
        for k, (mode, col) in enumerate([("Exact only", S1), ("Exact + fuzzy", S2)]):
            vals = sub.loc[mode, metrics].values.astype(float) * 100
            bars = ax.bar(x + (k - 0.5) * (w + 0.03), vals, w, color=col, label=mode)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(["Precision", "Recall", "F1 score"])
        ax.set_ylim(80, 102)
        ax.set_title(title, fontsize=10.5)
        style(ax, "y")
    axes[0].set_ylabel("Keyword detection (%)")
    axes[1].legend(loc="lower right")
    save(fig, "fig_6_3_exact_vs_fuzzy.png")


# ------------------------------------------------------------ 4 fuzzy threshold sweep
def threshold_sweep():
    s = pd.read_csv(RES / "fuzzy_threshold_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    p = s[s.config == PROPOSED]
    for col, c, lab in [("precision", S1, "Precision"), ("recall", S2, "Recall"), ("f1", S3, "F1 score")]:
        ax.plot(p.threshold, p[col] * 100, color=c, lw=2, marker="o", ms=5, label=lab)
    ax.set_xlabel("Fuzzy similarity threshold")
    ax.set_ylabel("Keyword detection (%)")
    ax.set_title("Proposed pipeline")
    ax.legend(loc="lower right")
    style(ax, "both")
    ax = axes[1]
    tess = [c for c in s.config.unique() if c.startswith("Tesseract") or c == PROPOSED]
    mean_f1 = s[s.config.isin(tess)].groupby("threshold").f1.mean() * 100
    lo = s[s.config.isin(tess)].groupby("threshold").f1.min() * 100
    hi = s[s.config.isin(tess)].groupby("threshold").f1.max() * 100
    ax.fill_between(mean_f1.index, lo, hi, color=S1, alpha=0.15, lw=0, label="range over Tesseract set-ups")
    ax.plot(mean_f1.index, mean_f1.values, color=S1, lw=2, marker="o", ms=5, label="mean F1")
    best = mean_f1.idxmax()
    ax.annotate(f"best mean F1 at {best}", (best, mean_f1[best]), xytext=(best - 22, mean_f1[best] - 12),
                arrowprops=dict(arrowstyle="-", color=MUTED), fontsize=9, color=INK)
    ax.set_xlabel("Fuzzy similarity threshold")
    ax.set_ylabel("F1 score (%)")
    ax.set_title("All five Tesseract set-ups")
    ax.legend(loc="lower right")
    style(ax, "both")
    save(fig, "fig_6_4_fuzzy_threshold.png")


# ------------------------------------------------------------ 5 processing time
def timing():
    d = pd.read_csv(RES / "ocr_per_image.csv")
    t = d.groupby("config", sort=False)[["time_s", "ocr_time_s"]].mean()
    t["other"] = t.time_s - t.ocr_time_s
    y = np.arange(len(t))[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.barh(y, t.ocr_time_s, 0.55, color=S1, label="OCR")
    ax.barh(y, t.other, 0.55, left=t.ocr_time_s + 0.015, color=S2, label="preprocessing, cleaning, matching")
    for yi, tot in zip(y, t.time_s):
        ax.text(tot + 0.05, yi, f"{tot:.2f} s", va="center", fontsize=8.8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([short(c) for c in t.index])
    ax.set_xlabel("Mean processing time per image (seconds, CPU)")
    ax.set_title("Processing time per image")
    ax.legend(loc="lower right")
    style(ax)
    save(fig, "fig_6_5_processing_time.png")


# ------------------------------------------------------------ 6 regex extraction per pattern
def regex_patterns():
    r = pd.read_csv(RES / "regex_per_pattern.csv").iloc[::-1]
    y = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(y, r.f1 * 100, 0.55, color=S1)
    for yi, (f, tp, fn, fp) in zip(y, r[["f1", "tp", "fn", "fp"]].values):
        ax.text(f * 100 + 1, yi, f"{f * 100:.0f}   (found {tp:.0f}, missed {fn:.0f}, false {fp:.0f})", va="center",
                fontsize=8.6, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(r.pattern)
    ax.set_xlim(0, 135)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("F1 score (%)")
    ax.set_title("Regex extraction per pattern (proposed pipeline)")
    style(ax)
    save(fig, "fig_6_6_regex_patterns.png")


if __name__ == "__main__":
    ocr_accuracy()
    category_heatmap()
    matching_modes()
    threshold_sweep()
    timing()
    regex_patterns()
