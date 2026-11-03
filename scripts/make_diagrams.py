"""Draw the methodology diagrams (block diagram, architecture, flowchart,
sequence diagram) and two algorithm figures computed from the sample data.

Usage:  python scripts/make_diagrams.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocrmatch.io_utils import load_image  # noqa: E402
from ocrmatch.preprocessing import _ink_mask, _rotate_bound, adaptive_threshold, otsu_threshold, to_gray  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, RULE = "#1B2A3A", "#5B6B7C", "#9AA8B6"
NAVY, NAVY_T = "#1D4E89", "#E6EEF7"
CFG_T, CFG = "#FFF4D6", "#B8860B"
EXACT, REGEX, FUZZY = "#28A03C", "#1470D7", "#F58C14"
EXACT_T, REGEX_T, FUZZY_T = "#DDF2E0", "#DCE9FB", "#FFE9CF"
GREY_T = "#F1F4F7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w * 10)
    ax.set_ylim(0, h * 10)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, lines=(), fill=NAVY_T, edge=NAVY, title_color=INK, size=10, lsize=8.6,
        radius=1.2, bold=True, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                fc=fill, ec=edge, lw=1.3))
    if lines:
        # canvas units are 10 per inch, so 1 pt = 10/72 units; centre the whole text block
        pt = 10 / 72
        title_h = size * pt * 1.2
        gap = 1.0
        lines_h = len(lines) * lsize * pt * 1.45
        top = y + h / 2 + (title_h + gap + lines_h) / 2
        ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=size,
                fontweight="bold" if bold else "normal", color=title_color)
        tx = x + w / 2 if align == "center" else x + 1.6
        ax.text(tx, top - title_h - gap, "\n".join(lines), ha=align, va="top",
                fontsize=lsize, color=MUTED, linespacing=1.45)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=size,
                fontweight="bold" if bold else "normal", color=title_color, linespacing=1.3)


def arrow(ax, p1, p2, color=INK, label=None, lw=1.4, style="-|>", rad=0.0, lpos=0.5, loff=(0, 1.2),
          lsize=7.8, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=0, shrinkB=0))
    if label:
        mx = p1[0] + (p2[0] - p1[0]) * lpos + loff[0]
        my = p1[1] + (p2[1] - p1[1]) * lpos + loff[1]
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=lsize, color=MUTED, style="italic",
                bbox=dict(fc="white", ec="none", pad=0.6))


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT / name}")


# ---------------------------------------------------------------- 5.1 block diagram
def block_diagram():
    fig, ax = canvas(15, 4.4)
    stages = [
        ("Input image", ["PNG / JPG / BMP / TIFF", "single file or folder", "type + decode check"]),
        ("Preprocessing", ["grayscale, CLAHE", "median / Gaussian blur", "deskew, Otsu / adaptive"]),
        ("OCR", ["Tesseract (LSTM)", "EasyOCR (CRAFT+CRNN)", "words, boxes, confidence"]),
        ("Text cleaning", ["punctuation, spacing", "0/O, 1/l/I, 5/S repair", "rejoin e-mails / URLs"]),
        ("Pattern matching", ["exact keywords", "regular expressions", "fuzzy (Levenshtein)"]),
        ("Output", ["annotated image", "JSON + CSV", "batch summary"]),
    ]
    w, h, gap, y = 20.5, 21, 4.6, 14
    for i, (t, lines) in enumerate(stages):
        x = 2 + i * (w + gap)
        fill = CFG_T if i == 4 else NAVY_T
        box(ax, x, y, w, h, t, lines, fill=fill, edge=NAVY, size=11, lsize=8.8)
        ax.text(x + w / 2, y + h + 1.6, f"Stage {i + 1}", ha="center", fontsize=8.5, color=MUTED)
        if i < len(stages) - 1:
            arrow(ax, (x + w + 0.4, y + h / 2), (x + w + gap - 0.4, y + h / 2), color=NAVY, lw=1.8)
    flows = ["BGR image", "clean binary image", "OCR words", "clean text + spans", "matches"]
    for i, f in enumerate(flows):
        x = 2 + (i + 1) * (w + gap) - gap / 2
        ax.text(x, y - 3.8, f, ha="center", fontsize=8, color=MUTED, style="italic")
    ax.text(2 + 4 * (w + gap) + w / 2, y + h + 5.2, "uses keywords.json (keywords,\nphrases, regex patterns)",
            ha="center", fontsize=8, color=CFG)
    ax.set_ylim(6, 44)
    save(fig, "fig_5_1_block_diagram.png")


# ---------------------------------------------------------------- 5.2 architecture
def architecture():
    fig, ax = canvas(15, 11)
    # interfaces
    box(ax, 30, 91, 34, 12, "Command-line interface", ["cli.py: argparse options, batch runs,", "console report"], size=10)
    box(ax, 86, 91, 34, 12, "Streamlit web UI", ["app.py: upload, toggles, keyword editor,", "highlighted text, downloads"], size=10)
    box(ax, 4, 91, 18, 12, "User", fill=GREY_T, edge=RULE, size=11)
    arrow(ax, (22, 97), (30, 97), color=MUTED)
    ax.plot([13, 13, 103], [103, 107, 107], color=MUTED, lw=1.2)
    arrow(ax, (103, 107), (103, 103), color=MUTED)

    # orchestrator
    ax.add_patch(FancyBboxPatch((3, 6), 144, 79, boxstyle="round,pad=0,rounding_size=2", fc="#FBFCFD", ec=RULE,
                                lw=1.2, ls="--"))
    ax.text(5, 82.3, "ocrmatch package", fontsize=9.5, color=MUTED, fontweight="bold")
    box(ax, 30, 68, 90, 12, "Pipeline orchestrator  (pipeline.py)",
        ["chains the six stages, times each one, applies the low-confidence retry, returns PipelineResult"],
        fill=NAVY, edge=NAVY, title_color="white", size=10.5, lsize=8.4)
    ax.texts[-1].set_color("#DCE6F2")
    arrow(ax, (47, 91), (47, 80), color=NAVY, label="image + options", lpos=0.4, loff=(-9, -1))
    arrow(ax, (103, 91), (103, 80), color=NAVY, label="image bytes + options", lpos=0.4, loff=(12, -1))

    # stages
    stages = [
        (4, "io_utils.py", ["collect_images()", "load_image()", "type & decode", "validation"], NAVY_T),
        (28, "preprocessing.py", ["resize, grayscale", "CLAHE, blur", "deskew (projection)", "Otsu / adaptive"], NAVY_T),
        (52, "ocr_engine.py", ["run_ocr()", "assemble_text()", "word spans"], NAVY_T),
        (76, "cleaning.py", ["normalise text", "fix 0/O 1/l 5/S", "drop noise tokens", "rejoin e-mail/URL"], NAVY_T),
        (100, "matching.py", ["find_matches()", "span -> words", "-> boxes, conf."], CFG_T),
        (124, "output.py", ["annotate()", "write_json/csv()", "batch_summary()"], NAVY_T),
    ]
    sw, sh, sy = 21, 20, 40
    for x, t, lines, fill in stages:
        box(ax, x, sy, sw, sh, t, lines, fill=fill, size=9.4, lsize=8)
        arrow(ax, (x + sw / 2, 68), (x + sw / 2, sy + sh), color=RULE, lw=1.1, style="<|-|>")
    for i in range(len(stages) - 1):
        x = stages[i][0] + sw
        arrow(ax, (x + 0.3, sy + 6), (stages[i + 1][0] - 0.3, sy + 6), color=NAVY, lw=1.6)
    labels = ["BGR array", "image + 3x3\ntransform", "OCRWord list", "clean words,\ntext, spans", "Match list"]
    for i, lab in enumerate(labels):
        x = stages[i][0] + sw + 1.5
        ax.text(x, sy - 1.2, lab, ha="center", va="top", fontsize=7.2, color=MUTED, style="italic")

    # engines
    box(ax, 50, 19, 12.5, 12, "Tesseract", ["LSTM, oem 3", "pytesseract"], fill="white", edge=NAVY, size=8.6, lsize=7.4)
    box(ax, 64, 19, 12.5, 12, "EasyOCR", ["CRAFT +", "CRNN (torch)"], fill="white", edge=NAVY, size=8.6, lsize=7.4)
    arrow(ax, (56, 31), (60, sy), color=NAVY, lw=1.1, style="<|-|>")
    arrow(ax, (70, 31), (64, sy), color=NAVY, lw=1.1, style="<|-|>")
    # matching modes
    for i, (t, sub, c, ct) in enumerate([("Exact", "whole word,\ncase-insens.", EXACT, EXACT_T),
                                         ("Regex", "8 patterns\nfrom config", REGEX, REGEX_T),
                                         ("Fuzzy", "Levenshtein\n>= threshold", FUZZY, FUZZY_T)]):
        x = 87 + i * 12.8
        box(ax, x, 19, 12, 12, t, sub.split("\n"), fill=ct, edge=c, size=8.6, lsize=7.2)
        arrow(ax, (x + 6, 31), (110.5, sy), color=c, lw=1.0, style="<|-")
    # outputs
    for i, t in enumerate(["annotated PNG", "result JSON", "matches CSV", "summary JSON"]):
        box(ax, 128, 30 - i * 6, 17, 4.8, t, fill="white", edge=RULE, size=7.8, bold=False, radius=0.8)
    arrow(ax, (134.5, sy), (134.5, 34.8), color=NAVY, lw=1.1)

    # config
    box(ax, 80, 6, 30, 10, "config/keywords.json", ["keywords, phrases, regexes,", "fuzzy & confidence thresholds"],
        fill=CFG_T, edge=CFG, size=9.2, lsize=7.6)
    box(ax, 112, 6, 13, 10, "config.py", ["JSON / YAML", "loader"], fill=CFG_T, edge=CFG, size=8.6, lsize=7.4)
    arrow(ax, (110, 11), (112, 11), color=CFG)
    arrow(ax, (118.5, 16), (113, 19), color=CFG, lw=1.1)
    # evaluation
    box(ax, 5, 8, 40, 22, "Evaluation  (evaluation/evaluate.py)",
        ["runs the Pipeline on the 10 sample images", "and compares with ground_truth.json:",
         "character / word accuracy, precision, recall, F1,", "fuzzy-threshold sweep, timing",
         "-> CSV tables and charts"], fill=GREY_T, edge=RULE, size=9, lsize=7.8)
    save(fig, "fig_5_2_architecture.png")


# ---------------------------------------------------------------- 5.3 flowchart
def flowchart():
    fig, ax = canvas(11, 17.5)
    cx = 55

    def term(y, t):
        ax.add_patch(FancyBboxPatch((cx - 22, y - 3.2), 44, 6.4, boxstyle="round,pad=0,rounding_size=3.2",
                                    fc=NAVY, ec=NAVY))
        ax.text(cx, y, t, ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")

    def proc(x, y, t, w=34, fill=NAVY_T, edge=NAVY, h=6.6):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=fill, ec=edge, lw=1.2))
        ax.text(x, y, t, ha="center", va="center", fontsize=8.6, color=INK, linespacing=1.25)

    def dec(x, y, t, w=30, h=11):
        ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                             closed=True, fc=CFG_T, ec=CFG, lw=1.3))
        ax.text(x, y, t, ha="center", va="center", fontsize=8.2, color=INK, linespacing=1.2)

    def yn(x, y, t):
        ax.text(x, y, t, fontsize=8, color=NAVY, fontweight="bold")

    def down(y1, y2, x=cx):
        arrow(ax, (x, y1), (x, y2), color=INK, lw=1.2)

    y = 168
    term(y, "Start: user uploads image / runs CLI")
    down(y - 3.2, 158.3)
    proc(cx, 155, "Load keywords.json and options\n(preprocessing, engine, modes, threshold)")
    down(151.7, 146.5)
    dec(cx, 141, "File type supported\nand image readable?")
    yn(cx + 16, 142.5, "No")
    arrow(ax, (cx + 15, 141), (88, 141), color=INK)
    proc(97, 141, "Report error,\nskip this file", w=18, fill="#FDECEC", edge="#C0392B")
    yn(cx + 1.5, 133.5, "Yes")
    down(135.5, 130.3)
    proc(cx, 127, "Preprocess: grayscale -> denoise -> deskew -> threshold\n(only the enabled steps)", w=52)
    down(123.7, 119.3)
    proc(cx, 116, "Run OCR (Tesseract or EasyOCR)\n-> words, boxes, confidences", w=40)
    down(112.7, 108.5)
    dec(cx, 103, "Mean OCR confidence\n< threshold (60)?")
    yn(cx + 16, 104.5, "Yes")
    arrow(ax, (cx + 15, 103), (80, 103), color=INK)
    dec(91, 103, "Auto-retry\non?", w=22, h=10)
    yn(102.3, 104.5, "No")
    ax.plot([102, 106, 106, 91], [103, 103, 87.5, 87.5], color=INK, lw=1.2)
    ax.text(106.8, 95, "warning\nonly", fontsize=7.4, color=MUTED, va="center", rotation=90)
    yn(92.2, 96.5, "Yes")
    arrow(ax, (91, 98), (91, 95.2), color=INK)
    proc(91, 91.6, "Re-run with rescue profile\n(x2, CLAHE, adaptive);\nkeep the better pass", w=26, h=7.2)
    ax.plot([91, 91, cx], [88, 87.5, 87.5], color=INK, lw=1.2)
    yn(cx + 1.5, 95.5, "No")
    down(97.5, 84.6)
    dec(cx, 79.5, "Any text\ndetected?", w=24, h=10)
    yn(cx - 15.5, 81, "No")
    arrow(ax, (cx - 12, 79.5), (23, 79.5), color=INK)
    proc(14, 79.5, "Warn: no text\nin image", w=18, fill="#FDECEC", edge="#C0392B")
    yn(cx + 1.5, 72.5, "Yes")
    down(74.5, 70.3)
    proc(cx, 67, "Clean text: normalise punctuation and spacing,\nfix 0/O 1/l/I 5/S, rejoin e-mails/URLs", w=48)
    down(63.7, 59.3)
    proc(cx, 56, "Exact keyword search + regex patterns\n(case-insensitive, whole word)", w=44,
         fill=EXACT_T, edge=EXACT)
    down(52.7, 48.3)
    proc(cx, 45, "Fuzzy: slide n-1, n, n+1 word windows, compute\nLevenshtein similarity with each keyword", w=48,
         fill=FUZZY_T, edge=FUZZY)
    down(41.7, 38.5)
    dec(cx, 33.5, "Similarity >=\nthreshold?", w=24, h=10)
    yn(cx + 13, 35, "No")
    arrow(ax, (cx + 12, 33.5), (82, 33.5), color=INK)
    proc(93, 33.5, "Discard window", w=20, fill=GREY_T, edge=RULE)
    yn(cx + 1.5, 26.5, "Yes")
    down(28.5, 25.7)
    proc(cx, 22.5, "Add match unless it repeats an exact hit; suppress overlaps", w=48, h=6)
    down(19.5, 17.3)
    dec(cx, 12.5, "Matches\nfound?", w=22, h=9.6)
    yn(cx + 12.5, 14, "No")
    arrow(ax, (cx + 11, 12.5), (80, 12.5), color=INK)
    proc(92, 12.5, "Warn: no match\nfound", w=20, fill="#FDECEC", edge="#C0392B")
    yn(cx + 1.5, 5.2, "Yes")
    # outputs (all three paths end here)
    proc(cx, -1, "Write annotated image, JSON and CSV; show results in UI / console", w=58, h=6)
    down(7.7, 2)
    ax.plot([92, 92], [9.2, -1], color=INK, lw=1.2)
    arrow(ax, (92, -1), (cx + 29, -1), color=INK)
    ax.plot([14, 14], [76.2, -1], color=INK, lw=1.2)
    arrow(ax, (14, -1), (cx - 29, -1), color=INK)
    down(-4, -8.5)
    dec(cx, -14, "More images\nin the batch?", w=24, h=10)
    yn(cx - 15, -12.5, "Yes")
    ax.plot([cx - 12, 4, 4, cx - 15], [-14, -14, 141, 141], color=INK, lw=1.2)
    arrow(ax, (cx - 16, 141), (cx - 15, 141), color=INK)
    yn(cx + 1.5, -21, "No")
    down(-19, -23.3)
    proc(cx, -26.5, "Write matches.csv and summary.json (batch summary)", w=48, h=6)
    down(-29.5, -32.8)
    term(-36, "End")
    ax.set_ylim(-41, 173)
    save(fig, "fig_5_3_flowchart.png")


# ---------------------------------------------------------------- sequence diagram (web UI)
def sequence():
    fig, ax = canvas(14, 8.6)
    actors = ["User", "Streamlit UI\n(app.py)", "Pipeline", "preprocessing\n+ ocr_engine", "cleaning +\nmatching",
              "output"]
    xs = [11.5 + i * 23.4 for i in range(len(actors))]
    top = 79
    ax.set_ylim(2, 88)
    for x, a in zip(xs, actors):
        box(ax, x - 9.5, top, 19, 6.5, a, fill=NAVY_T if a != "User" else GREY_T, size=8.6)
        ax.plot([x, x], [top, 4], color=RULE, lw=1, ls="--")
    msgs = [
        (0, 1, "upload image, choose options, edit keywords", 74),
        (1, 1, "cache lookup (same bytes + options?)", 68),
        (1, 2, "process_image(image, name)", 62),
        (2, 3, "preprocess(image, options)", 56),
        (3, 2, "processed image + transform", 51, True),
        (2, 3, "run_ocr(image, engine)", 46),
        (3, 2, "words, boxes, confidences", 41, True),
        (2, 2, "confidence < 60 ? retry with rescue profile", 36),
        (2, 4, "clean_words(); find_matches(text, spans, config)", 30),
        (4, 2, "matches with boxes and scores", 25, True),
        (2, 1, "PipelineResult (text, matches, timings)", 20, True),
        (1, 5, "annotate(); to_dict(); results_to_csv_text()", 14),
        (5, 1, "annotated PNG, JSON, CSV", 9.5, True),
        (1, 0, "stats, highlighted text, image, download buttons", 5, True),
    ]
    for m in msgs:
        a, b, text, y = m[:4]
        ret = len(m) > 4
        if a == b:
            x = xs[a]
            ax.plot([x, x + 6, x + 6, x + 0.6], [y + 1.4, y + 1.4, y - 1, y - 1], color=NAVY, lw=1.1)
            arrow(ax, (x + 1.2, y - 1), (x + 0.2, y - 1), color=NAVY, lw=1.1)
            ax.text(x + 7, y, text, fontsize=8, color=INK, va="center",
                    bbox=dict(fc="white", ec="none", pad=0.8))
            continue
        arrow(ax, (xs[a], y), (xs[b], y), color=MUTED if ret else NAVY, lw=1.2, ls="--" if ret else "-")
        ax.text((xs[a] + xs[b]) / 2, y + 0.9, text, ha="center", fontsize=8, color=INK,
                bbox=dict(fc="white", ec="none", pad=0.8))
    save(fig, "fig_5_4_sequence_ui.png")


# ---------------------------------------------------------------- algorithm figures
def otsu_figure():
    img = load_image(ROOT / "data" / "samples" / "08_low_contrast_notice.tiff")
    gray = to_gray(img)
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.28, wspace=0.08)
    ax0 = fig.add_subplot(gs[0, :])
    hist = np.bincount(gray.ravel(), minlength=256)
    ax0.bar(np.arange(256), hist, width=1.0, color="#2a78d6")
    ax0.axvline(t, color="#eb6834", lw=2)
    ax0.text(t + 2, hist.max() * 0.9, f"Otsu threshold = {t:.0f}", color=INK, fontsize=10)
    ax0.set_xlim(0, 255)
    ax0.set_xlabel("Gray level")
    ax0.set_ylabel("Pixel count")
    ax0.spines[["top", "right"]].set_visible(False)
    ax0.set_title("Gray-level histogram of 08_low_contrast_notice.tiff", fontsize=11, color=INK, loc="left")
    for i, (im, cap) in enumerate([(gray, "Grayscale input"), (otsu_threshold(gray), "Otsu (one global threshold)"),
                                   (adaptive_threshold(gray, 31, 15), "Adaptive (local mean - C)")]):
        a = fig.add_subplot(gs[1, i])
        a.imshow(im, cmap="gray", vmin=0, vmax=255)
        a.set_title(cap, fontsize=10, color=INK)
        a.axis("off")
    fig.savefig(OUT / "fig_5_5_otsu_vs_adaptive.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT / 'fig_5_5_otsu_vs_adaptive.png'}")


def deskew_figure():
    img = load_image(ROOT / "data" / "samples" / "06_skewed_scan.png")
    gray = to_gray(img)
    mask = _ink_mask(gray)
    h, w = mask.shape
    s = min(1.0, 800.0 / max(h, w))
    mask = cv2.resize(mask, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    angles = np.arange(-15, 15.01, 0.25)
    scores = []
    for a in angles:
        r, _ = _rotate_bound(mask, a, border=0)
        prof = r.sum(axis=1, dtype=np.float64)
        scores.append(np.sum(np.diff(prof) ** 2))
    scores = np.array(scores) / max(scores)
    best = angles[int(np.argmax(scores))]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax = axes[0]
    ax.plot(angles, scores, color="#2a78d6", lw=2)
    ax.plot([best], [1], "o", color="#eb6834", ms=8, mec="white", mew=2)
    ax.annotate(f"best angle {best:+.2f}°", (best, 1), xytext=(best + 3, 0.92), fontsize=10, color=INK)
    ax.set_xlabel("Candidate rotation (degrees)")
    ax.set_ylabel("Profile sharpness (normalised)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Projection-profile score per angle", fontsize=11, color=INK, loc="left")
    for a, ang, cap in [(axes[1], 0, "Row profile at 0° (skewed)"), (axes[2], best, f"Row profile at {best:+.1f}°")]:
        r, _ = _rotate_bound(mask, ang, border=0)
        prof = r.sum(axis=1) / 255
        a.fill_betweenx(np.arange(len(prof)), 0, prof, color="#1baf7a", lw=0)
        a.invert_yaxis()
        a.set_title(cap, fontsize=10, color=INK)
        a.set_xlabel("Ink pixels in row")
        a.set_yticks([])
        a.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_5_6_deskew_profile.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT / 'fig_5_6_deskew_profile.png'}")


if __name__ == "__main__":
    block_diagram()
    architecture()
    flowchart()
    sequence()
    otsu_figure()
    deskew_figure()
