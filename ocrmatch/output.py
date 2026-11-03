"""Stage 6 - outputs: annotated image, JSON / CSV results, batch summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

# BGR colours per match type (also used by the web UI, converted to hex)
COLORS_BGR = {
    "exact": (60, 160, 40),     # green
    "regex": (215, 110, 20),    # blue
    "fuzzy": (20, 140, 245),    # orange
}
COLORS_HEX = {k: "#%02x%02x%02x" % (v[2], v[1], v[0]) for k, v in COLORS_BGR.items()}

CSV_FIELDS = ["image", "match_type", "keyword", "matched_text", "score", "ocr_confidence",
              "bbox_x", "bbox_y", "bbox_w", "bbox_h", "processing_time_s"]


def _to_original(box, transform: np.ndarray) -> np.ndarray:
    """Map the 4 corners of a processed-image box back to the original image."""
    x, y, w, h = box
    corners = np.array([[x, y, 1], [x + w, y, 1], [x + w, y + h, 1], [x, y + h, 1]], dtype=float).T
    back = np.linalg.inv(transform) @ corners
    return back[:2].T.round().astype(np.int32)


def annotate(original: np.ndarray, matches, transform: np.ndarray | None = None,
             show_labels: bool = True, legend: bool = True) -> np.ndarray:
    """Draw a coloured box (polygon, if the page was deskewed) around each match."""
    transform = np.eye(3) if transform is None else transform
    canvas = original.copy() if original.ndim == 3 else cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]
    thickness = max(2, round(min(h, w) / 400))
    font_scale = max(0.4, min(h, w) / 1600)

    # one label per anchor point: matches that start at the same place
    # (e.g. "tax invoice" and "invoice") share a single combined label
    labels: dict[tuple[int, int], list] = {}
    for m in matches:
        color = COLORS_BGR.get(m.match_type, (0, 0, 255))
        for box in (m.line_boxes or ([m.bbox] if m.bbox else [])):
            poly = _to_original((box[0] - 3, box[1] - 3, box[2] + 6, box[3] + 6), transform)
            cv2.polylines(canvas, [poly], True, color, thickness, cv2.LINE_AA)
        if show_labels and m.line_boxes:
            poly = _to_original(m.line_boxes[0], transform)
            x, y = int(poly[:, 0].min()), int(poly[:, 1].min()) - 6
            text = m.keyword if m.match_type != "fuzzy" else f"{m.keyword} ({m.score:.0f})"
            key = next((k for k in labels if abs(k[0] - x) < 25 and abs(k[1] - y) < 12), (x, y))
            entry = labels.setdefault(key, [color, []])
            if text not in entry[1]:
                entry[1].append(text)

    for (x, y), (color, texts) in labels.items():
        label = " / ".join(texts)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        y = max(th + 4, y)
        cv2.rectangle(canvas, (x, y - th - 4), (x + tw + 4, y + 2), color, -1)
        cv2.putText(canvas, label, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

    if legend:
        canvas = _add_legend(canvas)
    return canvas


def _add_legend(canvas: np.ndarray) -> np.ndarray:
    h, w = canvas.shape[:2]
    bar_h = max(36, h // 30)
    bar = np.full((bar_h, w, 3), 248, dtype=np.uint8)
    scale = bar_h / 60
    x = 12
    for name, color in COLORS_BGR.items():
        cv2.rectangle(bar, (x, bar_h // 4), (x + bar_h // 2, 3 * bar_h // 4), color, -1)
        x += bar_h // 2 + 8
        label = f"{name} match"
        cv2.putText(bar, label, (x, int(bar_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (60, 60, 60), 1, cv2.LINE_AA)
        x += cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] + 28
    return np.vstack([canvas, bar])


def save_image(path: str | Path, image: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise IOError(f"Could not encode {path}")
    buf.tofile(str(path))
    return path


def match_rows(result) -> list[dict]:
    rows = []
    for m in result.matches:
        x, y, w, h = m.bbox if m.bbox else ("", "", "", "")
        rows.append({
            "image": result.image_name, "match_type": m.match_type, "keyword": m.keyword,
            "matched_text": m.matched_text.replace("\n", " "), "score": m.score,
            "ocr_confidence": m.ocr_confidence, "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
            "processing_time_s": result.timings.get("total", 0),
        })
    return rows


def results_to_csv_text(results) -> str:
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for r in results:
        writer.writerows(match_rows(r))
    return buf.getvalue()


def write_json(path: str | Path, data) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_csv(path: str | Path, results) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(results_to_csv_text(results), encoding="utf-8", newline="")
    return path


def batch_summary(results, failures: list[dict] | None = None) -> dict:
    """Aggregate statistics for a batch run."""
    failures = failures or []
    per_type = {t: 0 for t in COLORS_BGR}
    keyword_counts: dict[str, int] = {}
    for r in results:
        for m in r.matches:
            per_type[m.match_type] = per_type.get(m.match_type, 0) + 1
            keyword_counts[m.keyword] = keyword_counts.get(m.keyword, 0) + 1
    times = [r.timings.get("total", 0) for r in results]
    return {
        "images_processed": len(results),
        "images_failed": len(failures),
        "failures": failures,
        "total_matches": sum(len(r.matches) for r in results),
        "matches_by_type": per_type,
        "matches_by_keyword": dict(sorted(keyword_counts.items(), key=lambda kv: -kv[1])),
        "images_without_matches": [r.image_name for r in results if not r.matches],
        "mean_ocr_confidence": round(float(np.mean([r.mean_confidence for r in results])), 2) if results else 0,
        "total_time_s": round(float(sum(times)), 3),
        "mean_time_per_image_s": round(float(np.mean(times)), 3) if times else 0,
        "per_image": [
            {"image": r.image_name, "words": r.word_count, "matches": len(r.matches),
             "mean_ocr_confidence": r.mean_confidence, "time_s": r.timings.get("total", 0),
             "warnings": r.warnings}
            for r in results
        ],
    }
