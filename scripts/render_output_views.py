"""Render real pipeline outputs as images for the report:
  * the CLI console output (terminal view)
  * outputs/matches.csv (spreadsheet view) and a result JSON (code view)
  * a before/after preprocessing figure built from the pipeline's own images

Usage:  python scripts/render_output_views.py <cli_output.txt> <edge_output.txt> <out_dir>
"""

from __future__ import annotations

import csv
import html
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocrmatch.config import load_config  # noqa: E402
from ocrmatch.pipeline import Pipeline, PipelineOptions  # noqa: E402

PROMPT = "PS C:\\Users\\student\\image_text_pipeline&gt; "

TERMINAL_CSS = """
body { margin:0; background:#ffffff; font-family: 'Cascadia Mono','Consolas',monospace; }
.win { width: 1180px; margin: 0; border:1px solid #c9ced6; border-radius:8px; overflow:hidden; }
.bar { background:#f3f3f3; color:#333; font: 13px 'Segoe UI', sans-serif; padding:8px 14px; border-bottom:1px solid #ddd; }
pre { margin:0; background:#0c0c0c; color:#cccccc; font-size:13.5px; line-height:1.45; padding:14px 16px; white-space:pre; }
.p { color:#61d6d6; } .h { color:#f9f1a5; } .ok { color:#16c60c; } .w { color:#e74856; } .dim { color:#8a8a8a; }
.exact { color:#16c60c; } .regex { color:#3b78ff; } .fuzzy { color:#f9a43f; }
"""


def colour_line(line: str) -> str:
    esc = html.escape(line)
    s = line.strip()
    if s.startswith("==="):
        return f'<span class="h">{esc}</span>'
    if s.startswith("BATCH SUMMARY") or s.startswith("by type") or s.startswith("mean OCR"):
        return f'<span class="ok">{esc}</span>'
    if s.startswith("!") or s.startswith("skipped"):
        return f'<span class="w">{esc}</span>'
    if s.startswith("TYPE"):
        return f'<span class="dim">{esc}</span>'
    for t in ("exact", "regex", "fuzzy"):
        if s.startswith(t + " "):
            i = esc.index(t)
            return esc[:i] + f'<span class="{t}">{t}</span>' + esc[i + len(t):]
    return esc


def terminal_html(command: str, lines: list[str], title="Windows PowerShell") -> str:
    body = PROMPT + f'<span class="p">{html.escape(command)}</span>\n' + "\n".join(colour_line(l) for l in lines)
    return f"<html><head><style>{TERMINAL_CSS}</style></head><body><div class='win'><div class='bar'>{title}</div><pre>{body}</pre></div></body></html>"


def csv_html(path: Path, n=22) -> str:
    rows = list(csv.reader(path.open(encoding="utf-8")))
    head, data = rows[0], rows[1:n + 1]
    th = "".join(f"<th>{html.escape(h)}</th>" for h in ["#"] + head)
    trs = "".join("<tr>" + f"<td class='n'>{i + 2}</td>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>"
                  for i, r in enumerate(data))
    css = """body{margin:0;font-family:'Segoe UI',sans-serif;background:#fff}
    .win{border:1px solid #c9ced6;border-radius:8px;overflow:hidden;display:inline-block}
    .bar{background:#217346;color:#fff;font-size:13px;padding:7px 14px}
    table{border-collapse:collapse;font-size:12.5px}
    th{background:#f2f2f2;color:#333;font-weight:600;border:1px solid #d4d4d4;padding:4px 8px;text-align:left}
    td{border:1px solid #e1e1e1;padding:3px 8px;white-space:nowrap;color:#222}
    td.n,th:first-child{background:#f2f2f2;color:#666;text-align:center}"""
    return (f"<html><head><style>{css}</style></head><body><div class='win'><div class='bar'>matches.csv"
            f"  (first {len(data)} of {len(rows) - 1} rows)</div><table><tr>{th}</tr>{trs}</table></div></body></html>")


def json_html(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    words = data.pop("words", [])
    data["extracted_text"] = data["extracted_text"][:160] + " ..."
    data["cleaned_text"] = data["cleaned_text"][:160] + " ..."
    data["matches"] = data["matches"][:3] + [f"... {len(data['matches']) - 3} more matches"]
    data["words"] = words[:1] + [f"... {len(words) - 1} more words"]
    txt = json.dumps(data, indent=2, ensure_ascii=False)
    esc = html.escape(txt)
    import re
    esc = re.sub(r"(&quot;[^&]*?&quot;)(\s*:)", r'<span class="k">\1</span>\2', esc)
    esc = re.sub(r":\s(&quot;.*?&quot;)", r': <span class="s">\1</span>', esc)
    esc = re.sub(r":\s(-?\d+\.?\d*)", r': <span class="n">\1</span>', esc)
    css = """body{margin:0;background:#fff}.win{width:980px;border:1px solid #c9ced6;border-radius:8px;overflow:hidden}
    .bar{background:#f3f3f3;font:13px 'Segoe UI',sans-serif;padding:8px 14px;border-bottom:1px solid #ddd;color:#333}
    pre{margin:0;padding:14px 16px;font:12.5px/1.45 'Cascadia Mono',Consolas,monospace;color:#1f2328;background:#fbfbfb;white-space:pre-wrap}
    .k{color:#0550ae}.s{color:#0a3069}.n{color:#953800}"""
    return (f"<html><head><style>{css}</style></head><body><div class='win'><div class='bar'>outputs/json/{path.name}"
            f"</div><pre>{esc}</pre></div></body></html>")


def render(html_text: str, out: Path, width=1240):
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        page = b.new_page(viewport={"width": width, "height": 400}, device_scale_factor=1.6)
        page.set_content(html_text)
        page.locator(".win").screenshot(path=str(out))
        b.close()
    print(f"  wrote {out}")


def before_after(out: Path):
    cfg = load_config(ROOT / "config" / "keywords.json")
    pipe = Pipeline(cfg, PipelineOptions())
    cases = [("06_skewed_scan.png", "Skewed scan"), ("08_low_contrast_notice.tiff", "Low-contrast notice")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), gridspec_kw={"hspace": 0.32})
    for row, (name, label) in enumerate(cases):
        r = pipe.process_path(ROOT / "data" / "samples" / name)
        axes[row, 0].imshow(cv2.cvtColor(r.original, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title(f"{label}: before (original)", fontsize=11, loc="left")
        proc = r.processed if r.processed.ndim == 2 else cv2.cvtColor(r.processed, cv2.COLOR_BGR2RGB)
        axes[row, 1].imshow(proc, cmap="gray")
        steps = " -> ".join(s for s, _ in r.steps[1:])
        axes[row, 1].set_title(f"After: {steps}", fontsize=9.5, loc="left")
        axes[row, 1].text(0.0, -0.06, f"words read: {r.word_count}, confidence {r.mean_confidence:.1f}, "
                                      f"text coverage {r.text_coverage:.0%}",
                          transform=axes[row, 1].transAxes, fontsize=9, va="top", color="#5B6B7C")
        for a in axes[row]:
            a.axis("off")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    cli_txt, edge_txt, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = cli_txt.read_text(encoding="utf-8").splitlines()
    # first part: header and the first two images
    second_img = [i for i, l in enumerate(lines) if l.startswith("=== ")][2]
    render(terminal_html("python cli.py --input data/samples", lines[:second_img]), out_dir / "cli_01_batch_run.png")
    summary_at = next(i for i, l in enumerate(lines) if l.startswith("=" * 20))
    tail = lines[summary_at - 9:]
    edge = edge_txt.read_text(encoding="utf-8").splitlines()
    joined = ["(... output for the other images ...)", ""] + tail + ["", ""]
    html_text = terminal_html("python cli.py --input data/samples", joined)
    html_text = html_text.replace("</pre>", "\n" + PROMPT + '<span class="p">python cli.py --input data/edge_cases '
                                  '--output outputs/edge_case_run --quiet</span>\n'
                                  + "\n".join(colour_line(l) for l in edge) + "</pre>")
    render(html_text, out_dir / "cli_02_summary_and_errors.png")
    render(csv_html(ROOT / "outputs" / "matches.csv"), out_dir / "output_csv_view.png", width=1400)
    render(json_html(ROOT / "outputs" / "json" / "02_invoice.json"), out_dir / "output_json_view.png", width=1000)
    before_after(out_dir / "preprocessing_before_after.png")


if __name__ == "__main__":
    main()
