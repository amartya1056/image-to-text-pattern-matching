"""Gradio web UI for the Image-to-Text Pattern Matching Pipeline.

This is the interface used on Hugging Face Spaces. It drives exactly the same
`ocrmatch` pipeline as the Streamlit app (app.py) and the CLI.

Run locally:  python app_gradio.py   ->  http://localhost:7860
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from ocrmatch import output
from ocrmatch.config import ConfigError, config_from_dict, load_config
from ocrmatch.io_utils import SUPPORTED_EXTENSIONS
from ocrmatch.matching import MODES
from ocrmatch.ocr_engine import OCRError, easyocr_available
from ocrmatch.pipeline import Pipeline, PipelineOptions
from ocrmatch.preprocessing import PreprocessOptions

try:  # Hugging Face ZeroGPU: borrow a GPU only while EasyOCR runs
    import spaces

    gpu_task = spaces.GPU(duration=60)
except Exception:  # running locally, or not on ZeroGPU
    def gpu_task(fn):
        return fn

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "data" / "samples"
BASE_CONFIG = load_config(ROOT / "config" / "keywords.json")
PATTERN_NAMES = [p.name for p in BASE_CONFIG.patterns]
TYPE_LABEL = {"exact": "Exact", "regex": "Pattern", "fuzzy": "Fuzzy"}


def _sample_files() -> list[str]:
    """Sample images for the gallery. Browsers cannot display TIFF, so a PNG copy is used."""
    out = []
    preview_dir = Path(tempfile.gettempdir()) / "ocrmatch_samples"
    preview_dir.mkdir(exist_ok=True)
    for p in sorted(SAMPLES.iterdir()):
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if p.suffix.lower() in {".tif", ".tiff"}:
            png = preview_dir / (p.stem + ".png")
            if not png.exists():
                cv2.imwrite(str(png), cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR))
            out.append(str(png))
        else:
            out.append(str(p))
    return out


SAMPLE_FILES = _sample_files()
ENGINES = ["Tesseract"] + (["EasyOCR"] if easyocr_available() else [])

CSS = """
.gradio-container {max-width: 1320px !important; margin: auto;}
#title h1 {font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0;}
#title p {color: #5B6B7C; font-size: 1.02rem; margin-top: .3rem;}
.stats {display: flex; flex-wrap: wrap; border: 1px solid #DCE3EA; border-radius: 10px; background: #fff;}
.stats div.cell {padding: .7rem 1.1rem; border-right: 1px solid #E6EBF0; min-width: 120px;}
.stats div.cell:last-child {border-right: none;}
.stats .v {font-size: 1.35rem; font-weight: 700; color: #1B2A3A;}
.stats .k {font-size: .8rem; color: #5B6B7C;}
.sw {display: inline-block; width: .65rem; height: .65rem; border-radius: 2px; margin: 0 .25rem 0 .1rem;}
.note {background: #FFF7E8; border: 1px solid #F3DDB0; color: #6B4E16; border-radius: 8px; padding: .5rem .8rem; margin-top: .5rem;}
"""


def stats_html(r) -> str:
    counts = {t: sum(m.match_type == t for m in r.matches) for t in MODES}
    swatches = " ".join(f'<span class="sw" style="background:{output.COLORS_HEX[t]}"></span>{counts[t]}' for t in MODES)
    cells = [(r.word_count, "words read"), (len(r.matches), "matches"), (swatches, "exact / pattern / fuzzy"),
             (f"{r.mean_confidence:.1f}%", "mean OCR confidence"), (f"{r.text_coverage:.0%}", "text coverage"),
             (f"{r.skew_angle:+.1f}°", "skew corrected"), (f"{r.timings['total']:.2f} s", "processing time")]
    body = "".join(f'<div class="cell"><div class="v">{v}</div><div class="k">{k}</div></div>' for v, k in cells)
    notes = "".join(f'<div class="note">{w}</div>' for w in r.warnings)
    return f'<div class="stats">{body}</div>{notes}'


def highlighted(text: str, matches) -> list[tuple[str, str | None]]:
    """Split the text into (segment, label) pairs; overlapping matches keep the longer one."""
    chosen, taken = [], []
    for m in sorted(matches, key=lambda m: (-(m.end - m.start), m.start)):
        if not any(m.start < e and m.end > s for s, e in taken):
            taken.append((m.start, m.end))
            chosen.append(m)
    chosen.sort(key=lambda m: m.start)
    out, pos = [], 0
    for m in chosen:
        if m.start > pos:
            out.append((text[pos:m.start], None))
        out.append((text[m.start:m.end], TYPE_LABEL[m.match_type]))
        pos = m.end
    if pos < len(text):
        out.append((text[pos:], None))
    return out


def build_config(keywords_text, patterns, threshold, config_file):
    base = BASE_CONFIG
    if config_file is not None:
        try:
            import yaml

            raw = Path(config_file).read_text(encoding="utf-8")
            data = json.loads(raw) if str(config_file).endswith(".json") else yaml.safe_load(raw)
            base = config_from_dict(data)
        except (ConfigError, ValueError, OSError) as exc:
            raise gr.Error(f"That keywords file could not be used: {exc}")
    d = base.to_dict()
    d["keywords"] = [t.strip() for t in (keywords_text or "").splitlines() if t.strip()] or base.terms
    d["phrases"] = []
    d["patterns"] = {k: v for k, v in d["patterns"].items() if k in (patterns or [])}
    d["settings"]["fuzzy_threshold"] = float(threshold)
    return config_from_dict(d)


def _run(image, use_pre, grayscale, deskew, contrast, upscale, denoise, threshold_method, engine, auto_retry,
            modes, fuzzy_threshold, keywords_text, patterns, config_file):
    if image is None:
        raise gr.Error("Add an image first: upload one or click a sample below.")
    if not modes:
        raise gr.Error("Select at least one matching mode.")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    pre = (PreprocessOptions(resize=upscale, scale=2.0, grayscale=grayscale, contrast=contrast,
                             denoise={"Off": "none", "Median blur": "median", "Gaussian blur": "gaussian"}[denoise],
                             deskew=deskew,
                             threshold={"Off": "none", "Otsu (global)": "otsu",
                                        "Adaptive (local)": "adaptive"}[threshold_method])
           if use_pre else PreprocessOptions.disabled())
    config = build_config(keywords_text, patterns, fuzzy_threshold, config_file)
    opts = PipelineOptions(preprocess=pre, engine=engine.lower(), modes=tuple(m.lower() for m in modes),
                           fuzzy_threshold=float(fuzzy_threshold), auto_retry=auto_retry)
    try:
        r = Pipeline(config, opts).process_image(bgr, "image.png")
    except OCRError as exc:
        raise gr.Error(str(exc))

    annotated = cv2.cvtColor(r.annotated_image(), cv2.COLOR_BGR2RGB)
    table = pd.DataFrame([{"Type": TYPE_LABEL[m.match_type], "Keyword / pattern": m.keyword,
                           "Matched text": m.matched_text.replace("\n", " "), "Similarity": round(m.score),
                           "OCR conf.": None if m.ocr_confidence is None else round(m.ocr_confidence)}
                          for m in r.matches],
                         columns=["Type", "Keyword / pattern", "Matched text", "Similarity", "OCR conf."])
    steps = [(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB), f"{i + 1}. {label}")
             for i, (label, img) in enumerate(r.steps)]

    out_dir = Path(tempfile.mkdtemp())
    result = r.to_dict(include_words=True)
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = output.write_csv(out_dir / "matches.csv", [r])
    png_path = output.save_image(out_dir / "annotated.png", r.annotated_image())
    preview = r.to_dict()
    return (stats_html(r), annotated, table, highlighted(r.cleaned_text, r.matches), r.raw_text, steps,
            [str(json_path), str(csv_path), str(png_path)], preview)



@gpu_task
def _run_on_gpu(*args):
    """EasyOCR path: on Hugging Face ZeroGPU this borrows a GPU for the call."""
    return _run(*args)


def process(image, use_pre, grayscale, deskew, contrast, upscale, denoise, threshold_method, engine, auto_retry,
            modes, fuzzy_threshold, keywords_text, patterns, config_file):
    """Tesseract runs on the CPU straight away; only EasyOCR requests a GPU."""
    args = (image, use_pre, grayscale, deskew, contrast, upscale, denoise, threshold_method, engine, auto_retry,
            modes, fuzzy_threshold, keywords_text, patterns, config_file)
    return _run_on_gpu(*args) if engine == "EasyOCR" else _run(*args)

with gr.Blocks(title="Image-to-Text Pattern Matching") as demo:
    gr.Markdown("# Image-to-Text Pattern Matching\n"
                "Read the text in a document image, then find your keywords, phrases and patterns such as "
                "e-mails, dates and amounts, even when the OCR gets a few letters wrong.", elem_id="title")
    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=320):
            image = gr.Image(label="Image (PNG, JPG, BMP, TIFF)", type="numpy", height=300)
            gr.Examples(examples=SAMPLE_FILES, inputs=image, label="Or click a sample image", examples_per_page=10)
            run = gr.Button("Find matches", variant="primary")
            with gr.Accordion("Preprocessing", open=True):
                use_pre = gr.Checkbox(value=True, label="Preprocess before OCR")
                with gr.Row():
                    grayscale = gr.Checkbox(value=True, label="Grayscale")
                    deskew = gr.Checkbox(value=True, label="Deskew")
                with gr.Row():
                    contrast = gr.Checkbox(value=False, label="Contrast (CLAHE)")
                    upscale = gr.Checkbox(value=False, label="Upscale x2")
                denoise = gr.Dropdown(["Off", "Median blur", "Gaussian blur"], value="Off", label="Noise removal")
                threshold_method = gr.Dropdown(["Off", "Otsu (global)", "Adaptive (local)"], value="Off",
                                               label="Thresholding")
            with gr.Accordion("OCR engine", open=True):
                engine = gr.Radio(ENGINES, value="Tesseract", label="Engine",
                                  info="EasyOCR is a deep-learning engine (CRAFT + CRNN); slower but strong on stylised text.")
                auto_retry = gr.Checkbox(value=True, label="Retry with stronger preprocessing when the result looks poor")
            with gr.Accordion("Matching", open=True):
                modes = gr.CheckboxGroup(["Exact", "Regex", "Fuzzy"], value=["Exact", "Regex", "Fuzzy"],
                                         label="Matching modes")
                fuzzy_threshold = gr.Slider(50, 100, value=BASE_CONFIG.fuzzy_threshold, step=1,
                                            label="Fuzzy similarity threshold",
                                            info="Lower values tolerate more OCR errors but raise false matches.")
            with gr.Accordion("Keywords and patterns", open=False):
                keywords_text = gr.Textbox("\n".join(BASE_CONFIG.terms), lines=10,
                                           label="Keywords and phrases (one per line)")
                patterns = gr.CheckboxGroup(PATTERN_NAMES, value=PATTERN_NAMES, label="Regex patterns")
                config_file = gr.File(label="Or load a keywords file (.json / .yaml)",
                                      file_types=[".json", ".yaml", ".yml"], type="filepath")
        with gr.Column(scale=7):
            stats = gr.HTML('<div class="note" style="background:#fff;border-color:#DCE3EA;color:#5B6B7C">'
                            "Add an image, or click a sample, then press <b>Find matches</b>.</div>")
            with gr.Tabs():
                with gr.Tab("Matches"):
                    annotated = gr.Image(label="Matches outlined (green: exact, blue: pattern, orange: fuzzy)",
                                         type="numpy", interactive=False)
                    table = gr.Dataframe(label="Matches", interactive=False, wrap=True)
                with gr.Tab("Extracted text"):
                    text_hl = gr.HighlightedText(label="Cleaned text with matches highlighted", show_legend=True,
                                                 combine_adjacent=False,
                                                 color_map={"Exact": "#28A03C", "Pattern": "#1470D7",
                                                            "Fuzzy": "#F58C14"})
                    raw_text = gr.Textbox(label="Raw OCR output", lines=8, interactive=False)
                with gr.Tab("Preprocessing"):
                    steps = gr.Gallery(label="Each preprocessing step, in order", columns=3, height="auto")
                with gr.Tab("Download results"):
                    files = gr.File(label="JSON, CSV and annotated image", file_count="multiple")
                    json_view = gr.JSON(label="JSON preview")

    inputs = [image, use_pre, grayscale, deskew, contrast, upscale, denoise, threshold_method, engine, auto_retry,
              modes, fuzzy_threshold, keywords_text, patterns, config_file]
    outputs = [stats, annotated, table, text_hl, raw_text, steps, files, json_view]
    run.click(process, inputs, outputs)
    image.upload(process, inputs, outputs)

THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Public Sans"), "system-ui", "sans-serif"])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
