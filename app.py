"""Streamlit web UI for the Image-to-Text Pattern Matching Pipeline.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from ocrmatch import output
from ocrmatch.config import ConfigError, config_from_dict, load_config
from ocrmatch.io_utils import SUPPORTED_EXTENSIONS, ImageLoadError, decode_image
from ocrmatch.matching import MODES
from ocrmatch.ocr_engine import OCRError, easyocr_available
from ocrmatch.pipeline import Pipeline, PipelineOptions
from ocrmatch.preprocessing import PreprocessOptions

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "data" / "samples"
DEFAULT_CONFIG = ROOT / "config" / "keywords.json"
HIGHLIGHT = {"exact": "#CDEFD3", "regex": "#D3E4FB", "fuzzy": "#FFE1BD"}
TYPE_LABEL = {"exact": "Exact", "regex": "Pattern", "fuzzy": "Fuzzy"}

st.set_page_config(page_title="Image-to-Text Pattern Matching", page_icon="🔎", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select, label {
  font-family: 'Public Sans', system-ui, sans-serif;
}
.block-container { padding-top: 2.2rem; max-width: 1320px; }
h1.app-title { font-size: 2.05rem; font-weight: 700; letter-spacing: -0.02em; color: #1B2A3A; margin: 0 0 .25rem 0; }
p.app-sub { color: #5B6B7C; font-size: 1.02rem; margin: 0 0 1.4rem 0; max-width: 70ch; }
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #DCE3EA; }
section[data-testid="stSidebar"] h3 { font-size: .98rem; font-weight: 700; color: #1B2A3A; margin: 1.1rem 0 .2rem 0; }
.stats { display: flex; flex-wrap: wrap; gap: 0; background: #FFFFFF; border: 1px solid #DCE3EA;
         border-radius: 10px; margin: .4rem 0 1.1rem 0; }
.stat { padding: .8rem 1.3rem; border-right: 1px solid #E6EBF0; min-width: 130px; }
.stat:last-child { border-right: none; }
.stat .v { font-size: 1.45rem; font-weight: 700; color: #1B2A3A; line-height: 1.2; }
.stat .k { font-size: .82rem; color: #5B6B7C; }
.swatch { display: inline-block; width: .7rem; height: .7rem; border-radius: 2px; margin-right: .35rem; vertical-align: baseline; }
.page { background: #FFFFFF; border: 1px solid #DCE3EA; border-radius: 6px; padding: 1.8rem 2.1rem;
        font-family: 'Source Serif 4', Georgia, serif; font-size: 1.06rem; line-height: 1.75;
        color: #1F2B37; white-space: pre-wrap; max-height: 640px; overflow-y: auto;
        box-shadow: 0 1px 0 #E6EBF0, 0 8px 24px -18px rgba(27,42,58,.35); }
.page mark { padding: .05em .15em; border-radius: 3px; color: inherit; }
.page mark.exact { background: #CDEFD3; box-shadow: inset 0 -2px 0 #28A03C; }
.page mark.regex { background: #D3E4FB; box-shadow: inset 0 -2px 0 #1470D7; }
.page mark.fuzzy { background: #FFE1BD; box-shadow: inset 0 -2px 0 #F58C14; }
.legend { font-size: .9rem; color: #5B6B7C; margin: .2rem 0 .8rem 0; }
.legend span.item { margin-right: 1.2rem; }
.note { background: #FFF7E8; border: 1px solid #F3DDB0; color: #6B4E16; border-radius: 8px;
        padding: .6rem .9rem; margin: .3rem 0; font-size: .93rem; }
.empty { border: 1.5px dashed #C9D3DD; border-radius: 12px; padding: 2.6rem; text-align: center;
         color: #5B6B7C; background: #FFFFFF; }
.empty b { color: #1B2A3A; }
.stTabs [data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid #DCE3EA; }
.stTabs [data-baseweb="tab"] { font-weight: 600; padding-left: 0; padding-right: 0; }
div[data-testid="stImage"] img { border: 1px solid #DCE3EA; border-radius: 6px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False, max_entries=24)
def run_pipeline(image_bytes: bytes, name: str, options_json: str, config_json: str):
    config = config_from_dict(json.loads(config_json))
    o = json.loads(options_json)
    options = PipelineOptions(preprocess=PreprocessOptions(**o.pop("preprocess")), **o)
    image = decode_image(image_bytes, name)
    return Pipeline(config, options).process_image(image, name)


def highlighted_html(text: str, matches) -> str:
    """Wrap matched spans in <mark>. Overlaps keep the longer match."""
    chosen, taken = [], []
    for m in sorted(matches, key=lambda m: (-(m.end - m.start), m.start)):
        if any(m.start < e and m.end > s for s, e in taken):
            continue
        taken.append((m.start, m.end))
        chosen.append(m)
    chosen.sort(key=lambda m: m.start)
    out, pos = [], 0
    for m in chosen:
        out.append(html.escape(text[pos:m.start]))
        tip = f"{m.keyword} - {TYPE_LABEL[m.match_type].lower()} match, score {m.score:.0f}"
        out.append(f'<mark class="{m.match_type}" title="{html.escape(tip)}">'
                   f"{html.escape(text[m.start:m.end])}</mark>")
        pos = m.end
    out.append(html.escape(text[pos:]))
    return "".join(out)


def legend_html() -> str:
    items = [f'<span class="item"><span class="swatch" style="background:{output.COLORS_HEX[t]}"></span>'
             f"{TYPE_LABEL[t]} match</span>" for t in MODES]
    return f'<div class="legend">{"".join(items)}</div>'


def stats_html(r) -> str:
    counts = {t: sum(m.match_type == t for m in r.matches) for t in MODES}
    cells = [
        (f"{r.word_count}", "words read"),
        (f"{len(r.matches)}", "matches"),
        (" ".join(f'<span class="swatch" style="background:{output.COLORS_HEX[t]}"></span>{counts[t]}'
                  for t in MODES), "exact / pattern / fuzzy"),
        (f"{r.mean_confidence:.1f}%", "mean OCR confidence"),
        (f"{r.text_coverage:.0%}", "text coverage"),
        (f"{r.skew_angle:+.1f}°", "skew corrected"),
        (f"{r.timings['total']:.2f} s", "processing time"),
    ]
    body = "".join(f'<div class="stat"><div class="v">{v}</div><div class="k">{k}</div></div>' for v, k in cells)
    return f'<div class="stats">{body}</div>'


def to_png(image) -> bytes:
    return cv2.imencode(".png", image)[1].tobytes()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Image")
    source = st.radio("Image source", ["Upload an image", "Use a sample image"], label_visibility="collapsed")
    image_bytes, image_name = None, None
    if source == "Upload an image":
        up = st.file_uploader("Upload", type=[e.strip(".") for e in sorted(SUPPORTED_EXTENSIONS)],
                              label_visibility="collapsed")
        if up is not None:
            image_bytes, image_name = up.getvalue(), up.name
    else:
        samples = sorted(p.name for p in SAMPLES.glob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
        pick = st.selectbox("Sample", samples, label_visibility="collapsed")
        if pick:
            image_bytes, image_name = (SAMPLES / pick).read_bytes(), pick

    st.markdown("### Preprocessing")
    use_pre = st.toggle("Preprocess before OCR", value=True)
    pre = PreprocessOptions.disabled()
    if use_pre:
        c1, c2 = st.columns(2)
        grayscale = c1.checkbox("Grayscale", True)
        deskew = c2.checkbox("Deskew", True)
        contrast = c1.checkbox("Contrast (CLAHE)", False)
        resize = c2.checkbox("Upscale ×2", False)
        denoise = st.selectbox("Noise removal", ["none", "median", "gaussian"],
                               format_func=lambda s: {"median": "Median blur", "gaussian": "Gaussian blur",
                                                      "none": "Off"}[s])
        threshold = st.selectbox("Thresholding", ["none", "otsu", "adaptive"],
                                 format_func=lambda s: {"otsu": "Otsu (global)", "adaptive": "Adaptive (local)",
                                                        "none": "Off"}[s])
        kernel = st.slider("Blur kernel size", 1, 7, 3, step=2) if denoise != "none" else 3
        pre = PreprocessOptions(resize=resize, scale=2.0, grayscale=grayscale, contrast=contrast,
                                denoise=denoise, kernel_size=kernel, deskew=deskew, threshold=threshold)

    st.markdown("### OCR engine")
    engines = ["tesseract"] + (["easyocr"] if easyocr_available() else [])
    engine = st.radio("Engine", engines, horizontal=True, label_visibility="collapsed",
                      format_func=lambda e: {"tesseract": "Tesseract", "easyocr": "EasyOCR"}[e])
    if len(engines) == 1:
        st.caption("Install EasyOCR (`pip install easyocr`) to compare a second engine.")
    auto_retry = st.checkbox("Retry with stronger preprocessing when the result looks poor", True)

    st.markdown("### Matching")
    modes = st.multiselect("Modes", list(MODES), default=list(MODES),
                           format_func=lambda m: TYPE_LABEL[m], label_visibility="collapsed")

    st.markdown("### Keywords and patterns")
    cfg_file = st.file_uploader("Load a keywords file (.json / .yaml)", type=["json", "yaml", "yml"])
    try:
        if cfg_file is not None:
            import yaml
            raw = cfg_file.getvalue().decode("utf-8")
            base_cfg = config_from_dict(json.loads(raw) if cfg_file.name.endswith(".json") else yaml.safe_load(raw))
        else:
            base_cfg = load_config(DEFAULT_CONFIG)
    except (ConfigError, ValueError) as exc:
        st.error(f"That keywords file could not be used: {exc}. Using the default file instead.")
        base_cfg = load_config(DEFAULT_CONFIG)

    fuzzy_threshold = st.slider("Fuzzy similarity threshold", 50, 100, int(base_cfg.fuzzy_threshold),
                                help="Lower values tolerate more OCR errors but raise false matches.",
                                disabled="fuzzy" not in modes)
    terms_text = st.text_area("Keywords and phrases (one per line)", "\n".join(base_cfg.terms), height=190)
    pattern_names = [p.name for p in base_cfg.patterns]
    chosen_patterns = st.multiselect("Regex patterns", pattern_names, default=pattern_names)

# ------------------------------------------------------------------ main
st.markdown('<h1 class="app-title">Image-to-Text Pattern Matching</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-sub">Read the text in a document image, then find your keywords, phrases '
            'and patterns such as e-mails, dates and amounts, even when the OCR gets a few letters wrong.</p>',
            unsafe_allow_html=True)

if image_bytes is None:
    st.markdown('<div class="empty"><b>Add an image to begin.</b><br>Upload a PNG, JPG, BMP or TIFF in the '
                'sidebar, or switch to a sample image.</div>', unsafe_allow_html=True)
    st.stop()
if not modes:
    st.warning("Select at least one matching mode in the sidebar.")
    st.stop()

cfg_dict = base_cfg.to_dict()
cfg_dict["keywords"] = [t.strip() for t in terms_text.splitlines() if t.strip()]
cfg_dict["phrases"] = []
cfg_dict["patterns"] = {k: v for k, v in cfg_dict["patterns"].items() if k in chosen_patterns}
cfg_dict["settings"]["fuzzy_threshold"] = fuzzy_threshold
options = {"preprocess": pre.to_dict(), "engine": engine, "modes": list(modes),
           "fuzzy_threshold": float(fuzzy_threshold), "auto_retry": auto_retry}

try:
    with st.spinner(f"Reading text with {engine.capitalize()}…"):
        result = run_pipeline(image_bytes, image_name, json.dumps(options), json.dumps(cfg_dict))
except ImageLoadError as exc:
    st.error(f"{exc}. Try saving the image again as PNG or JPG.")
    st.stop()
except OCRError as exc:
    st.error(str(exc))
    st.stop()

st.markdown(stats_html(result), unsafe_allow_html=True)
for w in result.warnings:
    st.markdown(f'<div class="note">{html.escape(w)}</div>', unsafe_allow_html=True)

tab_matches, tab_text, tab_pre, tab_dl = st.tabs(["Matches", "Extracted text", "Preprocessing", "Download results"])

with tab_matches:
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        show_labels = st.toggle("Show labels on the image", value=True)
        st.image(result.annotated_image(labels=show_labels), channels="BGR", use_container_width=True,
                 caption=f"{image_name} with matches outlined")
    with right:
        if not result.matches:
            st.markdown('<div class="empty"><b>No matches found.</b><br>Add keywords in the sidebar, lower the '
                        'fuzzy threshold, or try different preprocessing.</div>', unsafe_allow_html=True)
        else:
            shown = st.pills("Filter", list(MODES), selection_mode="multi", default=list(MODES),
                             format_func=lambda m: TYPE_LABEL[m], label_visibility="collapsed") or []
            rows = [{"Type": TYPE_LABEL[m.match_type], "Keyword / pattern": m.keyword,
                     "Matched text": m.matched_text.replace("\n", " "), "Similarity": m.score,
                     "OCR conf.": m.ocr_confidence}
                    for m in result.matches if m.match_type in shown]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=520,
                         column_config={
                             "Type": st.column_config.TextColumn(width="small"),
                             "Similarity": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100,
                                                                           width="small"),
                             "OCR conf.": st.column_config.NumberColumn(format="%.0f", width="small"),
                         })

with tab_text:
    view = st.segmented_control("Text version", ["Cleaned text", "Raw OCR output"], default="Cleaned text",
                                label_visibility="collapsed")
    if view == "Raw OCR output":
        st.markdown(f'<div class="page">{html.escape(result.raw_text) or "(no text)"}</div>', unsafe_allow_html=True)
    else:
        st.markdown(legend_html(), unsafe_allow_html=True)
        st.markdown(f'<div class="page">{highlighted_html(result.cleaned_text, result.matches) or "(no text)"}</div>',
                    unsafe_allow_html=True)

with tab_pre:
    a, b = st.columns(2, gap="large")
    a.image(result.original, channels="BGR", use_container_width=True, caption="Before: original image")
    b.image(result.processed, channels="BGR" if result.processed.ndim == 3 else "GRAY",
            use_container_width=True, caption="After: image passed to OCR")
    if len(result.steps) > 2:
        st.markdown("**Each step in order**")
        cols = st.columns(min(4, len(result.steps)))
        for i, (label, img) in enumerate(result.steps):
            cols[i % len(cols)].image(img, channels="BGR" if img.ndim == 3 else "GRAY",
                                      use_container_width=True, caption=f"{i + 1}. {label}")
    st.caption("Time per stage: " + ", ".join(f"{k} {v:.3f} s" for k, v in result.timings.items()))

with tab_dl:
    stem = Path(image_name).stem
    payload = json.dumps(result.to_dict(include_words=True), indent=2, ensure_ascii=False)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download JSON", payload, f"{stem}_results.json", "application/json", use_container_width=True)
    c2.download_button("Download CSV", output.results_to_csv_text([result]), f"{stem}_matches.csv", "text/csv",
                       use_container_width=True)
    c3.download_button("Download annotated image", to_png(result.annotated_image()), f"{stem}_annotated.png",
                       "image/png", use_container_width=True)
    st.markdown("**JSON preview**")
    st.json(result.to_dict(), expanded=1)
