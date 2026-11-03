# Image-to-Text Pattern Matching Pipeline

A Python pipeline that reads the text in an image (OCR) and finds predefined
keywords, phrases and patterns in it - even when the OCR makes small mistakes.

```
Input image -> Preprocessing -> OCR -> Text cleaning -> Pattern matching -> Output
  (file /      (OpenCV:         (Tesseract  (case, spaces,  (exact, regex,    (annotated image,
   folder)      gray, blur,      or EasyOCR) punctuation,    fuzzy/Levenshtein) JSON, CSV,
                deskew, Otsu/                0/O 1/l 5/S)                       batch summary)
                adaptive)
```

## Try it online

**Live app:** _link added after deployment_

Upload an image (or pick one of the bundled samples), switch preprocessing steps on and off, edit the
keyword list, and download the results as JSON, CSV or an annotated image. The online version uses
Tesseract; EasyOCR is available when you run the project locally.

## Features

- Input: a single image or a folder (PNG, JPG, JPEG, BMP, TIF/TIFF); unreadable or unsupported files are skipped with a message.
- Preprocessing (each step can be switched on/off): resize, grayscale, CLAHE contrast, median/Gaussian blur, projection-profile deskew, Otsu or adaptive thresholding.
- OCR: Tesseract (primary) with word boxes and confidences; EasyOCR (optional) for comparison.
- Decision point: when mean OCR confidence is low, or the OCR boxes cover too little of the ink on the page (missed text), OCR is retried with a stronger preprocessing profile and the pass with the better coverage x confidence score is kept.
- Text cleaning: Unicode punctuation normalisation, noise-token removal, context-aware repair of 0/O, 1/l/I and 5/S confusions, rejoining of split e-mail addresses and URLs.
- Matching: exact (case-insensitive, whole word), regular expressions, and fuzzy (normalised Levenshtein similarity with a configurable threshold). Keywords, phrases and regexes live in `config/keywords.json` (YAML also accepted).
- Output: annotated image (green = exact, blue = regex, orange = fuzzy), JSON per image, one CSV of all matches, batch summary.
- Interfaces: command line (`cli.py`) and a Streamlit web UI (`app.py`).
- Evaluation script with character/word accuracy, precision/recall/F1, a fuzzy-threshold sweep and timing, saved as CSV tables and charts.

## Folder structure

```
image_text_pipeline/
├── app.py                    Streamlit web UI
├── cli.py                    command-line interface
├── run_ui.bat                double-click to start the web UI on Windows
├── requirements.txt          core dependencies (used by the cloud deployment)
├── requirements-easyocr.txt  adds the optional EasyOCR engine
├── packages.txt              system packages for Streamlit Cloud (Tesseract)
├── config/
│   └── keywords.json         keywords, phrases, regex patterns, thresholds
├── ocrmatch/                 the pipeline package
│   ├── io_utils.py           stage 1 - input validation and image loading
│   ├── preprocessing.py      stage 2 - OpenCV preprocessing and deskew
│   ├── ocr_engine.py         stage 3 - Tesseract / EasyOCR wrappers
│   ├── cleaning.py           stage 4 - text normalisation and OCR fixes
│   ├── matching.py           stage 5 - exact, regex and fuzzy matching
│   ├── output.py             stage 6 - annotation, JSON/CSV, batch summary
│   ├── pipeline.py           orchestrator that chains and times the stages
│   └── config.py             config file loader (JSON / YAML)
├── data/
│   ├── samples/              10 synthetic test images (replace or add real photos)
│   ├── ground_truth.json     exact text of each sample image
│   └── edge_cases/           a corrupt file and a non-image, to test error handling
├── scripts/
│   ├── generate_samples.py   re-creates the synthetic dataset
│   └── make_charts.py        draws the evaluation charts
├── evaluation/
│   ├── evaluate.py           accuracy / precision / recall / F1 / timing
│   └── results/              CSV tables, results.json and PNG charts
├── tests/                    pytest unit tests
└── outputs/                  created by the CLI (json/, annotated/, matches.csv, summary.json)
```

## Installation

### 1. Install the Tesseract OCR engine (binary)

`pytesseract` is only a Python wrapper; the Tesseract program must be installed separately.

**Windows**

```bash
winget install --id UB-Mannheim.TesseractOCR -e
```

or download the installer from <https://github.com/UB-Mannheim/tesseract/wiki>.
It installs to `C:\Program Files\Tesseract-OCR\`, which the pipeline finds automatically.
For another location set `TESSERACT_CMD` to the full path of `tesseract.exe`.

**Linux (Debian / Ubuntu)**

```bash
sudo apt update && sudo apt install -y tesseract-ocr
```

**macOS**

```bash
brew install tesseract
```

Check it with `tesseract --version` (version 4 or newer is required for the LSTM engine; this project was tested with 5.4).

### 2. Install the Python packages (Python 3.10+)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

To also compare with EasyOCR, install `requirements-easyocr.txt` instead. It adds PyTorch
(about 1 GB) and downloads its models (about 100 MB) the first time it runs. Without it,
everything works with Tesseract only.

## Usage

### Command line

```bash
python cli.py --input data/samples                        # whole folder
python cli.py --input data/samples/02_invoice.png           # one image
python cli.py --input data/samples --engine easyocr          # second engine
python cli.py --input data/samples --no-preprocess           # compare without preprocessing
python cli.py --input scan.jpg --threshold adaptive --clahe --resize 2
python cli.py --input data/samples --modes exact regex       # no fuzzy matching
python cli.py --input data/samples --fuzzy-threshold 90 --keywords "sale,offer"
python cli.py --help                                         # all options
```

Results go to `outputs/`: `json/<image>.json`, `annotated/<image>_annotated.png`,
`matches.csv` and `summary.json`.

### Web UI

```bash
streamlit run app.py
```

or double-click `run_ui.bat`. Open <http://localhost:8501>, upload an image (or pick a sample),
choose preprocessing options, edit the keyword list, and download the JSON, CSV or annotated image.

### Tests

```bash
python -m pytest -v
```

### Evaluation

```bash
python evaluation/evaluate.py          # add --no-easyocr to skip EasyOCR
python scripts/make_charts.py
```

Tables (CSV), `results.json` and charts (PNG) are written to `evaluation/results/`.

## Configuration

`config/keywords.json`:

```json
{
  "keywords": ["invoice", "total", "confidential"],
  "phrases":  ["due date", "account number"],
  "patterns": {
    "email": {"regex": "[A-Za-z0-9._%+-]+@...", "description": "E-mail address"}
  },
  "settings": {
    "fuzzy_threshold": 80,
    "min_fuzzy_length": 4,
    "min_word_confidence": 0,
    "low_confidence_threshold": 60
  }
}
```

The same structure can be written as YAML (`--config my_keywords.yaml`).

## Using your own images

Drop real photos or scans into `data/samples/` (or any folder) and run the CLI or UI on them.
To include a real image in the accuracy evaluation, add its exact text to `data/ground_truth.json`
under its file name.

## Deploying your own copy

The web UI is ready for [Streamlit Community Cloud](https://share.streamlit.io):

1. Fork or push this repository to GitHub.
2. On share.streamlit.io choose **Create app -> Deploy a public app from GitHub**, select the repository,
   branch `main` and main file `app.py`.
3. `requirements.txt` installs the Python packages and `packages.txt` installs the Tesseract binary
   on the server; no other setup is needed.
