"""Capture screenshots of the running Streamlit UI with a headless browser.

Start the UI first (streamlit run app.py), then:
    python scripts/capture_screenshots.py [output_dir] [--url http://localhost:8501]

Uses Playwright with the locally installed Microsoft Edge (channel="msedge"),
so no separate browser download is needed. Install with: pip install playwright
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"


def wait_for_result(page, timeout=90000):
    page.get_by_text("words read").first.wait_for(timeout=timeout)
    page.wait_for_timeout(1800)          # let images and the data grid finish painting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=str(ROOT / "docs" / "screenshots"))
    ap.add_argument("--url", default="http://localhost:8501")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def shot(page, name, full=False):
        page.screenshot(path=str(out / name), full_page=full)
        print(f"  wrote {out / name}")

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 940}, device_scale_factor=1.5)

        # 1. landing page
        page.goto(args.url)
        page.get_by_text("Add an image to begin").wait_for(timeout=60000)
        page.wait_for_timeout(800)
        shot(page, "ui_01_landing.png")

        # 2. upload the invoice -> matches tab
        page.locator("input[type=file]").first.set_input_files(str(SAMPLES / "02_invoice.png"))
        wait_for_result(page)
        shot(page, "ui_02_upload_matches.png")

        # 3. extracted text with highlights
        page.get_by_role("tab", name="Extracted text").click()
        page.wait_for_timeout(1200)
        shot(page, "ui_03_extracted_text.png")

        # 4. downloads + JSON preview
        page.get_by_role("tab", name="Download results").click()
        page.wait_for_timeout(1200)
        shot(page, "ui_04_downloads_json.png")

        # 5. skewed scan -> preprocessing tab (before / after)
        page.locator("input[type=file]").first.set_input_files(str(SAMPLES / "06_skewed_scan.png"))
        page.wait_for_timeout(1500)
        wait_for_result(page)
        page.get_by_role("tab", name="Preprocessing").click()
        page.wait_for_timeout(1500)
        shot(page, "ui_05_preprocessing.png")

        # 6. low-contrast notice -> decision point (automatic retry warning)
        page.locator("input[type=file]").first.set_input_files(str(SAMPLES / "08_low_contrast_notice.tiff"))
        page.wait_for_timeout(1500)
        wait_for_result(page)
        page.get_by_role("tab", name="Matches").click()
        page.wait_for_timeout(1200)
        shot(page, "ui_06_low_contrast_retry.png")

        # 7. feedback form -> fuzzy matches visible in the text view
        page.locator("input[type=file]").first.set_input_files(str(SAMPLES / "10_small_font_form.png"))
        page.wait_for_timeout(1500)
        wait_for_result(page)
        page.get_by_role("tab", name="Extracted text").click()
        page.wait_for_timeout(1200)
        shot(page, "ui_07_decoys_text.png")

        # 8. low-quality scan -> a fuzzy match recovered from an OCR error
        page.locator("input[type=file]").first.set_input_files(str(SAMPLES / "07_low_quality_scan.jpg"))
        page.wait_for_timeout(1500)
        wait_for_result(page)
        page.get_by_role("tab", name="Matches").click()
        page.wait_for_timeout(1500)
        shot(page, "ui_08_fuzzy_match.png")

        browser.close()


if __name__ == "__main__":
    main()
