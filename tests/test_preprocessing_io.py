import cv2
import numpy as np
import pytest

from ocrmatch.io_utils import ImageLoadError, collect_images, decode_image, load_image
from ocrmatch.preprocessing import PreprocessOptions, estimate_skew, preprocess, text_coverage


def text_like_image(angle: float = 0.0) -> np.ndarray:
    img = np.full((500, 700), 255, np.uint8)
    for i in range(8):
        cv2.putText(img, "THE QUICK BROWN FOX JUMPS", (40, 70 + i * 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    if angle:
        m = cv2.getRotationMatrix2D((350, 250), angle, 1.0)
        img = cv2.warpAffine(img, m, (700, 500), borderValue=255)
    return img


@pytest.mark.parametrize("angle", [-7.0, -3.0, 4.0, 8.0])
def test_deskew_recovers_rotation(angle):
    estimated = estimate_skew(text_like_image(angle))
    assert estimated == pytest.approx(-angle, abs=0.5)


def test_thresholding_outputs_binary_image():
    for method in ("otsu", "adaptive"):
        out = preprocess(text_like_image(), PreprocessOptions(threshold=method, deskew=False)).image
        assert set(np.unique(out)) <= {0, 255}


def test_disabled_preprocessing_returns_original():
    img = cv2.cvtColor(text_like_image(), cv2.COLOR_GRAY2BGR)
    res = preprocess(img, PreprocessOptions.disabled())
    assert res.image is img
    assert np.allclose(res.transform, np.eye(3))


def test_resize_transform_maps_points():
    res = preprocess(text_like_image(), PreprocessOptions(resize=True, scale=2.0, deskew=False))
    assert res.image.shape == (1000, 1400)
    assert np.allclose(res.transform @ [10, 20, 1], [20, 40, 1])


def test_text_coverage_detects_missed_text():
    img = text_like_image()
    full = [(30, 40 + i * 50, 600, 40) for i in range(8)]       # boxes over every text line
    assert text_coverage(img, full) > 0.95
    assert text_coverage(img, full[:2]) < 0.35                     # OCR "read" only 2 of 8 lines
    assert text_coverage(np.full((50, 50), 255, np.uint8), []) == 1.0   # blank page: nothing missed


def test_corrupt_bytes_raise_clear_error():
    with pytest.raises(ImageLoadError):
        decode_image(b"definitely not an image", "bad.png")


def test_unsupported_and_missing_files(tmp_path):
    (tmp_path / "a.png").write_bytes(cv2.imencode(".png", text_like_image())[1].tobytes())
    (tmp_path / "notes.txt").write_text("x")
    images, skipped = collect_images(tmp_path)
    assert [p.name for p in images] == ["a.png"]
    assert skipped == ["notes.txt: unsupported file type"]
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "missing.png")
    with pytest.raises(ImageLoadError):
        collect_images(tmp_path / "notes.txt")
