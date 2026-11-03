"""Stage 1 - input handling: collect image paths and decode them safely."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImageLoadError(Exception):
    """Raised when a file is missing, has an unsupported type or cannot be decoded."""


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def collect_images(path: str | Path, recursive: bool = False) -> tuple[list[Path], list[str]]:
    """Return (image paths, skipped-file messages) for a single file or a folder."""
    path = Path(path)
    skipped: list[str] = []
    if path.is_file():
        if not is_supported(path):
            raise ImageLoadError(f"Unsupported file type '{path.suffix}' for {path.name}")
        return [path], skipped
    if not path.is_dir():
        raise ImageLoadError(f"Input path does not exist: {path}")

    pattern = "**/*" if recursive else "*"
    images = []
    for p in sorted(path.glob(pattern)):
        if not p.is_file():
            continue
        if is_supported(p):
            images.append(p)
        else:
            skipped.append(f"{p.name}: unsupported file type")
    return images, skipped


def decode_image(data: bytes, name: str = "image") -> np.ndarray:
    """Decode raw bytes (e.g. a Streamlit upload) into a BGR image."""
    buf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ImageLoadError(f"{name}: file is corrupt or not a readable image")
    return image


def load_image(path: str | Path) -> np.ndarray:
    """Read an image from disk as a 3-channel BGR array.

    np.fromfile + imdecode is used instead of cv2.imread because imread
    cannot open paths containing non-ASCII characters on Windows.
    """
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"File not found: {path}")
    if not is_supported(path):
        raise ImageLoadError(f"Unsupported file type '{path.suffix}' for {path.name}")
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ImageLoadError(f"{path.name}: cannot be read ({exc})") from exc
    return decode_image(data.tobytes(), path.name)
