"""Stage 2 - image preprocessing with OpenCV.

Every step can be switched on or off so that its effect on OCR accuracy can be
compared. The order is: resize -> grayscale -> contrast -> denoise -> deskew -> threshold.

Besides the processed image, a 3x3 matrix is returned that maps a point in the
ORIGINAL image to the PROCESSED image. The output stage inverts it to draw
OCR bounding boxes on the original picture even after resizing / rotation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np


@dataclass
class PreprocessOptions:
    """Defaults = grayscale + deskew, the profile that scored best in the
    evaluation (word accuracy, keyword F1 and regex F1). Blur and Otsu /
    adaptive thresholding are available as toggles and are used by the
    rescue profile when the first OCR pass looks poor."""
    resize: bool = False
    scale: float = 2.0
    grayscale: bool = True
    contrast: bool = False          # CLAHE local contrast enhancement
    denoise: str = "none"           # "none" | "median" | "gaussian"
    kernel_size: int = 3
    deskew: bool = True
    max_skew: float = 15.0          # degrees searched in each direction
    threshold: str = "none"         # "none" | "otsu" | "adaptive"
    block_size: int = 31            # adaptive threshold neighbourhood (odd)
    adaptive_c: int = 15            # constant subtracted from the local mean

    @classmethod
    def disabled(cls) -> "PreprocessOptions":
        """No preprocessing at all - the raw image goes straight to OCR."""
        return cls(resize=False, grayscale=False, contrast=False, denoise="none",
                   deskew=False, threshold="none")

    @classmethod
    def standard(cls) -> "PreprocessOptions":
        """The classic chain: grayscale, median blur, deskew, Otsu threshold."""
        return cls(denoise="median", threshold="otsu")

    @classmethod
    def rescue(cls) -> "PreprocessOptions":
        """Aggressive profile used when the first OCR pass looks poor."""
        return cls(resize=True, scale=2.0, grayscale=True, contrast=True, denoise="median",
                   deskew=True, threshold="adaptive")

    @property
    def is_disabled(self) -> bool:
        return not (self.resize or self.grayscale or self.contrast or self.deskew
                    or self.denoise != "none" or self.threshold != "none")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreprocessResult:
    image: np.ndarray
    transform: np.ndarray                      # 3x3, original -> processed coordinates
    skew_angle: float = 0.0
    steps: list[tuple[str, np.ndarray]] = field(default_factory=list)


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _odd(k: int) -> int:
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1


def denoise(image: np.ndarray, method: str, ksize: int = 3) -> np.ndarray:
    k = _odd(ksize)
    if method == "median":
        return cv2.medianBlur(image, k)          # removes salt-and-pepper specks, keeps edges
    if method == "gaussian":
        return cv2.GaussianBlur(image, (k, k), 0)  # smooths sensor / JPEG noise
    return image


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """Global threshold chosen automatically to maximise between-class variance."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def adaptive_threshold(gray: np.ndarray, block_size: int = 31, c: int = 15) -> np.ndarray:
    """Per-pixel threshold = Gaussian-weighted local mean minus C (handles uneven light)."""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, _odd(block_size), c)


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask where text pixels are 255, regardless of light/dark background."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cv2.countNonZero(mask) > mask.size / 2:   # light text on dark background
        mask = cv2.bitwise_not(mask)
    return mask


def _rotate_bound(image: np.ndarray, angle: float, border: int | tuple = 255):
    """Rotate about the centre, enlarging the canvas so no corner is cut off."""
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2.0 - cx
    m[1, 2] += nh / 2.0 - cy
    rotated = cv2.warpAffine(image, m, (nw, nh), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return rotated, np.vstack([m, [0, 0, 1]])


def estimate_skew(gray: np.ndarray, max_angle: float = 15.0) -> float:
    """Projection-profile deskew.

    The text mask is rotated through candidate angles; for each one the number
    of ink pixels per row is computed. When the text lines are horizontal the
    profile alternates sharply between full rows (text) and empty rows (gaps),
    so the sum of squared differences between neighbouring rows is largest.
    A coarse 1-degree search is refined with a 0.1-degree search.
    """
    mask = _ink_mask(gray)
    h, w = mask.shape
    scale = min(1.0, 800.0 / max(h, w))
    if scale < 1.0:
        mask = cv2.resize(mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def score(angle: float) -> float:
        rotated, _ = _rotate_bound(mask, angle, border=0)
        profile = rotated.sum(axis=1, dtype=np.float64)
        return float(np.sum(np.diff(profile) ** 2))

    coarse = np.arange(-max_angle, max_angle + 0.001, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 1.0, best + 1.001, 0.1)
    best = max(fine, key=score)
    return round(float(best), 2)


def text_coverage(image: np.ndarray, boxes) -> float:
    """Fraction of ink pixels that lie inside the OCR word boxes.

    An adaptive threshold marks every text-like pixel, even faint text. If OCR
    returned boxes for only a small part of that ink, text was missed - a
    failure that the mean confidence cannot reveal, because words that were
    never read do not lower the average.
    """
    gray = to_gray(image)
    ink = cv2.bitwise_not(adaptive_threshold(gray, 31, 15))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))   # drop single specks
    total = cv2.countNonZero(ink)
    if total == 0:
        return 1.0
    covered = np.zeros_like(ink)
    for x, y, w, h in boxes:
        cv2.rectangle(covered, (x - 3, y - 3), (x + w + 3, y + h + 3), 255, -1)
    return round(cv2.countNonZero(cv2.bitwise_and(ink, covered)) / total, 4)


def preprocess(image: np.ndarray, opts: PreprocessOptions) -> PreprocessResult:
    transform = np.eye(3)
    steps: list[tuple[str, np.ndarray]] = [("Original", image)]
    img = image
    skew = 0.0

    if opts.resize and opts.scale and opts.scale != 1.0:
        img = cv2.resize(img, None, fx=opts.scale, fy=opts.scale, interpolation=cv2.INTER_CUBIC)
        transform = np.diag([opts.scale, opts.scale, 1.0]) @ transform
        steps.append((f"Resized x{opts.scale:g}", img))

    # Thresholding and CLAHE only make sense on a single channel.
    if opts.grayscale or opts.threshold != "none" or opts.contrast:
        img = to_gray(img)
        steps.append(("Grayscale", img))

    if opts.contrast:
        img = enhance_contrast(img)
        steps.append(("Contrast (CLAHE)", img))

    if opts.denoise != "none":
        img = denoise(img, opts.denoise, opts.kernel_size)
        steps.append((f"Denoised ({opts.denoise})", img))

    if opts.deskew:
        skew = estimate_skew(to_gray(img), opts.max_skew)
        if abs(skew) >= 0.3:
            border = 255 if img.ndim == 2 else (255, 255, 255)
            img, rot = _rotate_bound(img, skew, border=border)
            transform = rot @ transform
            steps.append((f"Deskewed ({skew:+.1f} deg)", img))
        else:
            skew = 0.0

    if opts.threshold == "otsu":
        img = otsu_threshold(img)
        steps.append(("Otsu threshold", img))
    elif opts.threshold == "adaptive":
        img = adaptive_threshold(img, opts.block_size, opts.adaptive_c)
        steps.append(("Adaptive threshold", img))

    return PreprocessResult(image=img, transform=transform, skew_angle=skew, steps=steps)
