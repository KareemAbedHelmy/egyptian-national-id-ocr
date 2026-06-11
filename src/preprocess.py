"""Image preprocessing utilities for Egyptian National ID OCR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import STANDARD_CARD_HEIGHT, STANDARD_CARD_WIDTH


@dataclass
class PreprocessResult:
    original: np.ndarray
    resized: np.ndarray
    warped: np.ndarray
    warped_standard: np.ndarray
    binary: np.ndarray
    card_contour: np.ndarray | None


def read_image(image_path: str | Path) -> np.ndarray:
    """Read an image from disk and validate it."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def resize_for_processing(image: np.ndarray, max_width: int = 1400) -> np.ndarray:
    """Resize large images while preserving aspect ratio."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy()

    scale = max_width / float(w)
    new_h = int(h * scale)
    return cv2.resize(image, (max_width, new_h), interpolation=cv2.INTER_AREA)


def order_points(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype="float32")
    if pts.shape != (4, 2):
        pts = pts.reshape(4, 2).astype("float32")

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply perspective transform to obtain a top-down card view."""
    rect = order_points(points)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(1, int(max(width_a, width_b)))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(1, int(max(height_a, height_b)))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def detect_card_contour(image: np.ndarray) -> np.ndarray | None:
    """Detect the largest 4-point contour that likely represents the ID card."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny thresholds can be tuned for your image quality.
    edges = cv2.Canny(blurred, 50, 160)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    image_area = image.shape[0] * image.shape[1]

    for contour in contours[:10]:
        area = cv2.contourArea(contour)
        if area < 0.15 * image_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)

    # Fallback: use minAreaRect for the largest reasonable contour.
    for contour in contours[:5]:
        area = cv2.contourArea(contour)
        if area < 0.15 * image_area:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return box.astype("float32")

    return None


def standardize_card_size(warped: np.ndarray) -> np.ndarray:
    """Resize warped card to a fixed template size."""
    return cv2.resize(
        warped,
        (STANDARD_CARD_WIDTH, STANDARD_CARD_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def binarize_for_ocr(image: np.ndarray, invert: bool = False) -> np.ndarray:
    """Denoise and binarize an image crop for OCR/debugging."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Improve contrast without over-amplifying the background security pattern.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    denoised = cv2.bilateralFilter(gray, 7, 50, 50)
    binary_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        binary_type,
        31,
        12,
    )

    # Small morphology pass to remove isolated noise.
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary


def deskew_text_image(image: np.ndarray) -> np.ndarray:
    """Correct small in-plane rotation using text foreground pixels.

    This is intended for small OCR crops, not for perspective correction.
    Perspective correction should happen first.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 20:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) > 15:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_card(image: np.ndarray) -> PreprocessResult:
    """Run full card-level preprocessing."""
    original = image.copy()
    resized = resize_for_processing(original)
    contour = detect_card_contour(resized)

    if contour is not None:
        warped = four_point_transform(resized, contour)
    else:
        # Fallback: continue with resized image so the pipeline still runs.
        warped = resized.copy()

    warped_standard = standardize_card_size(warped)
    binary = binarize_for_ocr(warped_standard)

    return PreprocessResult(
        original=original,
        resized=resized,
        warped=warped,
        warped_standard=warped_standard,
        binary=binary,
        card_contour=contour,
    )
