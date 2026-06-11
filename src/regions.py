"""Region-of-interest cropping for a standardized Egyptian ID image."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import ROI_CONFIG, ROI


def crop_roi(image: np.ndarray, roi: ROI, padding: int = 8) -> np.ndarray:
    """Crop one normalized ROI from an image."""
    h, w = image.shape[:2]
    x1 = max(0, int(roi.x1 * w) - padding)
    y1 = max(0, int(roi.y1 * h) - padding)
    x2 = min(w, int(roi.x2 * w) + padding)
    y2 = min(h, int(roi.y2 * h) + padding)
    return image[y1:y2, x1:x2].copy()


def crop_all_regions(image: np.ndarray) -> dict[str, np.ndarray]:
    """Crop all configured fields."""
    return {name: crop_roi(image, roi) for name, roi in ROI_CONFIG.items()}


def draw_roi_debug(image: np.ndarray) -> np.ndarray:
    """Draw configured ROIs on the standardized card image."""
    debug = image.copy()
    h, w = debug.shape[:2]

    for name, roi in ROI_CONFIG.items():
        x1 = int(roi.x1 * w)
        y1 = int(roi.y1 * h)
        x2 = int(roi.x2 * w)
        y2 = int(roi.y2 * h)
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            debug,
            name,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return debug


def save_regions(regions: dict[str, np.ndarray], output_dir: str | Path) -> None:
    """Save ROI crops for debugging/README screenshots."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for name, crop in regions.items():
        cv2.imwrite(str(output / f"{name}_crop.jpg"), crop)
