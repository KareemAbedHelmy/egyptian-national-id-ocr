"""OCR engine abstraction.

Default: EasyOCR because it is simple to install for a first working version.
Optional: PaddleOCR for a stronger production-style path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class OCRTextResult:
    text: str
    confidence: float
    raw: Any


class OCREngine:
    """Thin wrapper around EasyOCR or PaddleOCR."""

    def __init__(self, engine: str = "easyocr", use_gpu: bool = False):
        self.engine = engine.lower().strip()
        self.use_gpu = use_gpu

        if self.engine == "easyocr":
            import easyocr

            self.reader = easyocr.Reader(["ar", "en"], gpu=use_gpu)
        elif self.engine == "paddleocr":
            from paddleocr import PaddleOCR

            # PaddleOCR language names may differ between versions. If this fails,
            # try lang="arabic" or lang="ar" depending on your installed version.
            self.reader = PaddleOCR(use_angle_cls=True, lang="ar", show_log=False)
        else:
            raise ValueError("engine must be either 'easyocr' or 'paddleocr'")

    def recognize(self, image: np.ndarray, digits_only: bool = False) -> OCRTextResult:
        """Recognize text from an image crop."""
        if self.engine == "easyocr":
            return self._recognize_easyocr(image, digits_only=digits_only)
        return self._recognize_paddleocr(image)

    def _recognize_easyocr(self, image: np.ndarray, digits_only: bool = False) -> OCRTextResult:
        allowlist = None
        if digits_only:
            allowlist = "0123456789٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹"

        results = self.reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist=allowlist,
        )

        if not results:
            return OCRTextResult(text="", confidence=0.0, raw=[])

        # EasyOCR result: (bbox, text, confidence)
        # Sort by vertical position, then horizontal position.
        def sort_key(item: Any) -> tuple[float, float]:
            bbox = item[0]
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            return (sum(ys) / len(ys), sum(xs) / len(xs))

        results = sorted(results, key=sort_key)
        texts = [item[1] for item in results]
        confidences = [float(item[2]) for item in results]

        return OCRTextResult(
            text=" ".join(texts).strip(),
            confidence=sum(confidences) / len(confidences),
            raw=results,
        )

    def _recognize_paddleocr(self, image: np.ndarray) -> OCRTextResult:
        results = self.reader.ocr(image, cls=True)

        if not results or not results[0]:
            return OCRTextResult(text="", confidence=0.0, raw=results)

        # Common PaddleOCR format:
        # [[ [bbox], (text, confidence) ], ...]
        lines = results[0]
        texts: list[str] = []
        confidences: list[float] = []

        for line in lines:
            if len(line) < 2:
                continue
            text_conf = line[1]
            if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                texts.append(str(text_conf[0]))
                confidences.append(float(text_conf[1]))

        if not texts:
            return OCRTextResult(text="", confidence=0.0, raw=results)

        return OCRTextResult(
            text=" ".join(texts).strip(),
            confidence=sum(confidences) / len(confidences),
            raw=results,
        )
