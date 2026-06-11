"""End-to-end Egyptian National ID OCR pipeline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import DEFAULT_OUTPUT_DIR
from .ocr_engine import OCREngine
from .postprocess import (
    clean_field,
    extract_14_digit_id,
    is_arabic_text,
    normalize_arabic_text,
    validate_egyptian_national_id,
)
from .preprocess import binarize_for_ocr, preprocess_card, read_image
from .regions import crop_all_regions, draw_roi_debug, save_regions


def _save_debug_images(
    output_dir: str | Path,
    preprocess_result: Any,
    regions: dict[str, np.ndarray],
    binary_regions: dict[str, np.ndarray],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output / "01_original.jpg"), preprocess_result.original)
    cv2.imwrite(str(output / "02_resized.jpg"), preprocess_result.resized)
    cv2.imwrite(str(output / "03_warped_raw.jpg"), preprocess_result.warped)
    cv2.imwrite(str(output / "04_warped_standard.jpg"), preprocess_result.warped_standard)
    cv2.imwrite(str(output / "05_warped_binary.jpg"), preprocess_result.binary)
    cv2.imwrite(str(output / "06_roi_debug.jpg"), draw_roi_debug(preprocess_result.warped_standard))

    save_regions(regions, output)

    for name, crop in binary_regions.items():
        cv2.imwrite(str(output / f"{name}_binary.jpg"), crop)


def process_id_image_array(
    image: np.ndarray,
    engine_name: str = "easyocr",
    use_gpu: bool = False,
    save_debug: bool = False,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Process a loaded image array and return structured OCR output."""
    preprocess_result = preprocess_card(image)
    regions = crop_all_regions(preprocess_result.warped_standard)

    # OCR engines often perform well on the color crop, while binarized crops are
    # useful for numbers and debugging. We keep both.
    binary_regions = {
        name: binarize_for_ocr(crop)
        for name, crop in regions.items()
    }

    if save_debug:
        _save_debug_images(output_dir, preprocess_result, regions, binary_regions)

    ocr = OCREngine(engine=engine_name, use_gpu=use_gpu)

    # Run OCR per field. For the national ID, try both original and binarized crop.
    name_ocr = ocr.recognize(regions["full_name"])
    address_ocr = ocr.recognize(regions["address"])

    id_original_ocr = ocr.recognize(regions["national_id"], digits_only=True)
    id_binary_ocr = ocr.recognize(binary_regions["national_id"], digits_only=True)

    id_candidate_original = extract_14_digit_id(id_original_ocr.text)
    id_candidate_binary = extract_14_digit_id(id_binary_ocr.text)

    if id_candidate_original:
        id_text = id_candidate_original
        id_confidence = id_original_ocr.confidence
        id_raw_text = id_original_ocr.text
    else:
        id_text = id_candidate_binary or ""
        id_confidence = id_binary_ocr.confidence
        id_raw_text = id_binary_ocr.text

    full_name = clean_field("full_name", name_ocr.text)
    address = clean_field("address", address_ocr.text)
    id_info = validate_egyptian_national_id(id_text)

    return {
        "full_name": full_name,
        "address": address,
        "national_id": id_info.national_id,
        "validations": {
            "national_id_is_valid": id_info.is_valid,
            "name_is_arabic": is_arabic_text(full_name),
            "address_is_arabic": is_arabic_text(address),
            "national_id_validation_reason": id_info.reason,
        },
        "national_id_info": asdict(id_info),
        "confidence": {
            "full_name": round(name_ocr.confidence, 4),
            "address": round(address_ocr.confidence, 4),
            "national_id": round(id_confidence, 4),
        },
        "raw_ocr": {
            "full_name": name_ocr.text,
            "address": address_ocr.text,
            "national_id": id_raw_text,
        },
    }


def process_id_image(
    image_path: str | Path,
    engine_name: str = "easyocr",
    use_gpu: bool = False,
    save_debug: bool = False,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Process an image path and return structured OCR output."""
    image = read_image(image_path)
    return process_id_image_array(
        image=image,
        engine_name=engine_name,
        use_gpu=use_gpu,
        save_debug=save_debug,
        output_dir=output_dir,
    )
