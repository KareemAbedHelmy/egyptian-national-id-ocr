"""Command-line inference script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline import process_id_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Egyptian National ID OCR inference")
    parser.add_argument("--image", required=True, help="Path to JPG/PNG ID-card image")
    parser.add_argument(
        "--engine",
        default="easyocr",
        choices=["easyocr", "paddleocr"],
        help="OCR engine to use",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU if supported by OCR backend")
    parser.add_argument("--debug", action="store_true", help="Save preprocessing and ROI debug images")
    parser.add_argument("--output-dir", default="results", help="Directory for debug images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = process_id_image(
        image_path=Path(args.image),
        engine_name=args.engine,
        use_gpu=args.gpu,
        save_debug=args.debug,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
