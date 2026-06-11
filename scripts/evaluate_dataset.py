import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.pipeline import process_id_image
from src.metrics import character_error_rate, word_error_rate


def evaluate(labels_path: Path):
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    field_scores = {
        "full_name": [],
        "address": [],
        "national_id": [],
    }

    results = []

    for item in labels:
        image_path = item["image"]
        prediction = process_id_image(
            image_path=image_path,
            save_debug=False,
        )

        row = {
            "image": image_path,
            "ground_truth": item,
            "prediction": {
                "full_name": prediction.get("full_name", ""),
                "address": prediction.get("address", ""),
                "national_id": prediction.get("national_id", ""),
            },
            "cer": {},
            "wer": {},
        }

        for field in field_scores:
            pred_text = row["prediction"].get(field, "")
            gt_text = item.get(field, "")

            cer = character_error_rate(pred_text, gt_text)
            wer = word_error_rate(pred_text, gt_text)

            row["cer"][field] = cer
            row["wer"][field] = wer
            field_scores[field].append(cer)

        results.append(row)

    print("\nEvaluation Results")
    print("=" * 60)

    for field, scores in field_scores.items():
        avg_cer = sum(scores) / len(scores) if scores else 0
        print(f"{field:15s} CER: {avg_cer:.4f}")

    output_path = Path("results/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Detailed results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=str,
        default="data/synthetic_arabic/labels.json",
        help="Path to labels.json",
    )
    args = parser.parse_args()

    evaluate(Path(args.labels))


if __name__ == "__main__":
    main()