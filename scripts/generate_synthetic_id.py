"""Generate a very simple synthetic ID-like image for pipeline testing.

This is not meant to replicate official IDs. It is only for safe demo/testing
without using real sensitive documents.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


OUTPUT_PATH = Path("data/synthetic/synthetic_id_demo.png")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    card = np.full((630, 1000, 3), 245, dtype=np.uint8)
    cv2.rectangle(card, (20, 20), (980, 610), (180, 180, 180), 3)
    cv2.rectangle(card, (45, 150), (260, 400), (210, 210, 210), -1)
    cv2.rectangle(card, (45, 150), (260, 400), (120, 120, 120), 2)

    # OpenCV Hershey fonts do not support Arabic shaping. These placeholders
    # are still useful for testing perspective/ROI/debug flow.
    cv2.putText(card, "NAME: MOHAMED AHMED ALI", (310, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    cv2.putText(card, "ADDRESS: NASR CITY CAIRO", (310, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2)
    cv2.putText(card, "30101011234567", (300, 535), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (20, 20, 20), 3)

    # Add a mild perspective effect to test warping.
    src = np.float32([[0, 0], [999, 0], [999, 629], [0, 629]])
    dst = np.float32([[50, 35], [950, 5], [980, 610], [15, 590]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    canvas = np.full((700, 1100, 3), 255, dtype=np.uint8)
    warped = cv2.warpPerspective(card, matrix, (1100, 700), borderValue=(255, 255, 255))
    mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) < 250
    canvas[mask] = warped[mask]

    cv2.imwrite(str(OUTPUT_PATH), canvas)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
