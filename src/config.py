"""Project configuration.

The ROI coordinates are expressed as percentages of a standardized, warped
ID card image. You should tune these values based on your sample/synthetic
Egyptian ID card layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"

# Standard size after perspective transform. A consistent size makes ROI
# cropping much easier and more reproducible.
STANDARD_CARD_WIDTH = 1000
STANDARD_CARD_HEIGHT = 630


@dataclass(frozen=True)
class ROI:
    """Region of interest using normalized coordinates.

    Values are in the range [0, 1] and represent:
    x1, y1, x2, y2 as a percentage of width/height.
    """

    x1: float
    y1: float
    x2: float
    y2: float


# Approximate front-side Egyptian ID layout:
# - photo is usually on the left
# - Arabic fields are mostly on the right/center
# - national ID number is near the bottom
# Tune these after viewing your saved ROI debug images.
ROI_CONFIG: dict[str, ROI] = {
    "full_name": ROI(0.30, 0.20, 0.96, 0.37),
    "address": ROI(0.30, 0.40, 0.96, 0.66),
    "national_id": ROI(0.25, 0.77, 0.96, 0.94),
}

# Arabic Unicode ranges used for validation/cleaning.
ARABIC_LETTERS_PATTERN = r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s]+"
