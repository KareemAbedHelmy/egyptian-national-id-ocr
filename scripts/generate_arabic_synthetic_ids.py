import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import arabic_reshaper
from bidi.algorithm import get_display


OUTPUT_DIR = Path("data/synthetic_arabic")
LABELS_PATH = OUTPUT_DIR / "labels.json"

CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 630


NAMES = [
    "محمد احمد علي",
    "احمد محمود حسن",
    "علي محمد ابراهيم",
    "كريم خالد يوسف",
    "محمود سيد عبدالله",
]

ADDRESSES = [
    "مدينة نصر القاهرة",
    "المعادي القاهرة",
    "الدقي الجيزة",
    "سموحة الاسكندرية",
    "شبرا القاهرة",
]


def find_font():
    possible_fonts = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/seguiemj.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for font_path in possible_fonts:
        if Path(font_path).exists():
            return font_path

    raise FileNotFoundError(
        "No Arabic-capable font found. Try using Arial, Tahoma, or DejaVuSans."
    )


def rtl_text(text: str) -> str:
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


def generate_national_id(index: int) -> str:
    """
    Generate a synthetic 14-digit Egyptian national ID.

    Structure:
    C YY MM DD GG SSSS X

    C     = century digit
    YYMMDD = birth date
    GG    = governorate code
    SSSS  = serial number
    X     = check/control digit
    """
    century = "3"
    year = random.randint(0, 5)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    gov_code = random.choice(["01", "02", "03", "11", "12", "13", "14", "15"])
    serial_first_three = random.randint(100, 999)
    gender_digit = random.choice([1, 3, 5, 7, 9, 2, 4, 6, 8])
    serial = f"{serial_first_three}{gender_digit}"

    check_digit = random.randint(0, 9)

    national_id = (
        f"{century}"
        f"{year:02d}"
        f"{month:02d}"
        f"{day:02d}"
        f"{gov_code}"
        f"{serial}"
        f"{check_digit}"
    )

    assert len(national_id) == 14
    return national_id

def apply_perspective_variation(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]

    margin = random.randint(5, 35)

    src = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ])

    dst = np.float32([
        [random.randint(0, margin), random.randint(0, margin)],
        [w - 1 - random.randint(0, margin), random.randint(0, margin)],
        [w - 1 - random.randint(0, margin), h - 1 - random.randint(0, margin)],
        [random.randint(0, margin), h - 1 - random.randint(0, margin)],
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (w, h), borderValue=(245, 245, 245))

    return warped


def apply_image_degradation(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)

    brightness = random.uniform(0.85, 1.15)
    image *= brightness

    noise = np.random.normal(0, random.uniform(2, 8), image.shape)
    image += noise

    image = np.clip(image, 0, 255).astype(np.uint8)

    if random.random() < 0.4:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    return image


def draw_arabic_id(name: str, address: str, national_id: str, output_path: Path):
    font_path = find_font()

    title_font = ImageFont.truetype(font_path, 30)
    text_font = ImageFont.truetype(font_path, 38)
    id_font = ImageFont.truetype(font_path, 52)

    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), (245, 245, 245))
    draw = ImageDraw.Draw(image)

    # Card border
    draw.rectangle((5, 5, CANVAS_WIDTH - 5, CANVAS_HEIGHT - 5), outline=(160, 160, 160), width=4)

    # Photo placeholder
    draw.rectangle((30, 140, 250, 405), outline=(110, 110, 110), width=3, fill=(210, 210, 210))

    # Field regions roughly matching src/config.py
    draw.rectangle((300, 125, 960, 245), outline=(180, 180, 180), width=1)
    draw.rectangle((300, 245, 960, 415), outline=(180, 180, 180), width=1)
    draw.rectangle((250, 485, 960, 595), outline=(180, 180, 180), width=1)

    # Labels and text
    draw.text((840, 90), rtl_text("الاسم"), font=title_font, fill=(0, 0, 0))
    draw.text((840, 210), rtl_text("العنوان"), font=title_font, fill=(0, 0, 0))
    draw.text((790, 450), rtl_text("الرقم القومي"), font=title_font, fill=(0, 0, 0))

    draw.text((500, 160), rtl_text(name), font=text_font, fill=(0, 0, 0))
    draw.text((500, 285), rtl_text(address), font=text_font, fill=(0, 0, 0))
    draw.text((300, 515), national_id, font=id_font, fill=(0, 0, 0))

    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    image_np = apply_perspective_variation(image_np)
    image_np = apply_image_degradation(image_np)

    cv2.imwrite(str(output_path), image_np)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = []

    for i in range(args.count):
        name = random.choice(NAMES)
        address = random.choice(ADDRESSES)
        national_id = generate_national_id(i)

        filename = f"synthetic_arabic_id_{i:03d}.png"
        output_path = OUTPUT_DIR / filename

        draw_arabic_id(name, address, national_id, output_path)

        labels.append({
            "image": str(output_path).replace("\\", "/"),
            "full_name": name,
            "address": address,
            "national_id": national_id,
        })

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"Generated {args.count} synthetic Arabic ID images.")
    print(f"Labels saved to {LABELS_PATH}")


if __name__ == "__main__":
    main()