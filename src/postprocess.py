"""Post-processing and validation logic for OCR output."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re

from .config import ARABIC_LETTERS_PATTERN


EASTERN_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
WESTERN_DIGITS = "0123456789"
DIGIT_TRANSLATION = str.maketrans(
    EASTERN_ARABIC_DIGITS + PERSIAN_DIGITS,
    WESTERN_DIGITS + WESTERN_DIGITS,
)

ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
TATWEEL = "ـ"


@dataclass
class NationalIDInfo:
    national_id: str
    is_valid: bool
    birth_date: str | None = None
    governorate_code: str | None = None
    gender: str | None = None
    reason: str | None = None


def normalize_digits(text: str) -> str:
    """Convert Eastern Arabic/Persian digits to Western digits."""
    return text.translate(DIGIT_TRANSLATION)


def strip_field_label(field_name: str, text: str) -> str:
    """Remove common printed field labels from OCR output."""
    label_patterns = {
        "full_name": [
            r"^\s*name\s*[:：\-]?\s*",
            r"^\s*full\s*name\s*[:：\-]?\s*",
            r"^\s*الاسم\s*[:：\-]?\s*",
            r"^\s*اسم\s*[:：\-]?\s*",
        ],
        "address": [
            r"^\s*address\s*[:：\-]?\s*",
            r"^\s*العنوان\s*[:：\-]?\s*",
            r"^\s*عنوان\s*[:：\-]?\s*",
        ],
    }

    for pattern in label_patterns.get(field_name, []):
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()

def remove_field_labels(text: str) -> str:
    """Remove common printed ID-card labels from OCR text."""
    label_patterns = [
        r"\bname\b",
        r"\bfull\s*name\b",
        r"\baddress\b",
        r"الاسم",
        r"\bاسم\b",
        r"العنوان",
        r"العنو\s*ان",
        r"\bعنوان\b",
        r"الرقم\s*القومي",
        r"رقم\s*قومي",
    ]

    for pattern in label_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_name_address_text(text: str, field_name: str) -> str:
    """Clean OCR output for name/address fields."""
    text = normalize_digits(text)
    text = remove_field_labels(text)

    text = ARABIC_DIACRITICS.sub("", text)
    text = text.replace(TATWEEL, "")

    # Keep Arabic letters, English letters, and spaces.
    text = re.sub(
        r"[^A-Za-z\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s]",
        " ",
        text,
    )

    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_arabic_text(text: str) -> str:
    """Arabic-only normalization used when strict Arabic cleaning is needed."""
    text = ARABIC_DIACRITICS.sub("", text)
    text = text.replace(TATWEEL, "")
    text = re.sub(r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_14_digit_id(text: str) -> str | None:
    """Extract the most plausible 14-digit Egyptian national ID."""
    text = normalize_digits(text)
    digits = re.sub(r"\D", "", text)

    candidates = re.findall(r"\d{14}", digits)
    if not candidates:
        return None

    # Prefer candidates with a plausible Egyptian century digit.
    for candidate in candidates:
        if candidate[0] in {"2", "3"}:
            return candidate

    return candidates[0]


def validate_egyptian_national_id(national_id: str | None) -> NationalIDInfo:
    """Validate basic structure of an Egyptian national ID.

    Egyptian national IDs are 14 digits. Common high-level structure:
    C YY MM DD GG SSSS X
    C: century marker, usually 2 for 1900s or 3 for 2000s
    YYMMDD: birth date
    GG: governorate code
    SSSS: serial; the last serial digit is commonly used for gender parity
    X: checksum/control digit
    """
    if not national_id:
        return NationalIDInfo("", False, reason="missing_id")

    national_id = normalize_digits(national_id)

    if not re.fullmatch(r"\d{14}", national_id):
        return NationalIDInfo(national_id, False, reason="id_must_be_14_digits")

    century_digit = national_id[0]
    if century_digit not in {"2", "3"}:
        return NationalIDInfo(national_id, False, reason="invalid_century_digit")

    century = 1900 if century_digit == "2" else 2000
    year = century + int(national_id[1:3])
    month = int(national_id[3:5])
    day = int(national_id[5:7])

    try:
        birth = date(year, month, day)
    except ValueError:
        return NationalIDInfo(national_id, False, reason="invalid_birth_date")

    if birth > date.today():
        return NationalIDInfo(national_id, False, reason="birth_date_in_future")

    governorate_code = national_id[7:9]
    gender_digit = int(national_id[12])
    gender = "male" if gender_digit % 2 == 1 else "female"

    return NationalIDInfo(
        national_id=national_id,
        is_valid=True,
        birth_date=birth.isoformat(),
        governorate_code=governorate_code,
        gender=gender,
    )


def is_arabic_text(text: str) -> bool:
    """Return True when text contains only Arabic letters/spaces after cleaning."""
    if not text:
        return False
    return bool(re.fullmatch(ARABIC_LETTERS_PATTERN, text))

def clean_field(field_name: str, raw_text: str) -> str:
    """Clean OCR text according to the target field."""
    if field_name == "national_id":
        return extract_14_digit_id(raw_text) or ""

    if field_name in {"full_name", "address"}:
        return normalize_name_address_text(raw_text, field_name)

    return raw_text.strip()


