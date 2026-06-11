# Egyptian National ID OCR

A specialized OCR pipeline for extracting structured data from Egyptian National ID card images:

- Full Arabic name
- Arabic address
- 14-digit national ID number

The project combines image preprocessing, region-of-interest extraction, Arabic OCR, post-processing validation, and a FastAPI deployment endpoint.

> Use synthetic, sample, or redacted ID images only. Do not commit real ID images to GitHub.

---

## Pipeline

```text
Input JPG/PNG
   ↓
OpenCV preprocessing
   ↓
Perspective transform to top-down card view
   ↓
Resize card to a fixed template size
   ↓
Crop name, address, and national ID regions
   ↓
Run Arabic OCR per region
   ↓
Clean text and validate with Regex
   ↓
Return structured JSON
```

---

## Project Structure

```text
egyptian-national-id-ocr/
│
├── data/
│   ├── samples/
│   └── synthetic/
│
├── results/
│
├── scripts/
│   └── generate_synthetic_id.py
│
├── src/
│   ├── api.py
│   ├── config.py
│   ├── metrics.py
│   ├── ocr_engine.py
│   ├── pipeline.py
│   ├── postprocess.py
│   ├── preprocess.py
│   └── regions.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Generate a Synthetic Test Image

This creates a safe demo image that does not contain real ID data:

```bash
python scripts/generate_synthetic_id.py
```

Output:

```text
data/synthetic/synthetic_id_demo.png
```

---

## Run Inference

```bash
python main.py --image data/synthetic/synthetic_id_demo.png --debug
```

The `--debug` flag saves preprocessing outputs in the `results/` folder:

```text
results/
├── 01_original.jpg
├── 02_resized.jpg
├── 03_warped_raw.jpg
├── 04_warped_standard.jpg
├── 05_warped_binary.jpg
├── 06_roi_debug.jpg
├── full_name_crop.jpg
├── address_crop.jpg
├── national_id_crop.jpg
├── full_name_binary.jpg
├── address_binary.jpg
└── national_id_binary.jpg
```

---

## Example JSON Output

```json
{
  "full_name": "محمد أحمد علي",
  "address": "مدينة نصر القاهرة",
  "national_id": "30101011234567",
  "validations": {
    "national_id_is_valid": true,
    "name_is_arabic": true,
    "address_is_arabic": true,
    "national_id_validation_reason": null
  },
  "national_id_info": {
    "national_id": "30101011234567",
    "is_valid": true,
    "birth_date": "2001-01-01",
    "governorate_code": "12",
    "gender": "male",
    "reason": null
  },
  "confidence": {
    "full_name": 0.91,
    "address": 0.87,
    "national_id": 0.96
  }
}
```

---

## Run the API

Start the FastAPI server:

```bash
uvicorn src.api:app --reload
```

Open the interactive docs:

```text
http://127.0.0.1:8000/docs
```

Send a request:

```bash
curl -X POST "http://127.0.0.1:8000/extract-id" \
  -F "file=@data/synthetic/synthetic_id_demo.png"
```

---

## OCR Backend

The default backend is EasyOCR because it is straightforward to install and supports Arabic text.

```bash
python main.py --image path/to/image.jpg --engine easyocr --debug
```

Optional PaddleOCR support is included in the code, but PaddleOCR installation can vary by operating system and Python version. Install it manually if needed:

```bash
pip install paddleocr paddlepaddle
```

Then run:

```bash
python main.py --image path/to/image.jpg --engine paddleocr --debug
```

---

## Evaluation Metrics

The project includes Character Error Rate and Word Error Rate functions in `src/metrics.py`.

```python
from src.metrics import character_error_rate, word_error_rate

cer = character_error_rate("30101011234567", "30101011234567")
wer = word_error_rate("محمد أحمد علي", "محمد احمد علي")
```

Recommended report format:

```text
Field          CER
Name           0.12
Address        0.18
National ID    0.03
```

---

## Privacy and Security Notes

National ID cards contain sensitive personal information. In a production environment:

- Do not store uploaded images unless absolutely necessary.
- Do not commit real ID images to GitHub.
- Use HTTPS for all uploads.
- Encrypt any stored data at rest.
- Apply access control and audit logging.
- Redact or mask national ID numbers in logs.
- Delete temporary files after processing.

---

## Current Limitations

- ROI coordinates are approximate and should be tuned based on the card template and sample images.
- Real Arabic OCR accuracy depends on image quality and the OCR backend.
- Synthetic demo images are useful for pipeline testing but not a replacement for real evaluation data.
- Fine-tuning PaddleOCR/TrOCR on labeled Egyptian ID crops would improve production accuracy.

---

## Future Improvements

- Add a small labeled test set with CER/WER reporting.
- Fine-tune the recognition model on Arabic ID crops.
- Add automatic orientation detection for upside-down cards.
- Add image quality checks before OCR.
- Add Docker support.
- Add a Streamlit or Gradio demo.
