# Egyptian National ID OCR Pipeline

A specialized Optical Character Recognition (OCR) pipeline for extracting structured information from Egyptian National ID card images.

The system extracts:

* Full name
* Address
* 14-digit national ID number

It uses OpenCV-based image preprocessing, region-of-interest extraction, Arabic OCR, post-processing validation, and a FastAPI endpoint for deployment.

---

## Project Overview

This project was developed as an end-to-end OCR pipeline for Egyptian National ID cards. The goal is to handle realistic image conditions such as tilted cards, perspective distortion, low-quality images, noise, and varied lighting.

The pipeline does not simply pass the full image directly to OCR. Instead, it first corrects the card perspective, standardizes the layout, crops the important fields, applies OCR to each region, and then validates the extracted output.

---

## Features

* Perspective transformation to correct angled ID card images
* Image denoising and binarization using OpenCV
* Region-of-interest cropping for name, address, and national ID fields
* Arabic and English OCR support for synthetic testing
* Eastern Arabic digit normalization
* Regex-based national ID extraction
* Egyptian national ID validation logic
* Character Error Rate (CER) and Word Error Rate (WER) evaluation
* FastAPI endpoint for image upload and JSON response
* Synthetic Arabic ID generator for safe testing without real personal data

---

## Pipeline

```text
Input ID Image
    ↓
Image Preprocessing
    ↓
Perspective Transformation
    ↓
Standardized ID Layout
    ↓
ROI Cropping
    ↓
OCR Recognition
    ↓
Post-processing and Validation
    ↓
Structured JSON Output
```

---

## Repository Structure

```text
egyptian-national-id-ocr/
│
├── assets/
│   ├── og.jpg
│   ├── resized.jpg
│   ├── warped_raw.jpg
│   ├── warped_standard.jpg
│   ├── warped_binary.jpg
│   └── roi_debug.jpg
│
├── data/
│   ├── synthetic/
│   └── synthetic_arabic/
│
├── scripts/
│   ├── generate_synthetic_id.py
│   ├── generate_arabic_synthetic_ids.py
│   └── evaluate_dataset.py
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

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Generate Synthetic Test Data

Generate an English synthetic ID sample:

```bash
python scripts/generate_synthetic_id.py
```

Generate Arabic synthetic ID samples:

```bash
python scripts/generate_arabic_synthetic_ids.py --count 10
```

The Arabic generator creates safe synthetic ID images and a `labels.json` file for evaluation.

---

## Run Inference

Run OCR on an English synthetic sample:

```bash
python main.py --image data/synthetic/synthetic_id_demo.png --debug
```

Run OCR on an Arabic synthetic sample:

```bash
python main.py --image data/synthetic_arabic/synthetic_arabic_id_000.png --debug
```

The `--debug` option saves intermediate preprocessing images and ROI visualizations.

---

## Example Output

```json
{
  "full_name": "محمد احمد علي",
  "address": "القاهرة مدينة نصر",
  "national_id": "30409211598424",
  "validations": {
    "national_id_is_valid": true,
    "name_is_arabic": true,
    "address_is_arabic": true,
    "national_id_validation_reason": null
  },
  "national_id_info": {
    "national_id": "30409211598424",
    "is_valid": true,
    "birth_date": "2004-09-21",
    "governorate_code": "15",
    "gender": "female",
    "reason": null
  },
  "confidence": {
    "full_name": 0.5746,
    "address": 0.8705,
    "national_id": 0.9999
  },
  "raw_ocr": {
    "full_name": "محمد احمد علي",
    "address": "القاهرة مدينة نصر",
    "national_id": "30409211598424"
  }
}
```

---

## Preprocessing Results

### Original Image

![Original ID image](assets/og.jpg)

### Resized Image

![Resized ID image](assets/resized.jpg)

### Raw Perspective Transform

![Raw warped image](assets/warped_raw.jpg)

### Standardized Perspective Transform

![Standardized warped image](assets/warped_standard.jpg)

### Binarized Image

![Binarized ID image](assets/warped_binary.jpg)

### Region of Interest Debug View

![ROI debug image](assets/roi_debug.jpg)

---

## FastAPI Deployment

Start the API server:

```bash
uvicorn src.api:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Use the `/extract-id` endpoint to upload a JPG or PNG image and receive a structured JSON response.

---

## Evaluation

Run evaluation on the Arabic synthetic dataset:

```bash
python scripts/evaluate_dataset.py --labels data/synthetic_arabic/labels.json
```

The script reports Character Error Rate (CER) for each field:

```text
Evaluation Results
============================================================
full_name       CER: ...
address         CER: ...
national_id     CER: ...
============================================================
```

CER is calculated as:

```text
Character Error Rate = Edit Distance / Number of Characters in Ground Truth
```

This helps measure how close the OCR prediction is to the correct text.

---

## Post-processing and Validation

The post-processing layer improves raw OCR output using:

* Regex validation for the 14-digit national ID
* Removal of non-digit characters from national ID OCR output
* Normalization of Eastern Arabic numerals to Western digits
* Removal of OCR noise and unwanted symbols
* Arabic-only validation for name and address fields
* Basic logical parsing of the Egyptian national ID birth date, governorate code, and gender digit

---

## Data Privacy

This project uses synthetic data for testing and demonstration. Real national ID cards contain sensitive personal information and should not be uploaded to public repositories.

In a production environment:

* Do not store uploaded ID images unless strictly necessary
* Use HTTPS for secure data transmission
* Encrypt stored data at rest
* Restrict access to extracted personal information
* Log access carefully without exposing sensitive data
* Delete temporary files after processing
* Follow applicable data protection and privacy regulations

---

## Limitations

* The current implementation is optimized for synthetic/sample ID layouts.
* ROI coordinates may need adjustment for different card templates or camera conditions.
* Arabic OCR quality depends on image clarity, font style, and OCR engine performance.
* Severe blur, glare, occlusion, or missing card corners may reduce accuracy.

---

## Disclaimer

This project is intended for educational and technical demonstration purposes only. It should be tested using synthetic, sample, or properly redacted ID images. Do not commit or share real personal identification documents.
