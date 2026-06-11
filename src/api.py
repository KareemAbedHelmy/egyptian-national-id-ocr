"""FastAPI endpoint for Egyptian National ID OCR."""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from .pipeline import process_id_image_array

app = FastAPI(
    title="Egyptian National ID OCR API",
    description="Extract Arabic name, address, and 14-digit national ID from ID-card images.",
    version="0.1.0",
)


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "message": "Egyptian National ID OCR API is running"}


@app.post("/extract-id")
async def extract_id(
    file: UploadFile = File(...),
    engine: str = "easyocr",
    debug: bool = False,
) -> dict:
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Please upload a JPG or PNG image.")

    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

    try:
        return process_id_image_array(
            image=image,
            engine_name=engine,
            save_debug=debug,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
