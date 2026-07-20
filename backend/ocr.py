"""
OCR for photo/scan uploads. This is what powers the "take a photo" flow
on mobile — a plain <input capture="camera"> on the frontend feeds an
image here, no native app needed.
"""

import io
from PIL import Image
import pytesseract


def image_bytes_to_text(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes))
    # Basic preprocessing: convert to grayscale, which noticeably helps
    # Tesseract accuracy on phone-camera photos of printed documents.
    image = image.convert("L")
    text = pytesseract.image_to_string(image)
    return text.strip()
