"""
Text extraction for photo/scan/PDF uploads. This is what powers the "take a
photo" flow on mobile — a plain <input capture="environment"> on the frontend
feeds a file here, no native app needed.

Engines, in order of preference:
- PDFs with a text layer (e.g. patient-portal downloads): read the text
  directly with pypdf. Exact, free, no OCR needed.
- Images: Tesseract if it's installed on the machine (local, deterministic).
- Otherwise: Claude vision transcribes the image (or a scanned PDF).

Whatever the engine, the frontend shows the extracted text to the user to
check against their paper copy before anything is translated.
"""

import base64
import io
import os
import shutil

from PIL import Image, ImageOps, UnidentifiedImageError

import llm
from prompts import OCR_SYSTEM, OCR_USER_TEXT

MAX_IMAGE_SIDE = 2000  # px; larger photos are downscaled before OCR
MAX_PDF_PAGES = 20
# Transcription output grows with page count, so it gets more room than the
# shared default (thinking counts toward it). Non-streaming calls must stay
# below ~21k tokens.
OCR_MAX_TOKENS = min(int(os.getenv("OCR_MAX_TOKENS", "16000")), 21000)

# A small file can declare enormous pixel dimensions ("decompression bomb").
# 40 megapixels is far more than any phone photo; above that, refuse.
Image.MAX_IMAGE_PIXELS = 40_000_000


class OCRError(Exception):
    """Raised with a message that is safe to show to the user."""


def ocr_engine() -> str:
    """Which engine images go to: OCR_ENGINE=auto|tesseract|claude (default auto)."""
    choice = os.getenv("OCR_ENGINE", "auto").lower()
    has_tesseract = shutil.which("tesseract") is not None
    if choice == "tesseract":
        return "tesseract" if has_tesseract else "none"
    if choice == "claude":
        return "claude" if llm.api_key_configured() else "none"
    if has_tesseract:
        return "tesseract"
    return "claude" if llm.api_key_configured() else "none"


def extract_text(data: bytes, filename: str = "", content_type: str = "") -> tuple[str, str]:
    """Return (text, engine_used) for an uploaded image or PDF."""
    if data[:5] == b"%PDF-" or content_type == "application/pdf" or filename.lower().endswith(".pdf"):
        return _extract_pdf(data)
    return _extract_image(data)


def _extract_pdf(data: bytes) -> tuple[str, str]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if len(reader.pages) > MAX_PDF_PAGES:
            raise OCRError(f"That PDF has {len(reader.pages)} pages; the limit is {MAX_PDF_PAGES}.")
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except OCRError:
        raise
    except Exception as e:  # pypdf raises many different error types
        raise OCRError("Couldn't read that PDF. Try exporting it again, or paste the text.") from e

    if len(text) >= 20:
        return text, "pdf-text"

    # No text layer: it's a scanned PDF, so it needs vision.
    if not llm.api_key_configured():
        raise OCRError("This PDF is a scanned image and no OCR engine is available. Paste the text instead.")
    block = {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.standard_b64encode(data).decode("ascii"),
        },
    }
    return _claude_transcribe(block), "claude"


def _extract_image(data: bytes) -> tuple[str, str]:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Image.DecompressionBombError as e:
        raise OCRError("That image is too large to process. Try a normal phone photo or a screenshot.") from e
    except (UnidentifiedImageError, OSError) as e:
        raise OCRError(
            "That file isn't an image we can read. Upload a JPG, PNG or PDF "
            "(on iPhone: Settings → Camera → Formats → Most Compatible)."
        ) from e

    # Phone photos are often stored sideways with an EXIF rotation tag.
    image = ImageOps.exif_transpose(image)
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

    engine = ocr_engine()
    if engine == "tesseract":
        import pytesseract

        # Grayscale noticeably helps Tesseract on phone-camera photos.
        return pytesseract.image_to_string(image.convert("L")).strip(), "tesseract"
    if engine == "claude":
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(buffer.getvalue()).decode("ascii"),
            },
        }
        return _claude_transcribe(block), "claude"
    raise OCRError(
        "No OCR engine is available: install Tesseract or set ANTHROPIC_API_KEY. "
        "You can paste the text instead."
    )


def _claude_transcribe(block: dict) -> str:
    # A refusal or cut-off is a Claude-side problem, not bad input, so llm.LLMError
    # is left to propagate; main.claude_errors() turns it into a 502.
    result = llm.ask_claude(
        OCR_SYSTEM,
        [block, {"type": "text", "text": OCR_USER_TEXT}],
        effort="low",
        max_tokens=max(llm.MAX_OUTPUT_TOKENS, OCR_MAX_TOKENS),
    )
    if result.text.strip() == "NO_TEXT_FOUND":
        return ""
    return result.text
