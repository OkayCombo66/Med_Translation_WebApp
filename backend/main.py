import ipaddress
import os
import re
import threading
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load backend/.env before importing modules that read settings at import time.
load_dotenv(Path(__file__).with_name(".env"))

import anthropic  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import llm  # noqa: E402
from medlineplus import index_stats  # noqa: E402
from ocr import OCRError, extract_text, ocr_engine  # noqa: E402
from pipeline import run_pipeline  # noqa: E402
from prompts import READING_LEVELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
SAMPLES_DIR = ROOT / "samples"

MAX_DOCUMENT_CHARS = int(os.getenv("MAX_DOCUMENT_CHARS", "20000"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
# Requests per client IP per hour for the endpoints that call Claude. 0 = no limit.
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "30"))
# Documents processed at the same time. Extra requests get a "busy" message
# instead of queueing, so the page itself always stays responsive.
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "8"))

app = FastAPI(title="Clearchart — Plain-Language Medical Document Translator")

# The frontend is served by this same app, so CORS is only needed if you host
# the frontend somewhere else. List those origins in ALLOWED_ORIGINS.
allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


# --- request size limit ---------------------------------------------------------
# FastAPI reads the whole multipart body before the endpoint runs, so the size
# check has to happen earlier, as it streams in.
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024  # room for form fields and multipart boundaries


class BodySizeLimit:
    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/api/"):
            return await self.app(scope, receive, send)
        too_large = JSONResponse(
            {"detail": f"That request is too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)."},
            status_code=413,
        )
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and (not length.isdigit() or int(length) > self.max_bytes):
            return await too_large(scope, receive, send)

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self.max_bytes:
                raise _BodyTooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await too_large(scope, receive, send)


class _BodyTooLarge(Exception):
    pass


app.add_middleware(BodySizeLimit, max_bytes=MAX_REQUEST_BYTES)


# --- rate limiting -----------------------------------------------------------
# A simple in-memory sliding window per client IP, so a public demo link can't
# run up the API bill. It resets when the server restarts, which is fine here.
# Behind a proxy, set uvicorn's FORWARDED_ALLOW_IPS to the proxy's address so
# request.client.host is the real visitor (never '*': clients could spoof it).
RATE_WINDOW_SECONDS = 3600
MAX_TRACKED_CLIENTS = 100_000
_recent_requests: "OrderedDict[str, deque]" = OrderedDict()  # least recently seen first
_rate_lock = threading.Lock()


def _client_key(request: Request) -> str:
    host = request.client.host if request.client else "unknown"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host
    if ip.version == 6:  # one person can rotate through a whole /64 block
        return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return str(ip)


def check_rate_limit(request: Request) -> None:
    if RATE_LIMIT_PER_HOUR <= 0:
        return
    key = _client_key(request)
    now = time.monotonic()
    with _rate_lock:
        # Forget clients we haven't seen for a whole window (oldest first).
        while _recent_requests:
            oldest_key, oldest = next(iter(_recent_requests.items()))
            if oldest and now - oldest[-1] <= RATE_WINDOW_SECONDS and len(_recent_requests) <= MAX_TRACKED_CLIENTS:
                break
            _recent_requests.popitem(last=False)

        window = _recent_requests.pop(key, deque())
        while window and now - window[0] > RATE_WINDOW_SECONDS:
            window.popleft()
        _recent_requests[key] = window  # re-insert as most recently seen
        if len(window) >= RATE_LIMIT_PER_HOUR:
            raise HTTPException(429, "You've hit the hourly limit for this demo. Please try again later.")
        window.append(now)


_job_slots = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)


@contextmanager
def job_slot():
    if not _job_slots.acquire(blocking=False):
        raise HTTPException(503, "The server is busy with other documents. Please try again in a minute.")
    try:
        yield
    finally:
        _job_slots.release()


# --- error handling ----------------------------------------------------------
@contextmanager
def claude_errors():
    """Turn Claude API failures into HTTP errors with messages a user can act on."""
    try:
        yield
    except llm.LLMError as e:
        raise HTTPException(502, str(e)) from e
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
        raise HTTPException(
            503, "The server's Anthropic API key was rejected. Check ANTHROPIC_API_KEY in backend/.env."
        ) from e
    except anthropic.NotFoundError as e:
        raise HTTPException(502, f"The model '{llm.MODEL}' isn't available to this API key.") from e
    except anthropic.RateLimitError as e:
        raise HTTPException(429, "Claude is rate-limiting requests right now. Wait a minute and try again.") from e
    except anthropic.BadRequestError as e:
        raise HTTPException(502, f"Claude rejected the request: {e.message}") from e
    except anthropic.APIStatusError as e:
        raise HTTPException(503, "Claude is temporarily unavailable. Try again in a moment.") from e
    except anthropic.APIConnectionError as e:
        raise HTTPException(502, "Couldn't reach the Claude API. Check the server's internet connection.") from e


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "Something went wrong on the server."})


def require_api_key() -> None:
    if not llm.api_key_configured():
        raise HTTPException(503, "The server has no ANTHROPIC_API_KEY. Add it to backend/.env and restart.")


# --- API ---------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": llm.MODEL,
        "api_key_configured": llm.api_key_configured(),
        "ocr_engine": ocr_engine(),
        "medlineplus_topics": index_stats().get("topics", 0),
    }


SAMPLE_CATEGORIES = {"lab": "lab", "discharge": "discharge", "radiology": "radiology", "tricky": "tricky"}
ACRONYMS = {"cbc", "ct", "mri", "copd", "ckd", "bmp", "cmp", "a1c", "tsh", "ldl", "hdl"}


def describe_sample(path: Path) -> dict:
    words = path.stem.split("_")
    category = SAMPLE_CATEGORIES.get(words[0], "other")
    if category != "other":
        words = words[1:]
    title = " ".join(w.upper() if w in ACRONYMS else w for w in words)
    return {"id": path.stem, "title": title[:1].upper() + title[1:], "category": category}


@app.get("/api/samples")
def list_samples():
    return [describe_sample(p) for p in sorted(SAMPLES_DIR.glob("*.txt"))]


@app.get("/api/samples/{sample_id}")
def get_sample(sample_id: str):
    path = SAMPLES_DIR / f"{sample_id}.txt"
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", sample_id) or not path.is_file():
        raise HTTPException(404, "No sample with that name.")
    return {**describe_sample(path), "text": path.read_text(encoding="utf-8")}


@app.post("/api/ocr")
def ocr(request: Request, file: UploadFile = File(...)):
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"That file is too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    check_rate_limit(request)

    try:
        with job_slot(), claude_errors():
            text, engine = extract_text(data, file.filename or "", file.content_type or "")
    except OCRError as e:
        raise HTTPException(400, str(e)) from e

    if len(text.strip()) < 20:
        raise HTTPException(
            400,
            "Couldn't find readable text in that file. Try a clearer, well-lit photo, or paste the text directly.",
        )
    warning = None
    if len(text) > MAX_DOCUMENT_CHARS:
        warning = f"The extracted text is longer than {MAX_DOCUMENT_CHARS:,} characters; trim it before translating."
    return {"text": text, "engine": engine, "warning": warning}


@app.post("/api/translate")
def translate(request: Request, document_text: str = Form(""), reading_level: str = Form("8th")):
    document_text = document_text.strip()
    if not document_text:
        raise HTTPException(400, "No document text provided.")
    if len(document_text) > MAX_DOCUMENT_CHARS:
        raise HTTPException(413, f"That document is too long (max {MAX_DOCUMENT_CHARS:,} characters).")
    if reading_level not in READING_LEVELS:
        reading_level = "8th"
    require_api_key()
    check_rate_limit(request)
    with job_slot(), claude_errors():
        result = run_pipeline(document_text, reading_level)
    result.pop("usage", None)
    return result


# Serve the frontend as static files (registered last so /api routes win).
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
