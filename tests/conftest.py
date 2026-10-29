import os
import sys

# Tests import the backend modules the same way main.py does (flat imports).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Fixed settings, set before any backend module is imported. load_dotenv()
# never overrides variables that already exist, so neither backend/.env nor
# the shell can change these — and the real API key never enters the tests.
os.environ.update({
    "ANTHROPIC_API_KEY": "test-key",
    "ANTHROPIC_MODEL": "claude-sonnet-5-5",
    "MAX_OUTPUT_TOKENS": "6000",
    "OCR_ENGINE": "auto",
    "RATE_LIMIT_PER_HOUR": "0",
    "MAX_DOCUMENT_CHARS": "20000",
    "MAX_UPLOAD_MB": "10",
    "MAX_CONCURRENT_JOBS": "8",
    "ALLOWED_ORIGINS": "",
})
