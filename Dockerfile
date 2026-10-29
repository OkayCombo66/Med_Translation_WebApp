FROM python:3.13-slim

# Tesseract gives free, local OCR for photo uploads (Claude vision is the fallback).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend backend
COPY frontend frontend
COPY samples samples

# Set ANTHROPIC_API_KEY as an environment variable on your host; never bake it in.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --app-dir backend --host 0.0.0.0 --port ${PORT} --proxy-headers"]
