import os
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from anthropic import Anthropic
import textstat

from flagging import parse_lab_values, flags_to_prompt_context
from prompts import build_translation_messages
from faithfulness import run_faithfulness_check
from ocr import image_bytes_to_text

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1500"))

app = FastAPI(title="Plain-Language Medical Document Translator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before a real deploy
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_explanation_and_concepts(raw_output: str):
    """Split the model's output into the explanation text and the CONCEPTS: json line."""
    concepts = []
    explanation_lines = []
    for line in raw_output.splitlines():
        if line.strip().startswith("CONCEPTS:"):
            try:
                concepts = json.loads(line.split("CONCEPTS:", 1)[1].strip())
            except json.JSONDecodeError:
                concepts = []
        else:
            explanation_lines.append(line)
    return "\n".join(explanation_lines).strip(), concepts


@app.post("/api/translate")
async def translate(
    document_text: str = Form(None),
    reading_level: str = Form("8th"),
    file: UploadFile = File(None),
):
    # Resolve source text: either pasted directly, or OCR'd from an upload
    if file is not None:
        image_bytes = await file.read()
        try:
            document_text = image_bytes_to_text(image_bytes)
        except Exception as e:
            raise HTTPException(400, f"Could not read image: {e}")
        if not document_text or len(document_text) < 20:
            raise HTTPException(
                400,
                "Couldn't extract readable text from that photo. Try a clearer, "
                "well-lit shot, or paste the text directly.",
            )

    if not document_text or not document_text.strip():
        raise HTTPException(400, "No document text provided.")

    # 1. Deterministic lab parsing + flagging (never trust the LLM with this)
    labs = parse_lab_values(document_text)
    lab_context = flags_to_prompt_context(labs)

    # 2. Translation call
    system, user = build_translation_messages(document_text, lab_context, reading_level)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw_output = "".join(b.text for b in resp.content if b.type == "text")
    explanation, concepts = get_explanation_and_concepts(raw_output)

    # 3. Faithfulness check (deterministic numeric + LLM judge)
    faithfulness = run_faithfulness_check(client, MODEL, document_text, explanation)

    # 4. Readability metrics, for the eval story
    readability = {
        "source_grade_level": round(textstat.flesch_kincaid_grade(document_text), 1),
        "explanation_grade_level": round(textstat.flesch_kincaid_grade(explanation), 1),
    }

    return {
        "source_text": document_text,
        "explanation": explanation,
        "concepts": concepts,
        "lab_values": [
            {
                "name": l.name,
                "value": l.value,
                "unit": l.unit,
                "ref_low": l.ref_low,
                "ref_high": l.ref_high,
                "flag": l.flag,
            }
            for l in labs
        ],
        "faithfulness": faithfulness,
        "readability": readability,
    }


# Serve the frontend as static files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")



@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
