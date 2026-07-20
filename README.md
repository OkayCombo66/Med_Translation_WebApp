# Clearchart — Plain-Language Medical Document Translator

Paste, upload, or photograph a clinical document (discharge summary, lab
result, radiology report) and get a plain-language explanation, with
abnormal lab values flagged deterministically in code and every claim
checked against the source before it's shown to you.

**This is an educational tool, not medical advice.** Only use synthetic or
de-identified documents with it — never real patient data.

## Why this exists

Roughly a third of US adults struggle to understand their own clinical
documents. This project explores whether an LLM can close that gap safely —
"safely" meaning the model is only ever trusted with *language*, never with
*facts or judgment*. Facts (is this lab value high or low?) are computed in
plain code. Judgment (what does that mean?) is grounded in cited sources.
The model explains; it doesn't decide.

## Architecture

```
photo/paste ─▶ OCR (if photo) ─▶ deterministic lab parsing + flagging
                                          │
                                          ▼
                          LLM translation call (flags injected as
                          ground truth, reading level as instruction)
                                          │
                                          ▼
                    faithfulness check (numeric regex diff + LLM judge)
                                          │
                                          ▼
                          plain-language explanation + flags + checks
```

## Setup

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

You'll also need Tesseract installed system-wide for OCR:
- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt install tesseract-ocr`

Run it:

```bash
uvicorn main:app --reload --app-dir backend
```

Then open http://localhost:8000 — the frontend is served directly by the
backend, no separate build step needed.

## Running the eval

```bash
python eval/run_eval.py
```

This runs every `.txt` file in `samples/` through both the full pipeline and
a naive one-line-prompt baseline, and prints a comparison table of reading
grade level and numeric faithfulness. This table is the centerpiece of the
portfolio writeup — it's the evidence that the extra machinery (flagging,
structured prompting, verification) actually does something a naive prompt
doesn't.

## Roadmap (8-week plan)

- [x] **Weeks 1-2** — core pipeline: paste → structured explanation
- [x] **Weeks 1-2** — photo/upload input via OCR (no native app needed)
- [x] **Weeks 3-4** — deterministic lab flagging + faithfulness verifier
- [ ] **Weeks 5-6** — RAG over MedlinePlus for cited "learn more" links per
      concept (currently `concepts` are extracted but not yet retrieved
      against — see below)
- [ ] **Weeks 5-6** — polish UI, mobile responsiveness pass
- [ ] **Weeks 7-8** — build out samples/ to 15-25 documents, run full eval,
      write up results in this README with the comparison table

## Next build target: MedlinePlus RAG

The model already extracts a `concepts` list (conditions, tests,
medications) per document — that's the hook for retrieval. Plan:
1. Pull ~50-100 common MedlinePlus articles (topics matching what shows up
   in your sample documents) via their API, chunk into paragraphs.
2. Embed chunks (any embedding API), store as a flat file of `(text,
   vector)` pairs — no need for a real vector DB at this scale.
3. For each concept the model returns, retrieve the top matching chunk and
   surface it in the UI as a "Learn more" link with a one-line snippet.

## Deploying

Any host that runs a Python web service works (Render, Railway, Fly.io).
Set `ANTHROPIC_API_KEY` as an environment variable there rather than
committing `.env`. Add a rate limit before sharing a public link so a demo
page can't run up your API bill.
