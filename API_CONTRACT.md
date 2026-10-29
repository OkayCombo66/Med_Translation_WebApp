# Clearchart API contract (shared by backend + frontend)

All endpoints are same-origin (the FastAPI app serves `frontend/` at `/`).
Errors: non-2xx responses carry JSON `{"detail": "<human readable message>"}`.
429 = rate limited, 413 = input too large, 400 = bad input, 502/503 = upstream (Claude) problem.

## GET /api/health
{"status": "ok", "model": "claude-sonnet-5-5", "api_key_configured": true,
 "ocr_engine": "tesseract" | "claude" | "none", "medlineplus_topics": 1017}

## GET /api/samples
[{"id": "lab_cbc_iron_deficiency", "title": "CBC iron deficiency", "category": "lab" | "discharge" | "radiology" | "tricky" | "other"}]

## GET /api/samples/{id}
{"id": "...", "title": "...", "text": "..."}

## POST /api/ocr   (multipart/form-data: file = image (jpg/png/webp/gif) or PDF)
{"text": "extracted text", "engine": "tesseract" | "claude" | "pdf-text", "warning": null | "string"}
The frontend puts `text` into the paste textarea so the user can check/correct it before translating.

## POST /api/translate   (multipart/form-data: document_text, reading_level = "5th" | "8th" | "adult")
{
  "source_text": "...",
  "reading_level": "8th",
  "model": "claude-sonnet-5-5",
  "explanation": "Markdown. ## section headers. May contain citation markers like [1] or [2, 3] that refer to references[].id",
  "concepts": [{"term": "hyperkalemia", "kind": "condition" | "test" | "medication" | "procedure" | "other"}],
  "lab_values": [{
      "name": "Potassium", "canonical": "potassium" | null,
      "value": 5.8, "value_text": "5.8", "unit": "mmol/L" | null,
      "ref_low": 3.5 | null, "ref_high": 5.0 | null, "range_text": "3.5-5.0" | null,
      "range_source": "report" | "typical" | "report_flag" | null,
      "flag": "high" | "low" | "normal" | "unknown"}],
  "references": [{"id": 1, "title": "Potassium Blood Test", "url": "https://medlineplus.gov/...",
      "snippet": "one or two sentences", "kind": "lab_test" | "topic" | "drug",
      "matched": "Potassium", "cited": true}],
  "learn_more": [{"title": "...", "url": "...", "snippet": "...", "kind": "topic" | "drug" | "lab_test", "matched": "ACE inhibitor"}],
  "faithfulness": {
    "numeric": {"unsupported_numbers": ["12"], "numbers_from_reference_ranges": ["70", "99"],
                "total_numbers_in_explanation": 14, "numeric_faithfulness_score": 0.929},
    "judge": {"total_claims_checked": 12, "supported_count": 11,
              "unsupported_claims": [{"sentence": "...", "kind": "patient_specific" | "general", "reason": "..."}],
              "error": null | "string"}
  },
  "readability": {"source_grade_level": 11.2, "explanation_grade_level": 6.9},
  "timing_ms": {"total": 18234}
}
range_source meanings: "report" = range printed on the document; "typical" = app's fallback adult range
(document had none); "report_flag" = no range, flag taken from an H/L marker printed on the report.
