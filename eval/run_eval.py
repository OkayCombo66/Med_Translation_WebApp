"""
Evaluation harness.

Usage:
    python eval/run_eval.py

Add more synthetic documents to samples/ (one .txt file each) as you build
out your test set — aim for 15-25 covering discharge summaries, lab panels,
and radiology reports, including a few with deliberately tricky formatting.

This script produces the comparison table that's the centerpiece of the
portfolio writeup: full pipeline vs. a naive one-line prompt, on readability
and numeric faithfulness.
"""

import os
import sys
import glob
import textstat
from dotenv import load_dotenv
from anthropic import Anthropic

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from flagging import parse_lab_values, flags_to_prompt_context  # noqa: E402
from prompts import build_translation_messages  # noqa: E402
from faithfulness import numeric_faithfulness  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

MODEL = "claude-sonnet-4-6"
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

NAIVE_SYSTEM = "Explain this medical document simply for a patient."


def run_full_pipeline(doc_text: str) -> str:
    labs = parse_lab_values(doc_text)
    lab_context = flags_to_prompt_context(labs)
    system, user = build_translation_messages(doc_text, lab_context, "8th")
    resp = client.messages.create(
        model=MODEL, max_tokens=1200, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def run_naive_baseline(doc_text: str) -> str:
    resp = client.messages.create(
        model=MODEL, max_tokens=1200, system=NAIVE_SYSTEM,
        messages=[{"role": "user", "content": doc_text}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def evaluate_document(path: str) -> dict:
    with open(path) as f:
        doc_text = f.read()

    full_output = run_full_pipeline(doc_text)
    naive_output = run_naive_baseline(doc_text)

    full_numeric = numeric_faithfulness(doc_text, full_output)
    naive_numeric = numeric_faithfulness(doc_text, naive_output)

    return {
        "file": os.path.basename(path),
        "source_grade": round(textstat.flesch_kincaid_grade(doc_text), 1),
        "full_grade": round(textstat.flesch_kincaid_grade(full_output), 1),
        "naive_grade": round(textstat.flesch_kincaid_grade(naive_output), 1),
        "full_faithfulness": full_numeric["numeric_faithfulness_score"],
        "naive_faithfulness": naive_numeric["numeric_faithfulness_score"],
    }


def main():
    sample_files = sorted(glob.glob(os.path.join(
        os.path.dirname(__file__), "..", "samples", "*.txt")))
    if not sample_files:
        print("No sample documents found in samples/. Add some .txt files first.")
        return

    results = [evaluate_document(p) for p in sample_files]

    header = f"{'file':<28}{'src grade':>10}{'full grade':>12}{'naive grade':>13}{'full faith':>12}{'naive faith':>13}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['file']:<28}{r['source_grade']:>10}{r['full_grade']:>12}"
            f"{r['naive_grade']:>13}{r['full_faithfulness']:>12}{r['naive_faithfulness']:>13}"
        )

    avg = lambda k: round(sum(r[k] for r in results) / len(results), 3)
    print("-" * len(header))
    print(
        f"{'AVERAGE':<28}{avg('source_grade'):>10}{avg('full_grade'):>12}"
        f"{avg('naive_grade'):>13}{avg('full_faithfulness'):>12}{avg('naive_faithfulness'):>13}"
    )


if __name__ == "__main__":
    main()
