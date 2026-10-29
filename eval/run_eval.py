"""
Evaluation harness: full pipeline vs. a naive one-line prompt.

Usage:
    python eval/run_eval.py                 # all samples, 8th-grade level
    python eval/run_eval.py --limit 3       # quick smoke run
    python eval/run_eval.py --level 5th

Every .txt file in samples/ goes through:
  - the FULL pipeline (lab flagging + MedlinePlus grounding + structured
    prompt + verification) — exactly what the web app runs, via pipeline.py
  - a NAIVE baseline: "Explain this medical document simply for a patient."

Both outputs are scored the same way:
  - reading grade level (Flesch-Kincaid; lower = easier)
  - numeric faithfulness (share of numbers that trace back to the document)
  - unsupported claims about the patient, found by the same LLM judge
  - abnormal-result coverage (share of flagged labs the explanation mentions)
  - whether the "not medical advice" disclaimer is present

Results are printed and written to eval/results.md and eval/results.json.
Running it calls Claude ~4 times per document, so it costs real money
(roughly $0.05 per document with claude-sonnet-5-5).
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
load_dotenv(ROOT / "backend" / ".env")

import llm  # noqa: E402
from faithfulness import llm_judge_faithfulness, numeric_faithfulness  # noqa: E402
from flagging import LAB_TESTS, flags_to_prompt_context, parse_lab_values  # noqa: E402
from pipeline import grade_level, run_pipeline  # noqa: E402

NAIVE_SYSTEM = "Explain this medical document simply for a patient."


def run_naive_baseline(doc_text: str) -> str:
    return llm.ask_claude(NAIVE_SYSTEM, doc_text, effort="low").text


OCR_DIGIT_FIXES = [str.maketrans({"0": "o", "1": "i"}), str.maketrans({"0": "o", "1": "l"})]
ABBREVIATIONS = {"sat": "saturation", "ab": "antibodies", "abs": "antibodies"}


def lab_terms(lab) -> list[str]:
    """Names a reader would recognise for a lab: as printed, plus known aliases."""
    name = lab.name.lower()
    terms = {name}
    # "Potass1um" / "Gluc0se": OCR digits inside a name; the explanation spells it correctly.
    terms.update(name.translate(table) for table in OCR_DIGIT_FIXES)
    # "Thyroid Peroxidase (TPO) Antibodies": also try without and only the bracketed part.
    without_brackets = re.sub(r"\s*\([^)]*\)", "", name).strip()
    terms.add(without_brackets)
    terms.update(inner.strip() for inner in re.findall(r"\(([^)]*)\)", name))
    words = without_brackets.replace(",", " ").split()
    terms.add(" ".join(ABBREVIATIONS.get(w, w) for w in words))
    if len(words) > 1 and re.search(r"\(([^)]*)\)", name):
        terms.add(" ".join([re.findall(r"\(([^)]*)\)", name)[0]] + words[-1:]))  # "tpo antibodies"
    test = LAB_TESTS.get(lab.canonical) if lab.canonical else None
    if test is not None:
        terms.update(a.lower() for a in getattr(test, "aliases", []) if len(a) > 2)
        display = getattr(test, "display", None) or getattr(test, "display_name", None)
        if display:
            terms.add(display.lower())
    return sorted(t for t in terms if len(t) > 2)


def abnormal_coverage(labs, explanation: str):
    abnormal = [lab for lab in labs if lab.flag in ("high", "low")]
    if not abnormal:
        return None
    text = explanation.lower()
    mentioned = sum(
        any(re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", text) for term in lab_terms(lab))
        for lab in abnormal
    )
    return round(mentioned / len(abnormal), 3)


def has_disclaimer(text: str) -> bool:
    """Same test for both outputs: says it isn't medical advice, or points to a clinician."""
    text = text.lower()
    return "medical advice" in text or "talk to your doctor" in text or "ask your doctor" in text


def patient_claims_unsupported(judge: dict) -> int:
    return sum(1 for c in judge.get("unsupported_claims", []) if c.get("kind") == "patient_specific")


def evaluate_document(path: Path, level: str) -> dict:
    doc_text = path.read_text(encoding="utf-8")
    labs = parse_lab_values(doc_text)
    lab_context = flags_to_prompt_context(labs)

    started = time.perf_counter()
    full = run_pipeline(doc_text, level)
    full_seconds = time.perf_counter() - started

    naive_text = run_naive_baseline(doc_text)
    naive_judge = llm_judge_faithfulness(doc_text, naive_text, lab_context=lab_context)
    naive_numeric = numeric_faithfulness(doc_text, naive_text, lab_context)

    full_judge = full["faithfulness"]["judge"]
    full_numeric = full["faithfulness"]["numeric"]
    return {
        "file": path.name,
        "source_grade": grade_level(doc_text),
        "full_grade": full["readability"]["explanation_grade_level"],
        "naive_grade": grade_level(naive_text),
        "full_numeric": full_numeric["numeric_faithfulness_score"],
        "naive_numeric": naive_numeric["numeric_faithfulness_score"],
        "full_unsupported_numbers": full_numeric["unsupported_numbers"],
        "naive_unsupported_numbers": naive_numeric["unsupported_numbers"],
        "full_patient_unsupported": patient_claims_unsupported(full_judge),
        "naive_patient_unsupported": patient_claims_unsupported(naive_judge),
        "full_general_unsupported": len(full_judge.get("unsupported_claims", [])) - patient_claims_unsupported(full_judge),
        "full_claims_checked": full_judge.get("total_claims_checked", 0),
        "naive_claims_checked": naive_judge.get("total_claims_checked", 0),
        "judge_errors": [e for e in (full_judge.get("error"), naive_judge.get("error")) if e],
        "abnormal_labs": sum(1 for lab in labs if lab.flag in ("high", "low")),
        "full_abnormal_coverage": abnormal_coverage(labs, full["explanation"]),
        "naive_abnormal_coverage": abnormal_coverage(labs, naive_text),
        "full_disclaimer": has_disclaimer(full["explanation"]),
        "naive_disclaimer": has_disclaimer(naive_text),
        "references": len(full["references"]),
        "citations_used": sum(1 for r in full["references"] if r["cited"]),
        "full_seconds": round(full_seconds, 1),
        "full_output_tokens": full["usage"].get("output_tokens", 0),
        "full_explanation": full["explanation"],
        "naive_explanation": naive_text,
        "full_unsupported_claims": full_judge.get("unsupported_claims", []),
        "naive_unsupported_claims": naive_judge.get("unsupported_claims", []),
    }


def mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def pct(value):
    return "n/a" if value is None else f"{value * 100:.0f}%"


def summarize(results: list[dict]) -> dict:
    return {
        "documents": len(results),
        "source_grade": mean(r["source_grade"] for r in results),
        "full_grade": mean(r["full_grade"] for r in results),
        "naive_grade": mean(r["naive_grade"] for r in results),
        "full_numeric": mean(r["full_numeric"] for r in results),
        "naive_numeric": mean(r["naive_numeric"] for r in results),
        "full_patient_unsupported": sum(r["full_patient_unsupported"] for r in results),
        "naive_patient_unsupported": sum(r["naive_patient_unsupported"] for r in results),
        "full_general_unsupported": sum(r["full_general_unsupported"] for r in results),
        "full_claims_checked": sum(r["full_claims_checked"] for r in results),
        "naive_claims_checked": sum(r["naive_claims_checked"] for r in results),
        "full_abnormal_coverage": mean(r["full_abnormal_coverage"] for r in results),
        "naive_abnormal_coverage": mean(r["naive_abnormal_coverage"] for r in results),
        "full_disclaimer": mean(1.0 if r["full_disclaimer"] else 0.0 for r in results),
        "naive_disclaimer": mean(1.0 if r["naive_disclaimer"] else 0.0 for r in results),
        "docs_with_citations": sum(1 for r in results if r["citations_used"] > 0),
        "avg_seconds": mean(r["full_seconds"] for r in results),
    }


def markdown_report(results: list[dict], summary: dict, level: str) -> str:
    lines = [
        f"# Eval results — {date.today().isoformat()}",
        "",
        f"Model: `{llm.MODEL}` · reading level: {level} · documents: {summary['documents']}",
        "",
        "## Summary",
        "",
        "| Metric | Full pipeline | Naive prompt |",
        "|---|---|---|",
        f"| Reading grade level (source: {summary['source_grade']:.1f}) | {summary['full_grade']:.1f} | {summary['naive_grade']:.1f} |",
        f"| Numeric faithfulness | {pct(summary['full_numeric'])} | {pct(summary['naive_numeric'])} |",
        f"| Unsupported claims about the patient | {summary['full_patient_unsupported']} of {summary['full_claims_checked']} checked "
        f"| {summary['naive_patient_unsupported']} of {summary['naive_claims_checked']} checked |",
        f"| Abnormal results mentioned | {pct(summary['full_abnormal_coverage'])} | {pct(summary['naive_abnormal_coverage'])} |",
        f"| Disclaimer present | {pct(summary['full_disclaimer'])} | {pct(summary['naive_disclaimer'])} |",
        f"| Docs with MedlinePlus citations | {summary['docs_with_citations']}/{summary['documents']} | — |",
        "",
        f"Full pipeline also flagged {summary['full_general_unsupported']} general-background claims "
        f"as not backed by its references. Average full-pipeline time: {summary['avg_seconds']:.1f}s per document.",
        "",
        "## Per document",
        "",
        "| File | Src grade | Full grade | Naive grade | Full numeric | Naive numeric | Full / naive unsupported patient claims | Abnormal covered (full / naive) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['file']} | {r['source_grade']} | {r['full_grade']} | {r['naive_grade']} | "
            f"{pct(r['full_numeric'])} | {pct(r['naive_numeric'])} | "
            f"{r['full_patient_unsupported']} / {r['naive_patient_unsupported']} | "
            f"{pct(r['full_abnormal_coverage'])} / {pct(r['naive_abnormal_coverage'])} |"
        )
    judge_errors = [r for r in results if r["judge_errors"]]
    if judge_errors or summary.get("failed_documents"):
        lines += ["", "## Problems during the run", ""]
        for r in judge_errors:
            lines.append(f"- `{r['file']}` — fact-check didn't run: {'; '.join(r['judge_errors'])} (counted as 0 unsupported)")
        for name in summary.get("failed_documents", []):
            lines.append(f"- `{name}` — failed and is excluded from the tables")
    flagged_numbers = [r for r in results if r["full_unsupported_numbers"] or r["naive_unsupported_numbers"]]
    if flagged_numbers:
        lines += ["", "## Numbers not found in the source", ""]
        for r in flagged_numbers:
            lines.append(
                f"- `{r['file']}` — full: {', '.join(r['full_unsupported_numbers']) or 'none'}; "
                f"naive: {', '.join(r['naive_unsupported_numbers']) or 'none'}"
            )
    return "\n".join(lines) + "\n"


def rescore(out: Path) -> None:
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    for r in payload["results"]:
        doc_text = (ROOT / "samples" / r["file"]).read_text(encoding="utf-8")
        labs = parse_lab_values(doc_text)
        lab_context = flags_to_prompt_context(labs)
        for side in ("full", "naive"):
            explanation = r[f"{side}_explanation"]
            numeric = numeric_faithfulness(doc_text, explanation, lab_context)
            r[f"{side}_grade"] = grade_level(explanation)
            r[f"{side}_numeric"] = numeric["numeric_faithfulness_score"]
            r[f"{side}_unsupported_numbers"] = numeric["unsupported_numbers"]
            r[f"{side}_abnormal_coverage"] = abnormal_coverage(labs, explanation)
            r[f"{side}_disclaimer"] = has_disclaimer(explanation)
        r["source_grade"] = grade_level(doc_text)
        r["abnormal_labs"] = sum(1 for lab in labs if lab.flag in ("high", "low"))
    summary = summarize(payload["results"])
    summary["failed_documents"] = payload["summary"].get("failed_documents", [])
    payload["summary"] = summary
    report = markdown_report(payload["results"], summary, payload["level"])
    (out / "results.md").write_text(report, encoding="utf-8")
    (out / "results.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", default="8th", choices=["5th", "8th", "adult"])
    parser.add_argument("--limit", type=int, default=0, help="only run the first N samples")
    parser.add_argument("--only", default="", help="only run samples whose filename contains this text")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=str(ROOT / "eval"), help="folder for results.md / results.json")
    parser.add_argument("--rescore", action="store_true",
                        help="re-score the saved outputs in results.json without calling Claude "
                             "(recomputes grade level, numbers, abnormal coverage and disclaimer)")
    args = parser.parse_args()

    if args.rescore:
        return rescore(Path(args.out))

    if not llm.api_key_configured():
        sys.exit("Set ANTHROPIC_API_KEY in backend/.env first.")

    paths = sorted((ROOT / "samples").glob("*.txt"))
    if args.only:
        paths = [p for p in paths if args.only in p.name]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        sys.exit("No sample documents found in samples/.")

    print(f"Evaluating {len(paths)} documents with {llm.MODEL} ({args.workers} at a time)...")
    def safe_evaluate(path):
        try:
            return evaluate_document(path, args.level)
        except Exception as e:  # keep the documents that did finish (they're paid for)
            print(f"  ! {path.name} failed: {e}")
            return {"file": path.name, "error": str(e)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(safe_evaluate, paths))
    results = [r for r in outcomes if "error" not in r]
    failed = [r for r in outcomes if "error" in r]
    if not results:
        sys.exit("Every document failed; nothing to report.")

    summary = summarize(results)
    summary["failed_documents"] = [f["file"] for f in failed]
    report = markdown_report(results, summary, args.level)
    print(report)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text(report, encoding="utf-8")
    payload = {"model": llm.MODEL, "level": args.level, "date": date.today().isoformat(), "summary": summary, "results": results, "failed": failed}
    (out / "results.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"Wrote {out / 'results.md'} and {out / 'results.json'}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
