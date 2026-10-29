"""
The full translation pipeline, shared by the web API (main.py) and the eval
(eval/run_eval.py) so both always measure exactly the same thing.

    document text
      → deterministic lab parsing + flagging          (flagging.py, no AI)
      → MedlinePlus retrieval for what the doc mentions (medlineplus.py, no AI)
      → Claude translation, grounded on both            (prompts.py + llm.py)
      → "learn more" links for the concepts it used     (medlineplus.py)
      → faithfulness check: numbers + LLM judge         (faithfulness.py)
"""

import re
import time
from dataclasses import asdict

import textstat

import llm
from faithfulness import run_faithfulness_check
from flagging import flags_to_prompt_context, parse_lab_values
from medlineplus import find_references, learn_more_for_concepts
from prompts import TRANSLATION_SCHEMA, build_translation_messages

MARKDOWN_RE = re.compile(r"\[\s*\d+(?:\s*[-–—,]\s*\d+)*\s*\]|[#*_`>]|^\s*(?:[-•+]|\d+[.)])\s+", re.MULTILINE)
CITATION_GROUP_RE = re.compile(r"\[\s*(\d+(?:\s*[-–—,]\s*\d+)*)\s*\]")
STRUCTURAL_LINE_RE = re.compile(r"^\s*(?:#{1,6}\s|[-*+•]\s|\d+[.)]\s)")


def _end_lines_as_sentences(text: str) -> str:
    """Headers, bullets, table rows and paragraph ends read as separate sentences,
    but readability formulas only split on . ! ? — so add a period where a line
    ends a unit and has no punctuation of its own."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        ends_unit = STRUCTURAL_LINE_RE.match(line) or not next_line.strip() or STRUCTURAL_LINE_RE.match(next_line)
        stripped = line.rstrip()
        if stripped.strip() and ends_unit and stripped[-1] not in ".!?:;":
            stripped += "."
        out.append(stripped)
    return "\n".join(out)


def grade_level(text: str) -> float:
    """Flesch-Kincaid grade of the text as a reader sees it (no Markdown symbols)."""
    plain = MARKDOWN_RE.sub("", _end_lines_as_sentences(text))
    return round(textstat.flesch_kincaid_grade(plain), 1) if plain.strip() else 0.0


def cited_reference_ids(explanation: str, reference_count: int) -> set[int]:
    """Reference numbers cited as [2], [1, 3] or [1-3]."""
    cited = set()
    for group in CITATION_GROUP_RE.findall(explanation):
        for part in re.split(r"\s*,\s*", group):
            bounds = [int(n) for n in re.split(r"\s*[-–—]\s*", part)]
            for n in range(bounds[0], bounds[-1] + 1):
                if 1 <= n <= reference_count:
                    cited.add(n)
    return cited


def reference_to_dict(ref, **extra) -> dict:
    return {
        "title": ref.title,
        "url": ref.url,
        "snippet": ref.snippet,
        "kind": ref.kind,
        "matched": ref.matched,
        **extra,
    }


def run_pipeline(document_text: str, reading_level: str = "8th", check: bool = True) -> dict:
    """Run every step and return the JSON described in API_CONTRACT.md.

    Raises llm.LLMError or anthropic API errors if the translation call fails;
    callers decide how to report those.
    """
    started = time.perf_counter()

    # 1. Deterministic lab parsing + flagging (never trust the LLM with this)
    labs = parse_lab_values(document_text)
    lab_context = flags_to_prompt_context(labs)

    # 2. Retrieval: MedlinePlus passages for the abnormal tests, conditions and
    #    medicines found in the document. These ground the general explanations.
    references = find_references(document_text, labs)

    # 3. Translation call (structured output: explanation + concept list)
    system, user = build_translation_messages(document_text, lab_context, reading_level, references)
    result = llm.ask_claude(system, user, schema=TRANSLATION_SCHEMA, effort="low")
    explanation = result.data["explanation"].strip()
    concepts = result.data["concepts"]

    # 4. "Learn more" links for concepts the explanation mentions that weren't
    #    already covered by the references.
    learn_more = learn_more_for_concepts(concepts, exclude_urls={r.url for r in references}, document_text=document_text)

    # 5. Faithfulness check (deterministic numeric + LLM judge)
    faithfulness = run_faithfulness_check(document_text, explanation, lab_context, references) if check else None

    cited = cited_reference_ids(explanation, len(references))
    return {
        "source_text": document_text,
        "reading_level": reading_level,
        "model": result.model,
        "explanation": explanation,
        "concepts": concepts,
        "lab_values": [asdict(lab) for lab in labs],
        "references": [
            reference_to_dict(ref, id=i, cited=i in cited) for i, ref in enumerate(references, start=1)
        ],
        "learn_more": [reference_to_dict(ref) for ref in learn_more],
        "faithfulness": faithfulness,
        "readability": {
            "source_grade_level": grade_level(document_text),
            "explanation_grade_level": grade_level(explanation),
        },
        "usage": result.usage,
        "timing_ms": {"total": round((time.perf_counter() - started) * 1000)},
    }
