"""
Faithfulness verification, run AFTER generation.

Two layers, deliberately in this order:
1. Deterministic numeric check (no AI, catches the worst hallucination class).
2. LLM-as-judge sentence-level check (catches subtler unsupported claims).
"""

import re
from decimal import Decimal, InvalidOperation

import anthropic

import llm
from prompts import JUDGE_SCHEMA, build_judge_messages

# A number like 5, 5.8, 1,000 or 1,250.5. A trailing period ("22.") is not part
# of the number, which matters because sentences often end in a value.
NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")

# Citation markers like [2], [1, 3] or [1-3] point at references, they aren't values.
CITATION_RE = re.compile(r"\[\s*\d+(?:\s*[-–—,]\s*\d+)*\s*\]")
# Markdown numbered-list markers ("1. Take...", "2) Call...") number steps, they aren't values.
LIST_MARKER_RE = re.compile(r"^[ \t]*\d+[.)][ \t]+", re.MULTILINE)


def normalize_number(raw: str) -> str:
    """'5.0' and '5' are the same value, as are '07' and '7' and '1,000' and '1000'."""
    try:
        value = Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return raw
    normalized = value.normalize()
    # Decimal("100").normalize() is 1E+2; format it back as plain digits.
    return format(normalized, "f")


def extract_numbers(text: str, split_lists: bool = False) -> set[str]:
    numbers = set()
    for raw in NUMBER_RE.findall(text):
        numbers.add(normalize_number(raw))
        if split_lists and "," in raw:
            # "180,210,240" in a source may be three readings rather than 180210240.
            numbers.update(normalize_number(part) for part in raw.split(","))
    return numbers


def explanation_numbers(explanation: str) -> set[str]:
    """Numbers a reader would see as values (citations and list markers removed)."""
    text = LIST_MARKER_RE.sub("", CITATION_RE.sub(" ", explanation))
    return extract_numbers(text)


def numeric_faithfulness(source: str, explanation: str, lab_context: str = "") -> dict:
    """
    Every number in the explanation must appear in the source document.

    Numbers that only appear in `lab_context` (the typical adult ranges the app
    adds when a report prints no range) are allowed but reported separately, so
    the reader can tell they didn't come from their own document.
    """
    source_numbers = extract_numbers(source, split_lists=True)
    context_numbers = extract_numbers(lab_context) - source_numbers
    used_numbers = explanation_numbers(explanation)

    from_ranges = used_numbers & context_numbers
    unsupported = used_numbers - source_numbers - context_numbers
    total = len(used_numbers)
    return {
        "unsupported_numbers": sorted(unsupported, key=_sort_key),
        "numbers_from_reference_ranges": sorted(from_ranges, key=_sort_key),
        "total_numbers_in_explanation": total,
        "numeric_faithfulness_score": (
            1.0 if total == 0 else round(1 - len(unsupported) / total, 3)
        ),
    }


def _sort_key(number: str) -> float:
    try:
        return float(number)
    except ValueError:
        return float("inf")


def llm_judge_faithfulness(source: str, explanation: str, references=(), lab_context: str = "") -> dict:
    """Ask Claude to check each claim. Fails open with a visible error, never silently."""
    system, user = build_judge_messages(source, explanation, references, lab_context)
    try:
        result = llm.ask_claude(system, user, schema=JUDGE_SCHEMA, effort="medium")
    except llm.LLMError as e:
        return _judge_error(str(e))
    except anthropic.APIError:
        # The explanation already exists; don't throw it away because the
        # second call failed. Show the user that the check didn't run instead.
        return _judge_error("The fact-checking pass couldn't reach Claude, so it didn't run.")
    data = result.data
    data["error"] = None
    return data


def _judge_error(message: str) -> dict:
    return {
        "total_claims_checked": 0,
        "supported_count": 0,
        "unsupported_claims": [],
        "error": message,
    }


def run_faithfulness_check(source: str, explanation: str, lab_context: str = "", references=()) -> dict:
    numeric = numeric_faithfulness(source, explanation, lab_context)
    judged = llm_judge_faithfulness(source, explanation, references, lab_context)
    return {"numeric": numeric, "judge": judged}
