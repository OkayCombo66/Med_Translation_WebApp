"""
Faithfulness verification, run AFTER generation.

Two layers, deliberately in this order:
1. Deterministic numeric check (no AI, catches the worst hallucination class).
2. LLM-as-judge sentence-level check (catches subtler unsupported claims).
"""

import re
import json
from prompts import build_judge_messages

NUMBER_RE = re.compile(r"\d+\.?\d*")


def extract_numbers(text: str) -> set[str]:
    return set(NUMBER_RE.findall(text))


def numeric_faithfulness(source: str, explanation: str) -> dict:
    source_numbers = extract_numbers(source)
    explanation_numbers = extract_numbers(explanation)
    unsupported = explanation_numbers - source_numbers
    total = len(explanation_numbers)
    return {
        "unsupported_numbers": sorted(unsupported),
        "total_numbers_in_explanation": total,
        "numeric_faithfulness_score": (
            1.0 if total == 0 else round(1 - len(unsupported) / total, 3)
        ),
    }


def llm_judge_faithfulness(client, model: str, source: str, explanation: str) -> dict:
    system, user = build_judge_messages(source, explanation)
    resp = client.messages.create(
        model=model,
        max_tokens=800,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(block.text for block in resp.content if block.type == "text")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Model didn't return clean JSON — fail open with a visible marker
        # rather than silently trusting the output.
        parsed = {
            "unsupported_claims": [],
            "supported_count": 0,
            "total_claims_checked": 0,
            "judge_parse_error": True,
            "raw_response": raw,
        }
    return parsed


def run_faithfulness_check(client, model: str, source: str, explanation: str) -> dict:
    numeric = numeric_faithfulness(source, explanation)
    judged = llm_judge_faithfulness(client, model, source, explanation)
    return {"numeric": numeric, "judge": judged}
