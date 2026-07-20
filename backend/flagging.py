"""
Deterministic lab-value flagging.

Design principle for this whole project: facts and judgments about what is
"high" or "low" are computed here, in plain code, never left to the model.
The LLM only ever explains a flag that code has already decided on.
"""

import re
from dataclasses import dataclass
from typing import Optional

# Fallback reference ranges for common tests, used only if a document
# doesn't print its own range. Values are illustrative adult ranges —
# real ranges vary by lab/method, which is exactly why we prefer the
# range printed in the source document when available.
COMMON_RANGES = {
    "glucose":        (70, 99, "mg/dL"),
    "hemoglobin":      (13.5, 17.5, "g/dL"),
    "hematocrit":      (38.8, 50.0, "%"),
    "wbc":             (4.5, 11.0, "x10^3/uL"),
    "platelets":       (150, 450, "x10^3/uL"),
    "sodium":          (135, 145, "mmol/L"),
    "potassium":       (3.5, 5.0, "mmol/L"),
    "chloride":        (96, 106, "mmol/L"),
    "creatinine":      (0.6, 1.3, "mg/dL"),
    "bun":             (7, 20, "mg/dL"),
    "calcium":         (8.5, 10.5, "mg/dL"),
    "total cholesterol": (0, 200, "mg/dL"),
    "ldl":             (0, 100, "mg/dL"),
    "hdl":             (40, 60, "mg/dL"),
    "triglycerides":   (0, 150, "mg/dL"),
    "tsh":             (0.4, 4.0, "mIU/L"),
    "alt":             (7, 56, "U/L"),
    "ast":             (10, 40, "U/L"),
}


@dataclass
class LabValue:
    name: str
    value: float
    unit: Optional[str]
    ref_low: Optional[float]
    ref_high: Optional[float]
    flag: str  # "high" | "low" | "normal" | "unknown"


# Matches lines roughly like: "Potassium 5.8 mmol/L (3.5-5.0)"
LAB_LINE_RE = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9 \-/]{2,30}?)\s*[:\-]?\s*"
    r"(?P<value>\d+\.?\d*)\s*"
    r"(?P<unit>mg/dL|g/dL|mmol/L|mIU/L|U/L|x10\^3/uL|%)?\s*"
    r"(?:\(?\s*(?P<low>\d+\.?\d*)\s*[-–]\s*(?P<high>\d+\.?\d*)\s*\)?)?",
    re.IGNORECASE,
)


def parse_lab_values(text: str) -> list[LabValue]:
    """
    Best-effort structured extraction of lab lines from raw text.
    This is intentionally conservative: it's fine to miss an oddly
    formatted line (it'll just be explained as prose by the LLM),
    but it should not fabricate a value that isn't in the text.
    """
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) > 120:
            continue
        m = LAB_LINE_RE.search(line)
        if not m or not m.group("value"):
            continue

        name_raw = m.group("name").strip().lower()
        name_key = name_raw
        value = float(m.group("value"))
        unit = m.group("unit")

        ref_low = float(m.group("low")) if m.group("low") else None
        ref_high = float(m.group("high")) if m.group("high") else None

        if ref_low is None or ref_high is None:
            fallback = COMMON_RANGES.get(name_key)
            if fallback:
                ref_low, ref_high, fallback_unit = fallback
                unit = unit or fallback_unit

        flag = "unknown"
        if ref_low is not None and ref_high is not None:
            if value < ref_low:
                flag = "low"
            elif value > ref_high:
                flag = "high"
            else:
                flag = "normal"

        # Skip lines that don't look like a real lab result (avoids
        # matching random numbers in narrative text)
        if flag == "unknown" and name_key not in COMMON_RANGES:
            continue

        results.append(
            LabValue(
                name=m.group("name").strip(),
                value=value,
                unit=unit,
                ref_low=ref_low,
                ref_high=ref_high,
                flag=flag,
            )
        )
    return results


def flags_to_prompt_context(labs: list[LabValue]) -> str:
    """Turn flagged values into ground-truth context injected into the LLM prompt."""
    if not labs:
        return "No structured lab values were detected in this document."
    lines = []
    for lab in labs:
        range_str = (
            f"ref {lab.ref_low}-{lab.ref_high}" if lab.ref_low is not None else "ref unknown"
        )
        lines.append(
            f"- {lab.name}: {lab.value} {lab.unit or ''} ({range_str}) -> FLAG: {lab.flag.upper()}"
        )
    return "\n".join(lines)
