"""
Deterministic lab-value parsing and flagging.

Design principle for this whole project: facts and judgments about what is
"high" or "low" are computed here, in plain code, never left to the model.
The LLM only ever explains a flag that code has already decided on.

How it works
------------
We read the document one line at a time and only trust lines that look like a
row of a results table:

    Potassium      5.8   H   mmol/L    (3.5-5.0)
    ^ name at the  ^ value, then an optional H/L marker, unit and
      line start     reference range, in any order

1. _clean_line() tidies the line (pipe tables such as "| Potassium | 5.4 |"
   become plain columns).
2. _LINE_RE matches "<name> <value>" at the very START of the line. Every word
   of the name must start with a letter, so the name can never swallow the
   value, and "Hemoglobin A1c 6.8" can't be misread as "Hemoglobin 1"
   ("A1c" starts with a letter, so it is part of the name).
3. _parse_tail() reads what follows the value: an H/L marker, a unit and a
   reference range. If the text right after the value is none of those
   ("in 2 weeks", "Glucose 2 hr post meal"), the line is rejected.
4. canonical_test_key() looks the name up in LAB_TESTS.
5. The flag comes from, in order of preference:
     a) the range printed on the report          -> range_source "report"
     b) an H/L marker printed on the report      -> range_source "report_flag"
     c) our typical adult range, if safe to use  -> range_source "typical"
     d) nothing                                  -> flag "unknown"

Being conservative is the point. Missing an oddly formatted line is fine (the
LLM will still explain it as prose); inventing a value, or a flag the document
doesn't support, is not.
"""

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass
class LabValue:
    """One lab result found in the document.

    API_CONTRACT.md's lab_values[] objects are exactly dataclasses.asdict(LabValue),
    so don't rename or reorder these fields without updating the contract.
    """
    name: str                    # test name as printed, e.g. "Potassium"
    canonical: Optional[str]     # key into LAB_TESTS, or None for an unrecognised test
    value: float
    value_text: str              # value exactly as printed, incl. comparator: "5.8", "0.60", "<0.01"
    unit: Optional[str]
    ref_low: Optional[float]
    ref_high: Optional[float]
    range_text: Optional[str]    # range as printed ("3.5-5.0", "<200"), or our formatted typical range
    range_source: Optional[str]  # "report" | "typical" | "report_flag" | None
    flag: str                    # "high" | "low" | "normal" | "unknown"


@dataclass(frozen=True)
class LabTest:
    """What we know about one common lab test."""
    display_name: str
    aliases: tuple[str, ...] = ()
    # Typical adult range, used ONLY when the document prints no range of its own.
    # None on one side = one-sided range (LDL only has an upper limit).
    # None on both sides = we deliberately don't guess (the range depends on
    # sex, age, assay or therapy, so a generic range could mislead).
    typical_low: Optional[float] = None
    typical_high: Optional[float] = None
    typical_unit: Optional[str] = None
    # Other unit spellings that give the SAME numbers (mEq/L == mmol/L for sodium).
    # A typical range is only applied when the printed unit is one of these.
    other_units: tuple[str, ...] = ()
    # True when every common unit system gives the same numbers, so a value
    # printed WITHOUT a unit can still be compared with the typical range.
    unit_optional: bool = False


_MG_DL = "mg/dL"
_MMOL_L = "mmol/L"
_PER_UL = "x10^3/uL"
_PER_UL_OTHERS = ("10^3/uL", "K/uL", "x10^9/L", "10^9/L", "thou/uL", "K/mm3", "x10^3/mm3")

# Canonical key -> what we know about the test. Other modules depend on these
# exact keys (e.g. to link each test to its MedlinePlus page).
#
# Typical ranges are rounded, widely cited adult values. Real reference ranges
# vary by lab and method, which is exactly why the range printed on the
# document always wins.
LAB_TESTS: dict[str, LabTest] = {
    # --- Basic / comprehensive metabolic panel ---
    "glucose": LabTest(
        "Glucose",
        ("glu", "blood glucose", "blood sugar", "fasting blood sugar", "fbs", "glucose fasting"),
        70, 99, _MG_DL),
    "sodium": LabTest("Sodium", ("na",), 135, 145, _MMOL_L, ("mEq/L",), unit_optional=True),
    "potassium": LabTest("Potassium", ("k",), 3.5, 5.0, _MMOL_L, ("mEq/L",), unit_optional=True),
    "chloride": LabTest("Chloride", ("cl",), 96, 106, _MMOL_L, ("mEq/L",), unit_optional=True),
    "co2": LabTest(
        "Carbon dioxide",
        ("carbon dioxide", "carbon dioxide total", "bicarbonate", "hco3", "total co2", "co2 total", "tco2"),
        22, 29, _MMOL_L, ("mEq/L",), unit_optional=True),
    "bun": LabTest(
        "Blood urea nitrogen",
        ("blood urea nitrogen", "urea nitrogen", "urea nitrogen bun"),
        7, 20, _MG_DL),
    "creatinine": LabTest("Creatinine", ("creat", "cr")),  # sex-dependent: no typical range
    "egfr": LabTest(
        "eGFR",
        ("estimated gfr", "gfr estimated", "egfr ckd epi", "gfr"),
        typical_low=60, typical_unit="mL/min/1.73m2",
        other_units=("mL/min/1.73m^2", "mL/min/1.73", "mL/min"), unit_optional=True),
    "calcium": LabTest("Calcium", ("ca", "calcium total", "total calcium"), 8.5, 10.5, _MG_DL),
    "magnesium": LabTest("Magnesium", ("mg",), 1.7, 2.2, _MG_DL),
    "phosphorus": LabTest(
        "Phosphorus",
        ("phosphate", "phos", "inorganic phosphorus", "phosphorus inorganic"),
        2.5, 4.5, _MG_DL),
    "total_protein": LabTest("Total protein", ("protein total",), 6.0, 8.3, "g/dL"),
    "albumin": LabTest("Albumin", ("alb",), 3.4, 5.4, "g/dL"),
    "total_bilirubin": LabTest(
        "Total bilirubin",
        ("bilirubin total", "bilirubin", "t bili", "tbili", "total bili", "bili total"),
        0.1, 1.2, _MG_DL),
    "alk_phos": LabTest(
        "Alkaline phosphatase",
        ("alkaline phosphatase", "alk phosphatase", "alp", "alkp"),
        44, 147, "U/L", ("IU/L",)),
    "alt": LabTest(
        "ALT",
        ("alanine aminotransferase", "alanine transaminase", "sgpt", "alt sgpt"),
        7, 56, "U/L", ("IU/L",)),
    "ast": LabTest(
        "AST",
        ("aspartate aminotransferase", "aspartate transaminase", "sgot", "ast sgot"),
        10, 40, "U/L", ("IU/L",)),
    "anion_gap": LabTest("Anion gap"),  # assay-dependent: no typical range

    # --- Lipids (one-sided "desirable" limits) ---
    "total_cholesterol": LabTest(
        "Total cholesterol",
        ("cholesterol total", "cholesterol", "chol", "tc"),
        typical_high=200, typical_unit=_MG_DL),
    "ldl": LabTest(
        "LDL cholesterol",
        ("ldl cholesterol", "ldl c", "ldl chol", "ldl calculated", "ldl cholesterol calc",
         "low density lipoprotein"),
        typical_high=100, typical_unit=_MG_DL),
    # HDL only has a LOWER limit: a high HDL is never flagged with the typical range.
    "hdl": LabTest(
        "HDL cholesterol",
        ("hdl cholesterol", "hdl c", "hdl chol", "high density lipoprotein"),
        typical_low=40, typical_unit=_MG_DL),
    "triglycerides": LabTest(
        "Triglycerides",
        ("triglyceride", "trig", "trigs", "tg"),
        typical_high=150, typical_unit=_MG_DL),

    # --- Diabetes / thyroid ---
    "a1c": LabTest(
        "Hemoglobin A1c",
        ("hemoglobin a1c", "haemoglobin a1c", "hba1c", "hgb a1c", "hb a1c", "glycohemoglobin",
         "glycated hemoglobin", "glycosylated hemoglobin"),
        typical_high=5.7, typical_unit="%"),
    "tsh": LabTest(
        "TSH",
        ("thyroid stimulating hormone", "thyrotropin"),
        0.4, 4.0, "mIU/L", ("uIU/mL", "mU/L"), unit_optional=True),
    "free_t4": LabTest(
        "Free T4",
        ("t4 free", "ft4", "free thyroxine", "thyroxine free"),
        0.8, 1.8, "ng/dL"),

    # --- Complete blood count ---
    "wbc": LabTest(
        "White blood cell count",
        ("white blood cell count", "white blood cells", "white blood cell", "white count",
         "wbc count", "leukocytes", "leukocyte count"),
        4.5, 11.0, _PER_UL, _PER_UL_OTHERS),
    # RBC, hemoglobin and hematocrit ranges differ by sex: no typical range.
    "rbc": LabTest(
        "Red blood cell count",
        ("red blood cell count", "red blood cells", "red blood cell", "rbc count", "erythrocytes")),
    "hemoglobin": LabTest("Hemoglobin", ("haemoglobin", "hgb", "hb")),
    "hematocrit": LabTest("Hematocrit", ("haematocrit", "hct")),
    "mcv": LabTest(
        "MCV",
        ("mean corpuscular volume", "mean cell volume"),
        80, 100, "fL", ("um3",), unit_optional=True),
    "platelets": LabTest(
        "Platelets",
        ("platelet", "platelet count", "plt", "plts", "thrombocytes"),
        150, 450, _PER_UL, _PER_UL_OTHERS),

    # --- Other common tests: no typical range on purpose ---
    "ferritin": LabTest("Ferritin"),  # sex-dependent
    "inr": LabTest("INR", ("pt inr", "international normalized ratio")),  # depends on therapy
    "psa": LabTest("PSA", ("prostate specific antigen", "psa total", "total psa")),  # age-dependent
    "vitamin_d": LabTest(
        "Vitamin D",
        ("vit d", "vitamin d 25 oh", "25 oh vitamin d", "vitamin d 25 hydroxy", "25 hydroxy vitamin d",
         "25 hydroxyvitamin d", "vitamin d total")),
    "bnp": LabTest("BNP", ("b type natriuretic peptide", "brain natriuretic peptide")),
    "uacr": LabTest(
        "Urine albumin/creatinine ratio",
        ("urine albumin creatinine ratio", "microalbumin creatinine ratio", "albumin creatinine ratio",
         "urine microalbumin creatinine ratio", "albumin creatinine ratio urine", "acr"),
        typical_high=30, typical_unit="mg/g", other_units=("ug/mg", "mg/g creat", "mg/gCr")),
}


# ---------------------------------------------------------------------------
# Test-name lookup
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Lowercase and keep only letters/digits: 'LDL-C' -> 'ldl c', 'Glucose, Fasting' -> 'glucose fasting'."""
    return " ".join(re.findall(r"[a-z0-9]+", name.lower()))


def _build_alias_index() -> dict[str, str]:
    """Map every normalized alias (plus the key and display name) to its canonical key."""
    index: dict[str, str] = {}
    for key, test in LAB_TESTS.items():
        for alias in (key.replace("_", " "), test.display_name, *test.aliases):
            norm = _normalize_name(alias)
            if index.get(norm, key) != key:
                raise ValueError(f"Alias {alias!r} is used for both {index[norm]!r} and {key!r}")
            index[norm] = key
    return index


_ALIAS_INDEX = _build_alias_index()

# Words that describe the specimen or method rather than the test itself, so
# "Potassium, Serum" and "Glucose (fasting)" still find their test.
# ("direct" is deliberately NOT here: "Bilirubin, Direct" is a different test.)
_QUALIFIER_WORDS = {
    "serum", "plasma", "blood", "whole", "fasting", "random", "level", "calc", "calculated",
    "quant", "quantitative",
}


def canonical_test_key(name: str) -> Optional[str]:
    """Return the LAB_TESTS key for a printed test name, or None if unrecognised.

    Matching is exact on the WHOLE name (after normalizing case and punctuation),
    never a substring search, so short aliases like "K" can't match words inside
    a sentence ("her potassium was" is not a test name).
    """
    full = _normalize_name(name)
    without_parens = _normalize_name(re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", name))
    without_qualifiers = " ".join(w for w in without_parens.split() if w not in _QUALIFIER_WORDS)
    for candidate in (full, without_parens, without_qualifiers):
        if candidate in _ALIAS_INDEX:
            return _ALIAS_INDEX[candidate]
    return None


# ---------------------------------------------------------------------------
# Regex building blocks
# ---------------------------------------------------------------------------

# A number as printed: "1,250" (thousands separator) or "5.8" / "112" / "0.60".
_NUM = r"(?:[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)"
# Comparators that can appear in values ("<0.01") and one-sided ranges (">=60", "≥60").
_CMP = r"(?:<=|>=|=<|=>|[<>≤≥])"
_LESS_THAN = {"<", "<=", "=<", "≤"}
# Range separator: hyphen, the various Unicode dashes OCR produces, or the word "to".
_DASH = r"(?:\s*[-–—−‐‑]\s*|\s+to\s+)"

# Units. Matched case-insensitively.
_UNIT_ALTERNATIVES = [
    r"(?:x\s?)?10\s?(?:\^|\*|e)\s?[0-9]+\s?/\s?[a-zµμ]+",  # x10^3/uL, 10*3/uL, x10E3/uL, x 10^9/L
    r"ml/min/1\.73\s?m(?:2|²|\^2)",                         # eGFR unit, sometimes with a space before m2
    r"[a-zµμ][a-z0-9µμ.^*²]*(?:/[a-z0-9µμ.^*²]+)+",         # anything with a slash: mmol/L, ng/mL, K/uL
    r"%|fl|pg|sec|seconds|ratio|mmhg",                      # common units without a slash
]
_UNIT = "(?:" + "|".join(_UNIT_ALTERNATIVES) + r")(?=\s|$|[(),;])"
_UNIT_RE = re.compile(_UNIT, re.IGNORECASE)

# H/L markers printed next to a value: H, L, HH, (H), [L], H*, High, LOW, Hi ...
# A bare "*" or "!" means "abnormal" without saying which way.
_MARKER_RE = re.compile(
    r"(?:[(\[]?\*?(?:HH|LL|H|L)\*?[)\]]?"   # single letters: upper case only ("l" is often OCR for "1")
    r"|[(\[]?(?i:high|low|hi|lo)[)\]]?"     # words: any case
    r"|\*{1,2}|!)"
    r"(?=\s|$|\()"
)

# Optional label in front of a range: "Ref:", "Reference range:", "Normal", "Desirable" ...
_RANGE_LABEL = (
    r"(?:(?:(?:ref(?:erence)?\.?|normal|nr|desirable|optimal)(?:\s+(?:range|interval|values?))?|range)"
    r"\s*[:=]?\s*)"
)
# The range itself: "3.5-5.0" / "70 to 99" (two-sided) or "<200" / ">= 60" (one-sided).
_RANGE_CORE = (
    rf"(?:(?P<low>{_NUM}){_DASH}(?P<high>{_NUM})"
    rf"|(?P<cmp>{_CMP})\s?(?P<bound>{_NUM}))"
)
# "(3.5-5.0)", "[70-99]", "(ref <150)", "(3.5-5.0 mmol/L)", "(<200 desirable)"
_RANGE_PAREN_RE = re.compile(
    rf"[(\[]\s*{_RANGE_LABEL}?(?P<core>{_RANGE_CORE})(?:\s*(?P<unit>{_UNIT}))?[^()\[\]]{{0,25}}[)\]]",
    re.IGNORECASE,
)
# "3.5-5.0", "Ref: <5.7", "≥60" with no brackets (column tables)
_RANGE_BARE_RE = re.compile(rf"{_RANGE_LABEL}?(?P<core>{_RANGE_CORE})(?=\s|$|[,;])", re.IGNORECASE)

# One word of a test name. It must start with a letter, "(" + letter, or look
# like "25-OH". Because no name word can start with a plain digit, the name
# stops right before the value.
# A bracketed qualifier can hold spaces and digits: "eGFR (CKD-EPI 2021)".
_NAME_WORD = r"(?:[(\[][A-Za-z][^)\]]{0,30}[)\]]|(?:[A-Za-z]|[(\[][A-Za-z]|[0-9]+-[A-Za-z])[^\s:=]*)"

# "<name> <value>" at the start of a line, e.g. "Potassium: 5.8", "- K 5.9H", "Troponin I <0.01"
_LINE_RE = re.compile(
    r"^(?:[-*•·]\s*)?"                                       # optional bullet
    rf"(?P<name>{_NAME_WORD}(?:\s+{_NAME_WORD}){{0,5}})"     # 1-6 name words
    r"(?:\s*[:=]\s*|\s+[-–—]\s+|\s+)"                        # separator: ":", "=", " - " or spaces
    rf"(?P<value>(?:(?P<cmp>{_CMP})\s?)?(?P<num>{_NUM}))"    # the value, maybe with "<" / ">"
    r"(?P<pct>%)?"                                           # "6.8%"
    r"(?P<gmark>\*{1,2}|(?:HH|LL|H|L)\*?)?"                  # marker glued to the value: "5.9H", "5.9*"
    r"(?=\s|$|\()"                                           # the value must end here: rejects "148/92", "2-4", "10mg"
)

_MAX_LINE_LENGTH = 200

# An earlier result in a trend column: "8.6", "8.6 H", "171H", "<0.01".
_PREVIOUS_RESULT_RE = re.compile(rf"{_CMP}?\s?{_NUM}(?:\s?(?:HH|LL|H|L)\b|\*)?(?=\s|$)")

# Multi-line result blocks:
#   TSH (Thyroid Stimulating Hormone)
#     Result: 11.8 mIU/L        HIGH
#     Reference range: 0.45 - 4.50 mIU/L
_RESULT_LINE_RE = re.compile(r"^\s*(?:result|value)\s*[:=]\s*(?P<rest>.+)$", re.IGNORECASE)
_RANGE_LINE_RE = re.compile(r"^\s*(?P<rest>(?:ref(?:erence)?\.?|normal)(?:\s+(?:range|interval))?\s*[:=].+)$", re.IGNORECASE)
_NAME_ONLY_RE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9 ,()/\-]{1,60}$")

# Words that show an unrecognised "name" is really part of a sentence.
_NARRATIVE_WORDS = {
    "was", "were", "is", "are", "be", "been", "has", "had", "have", "her", "his", "their", "your",
    "my", "patient", "pt", "take", "taking", "continue", "start", "stop", "follow", "return",
    "repeat", "recheck", "check", "in", "on", "at", "for", "the", "and", "with", "of", "to", "from",
    "daily", "day", "days", "week", "weeks", "month", "months", "year", "years", "age", "dob", "page",
}


# ---------------------------------------------------------------------------
# Ranges and flags
# ---------------------------------------------------------------------------

@dataclass
class _Range:
    low: Optional[float]
    high: Optional[float]
    text: str
    low_inclusive: bool = True    # ">40" excludes 40 itself; ">=40" and "40-60" include it
    high_inclusive: bool = True   # "<200" excludes 200 itself; "0-200" includes it


def _to_float(text: str) -> float:
    return float(text.replace(",", ""))


def _range_from_match(m: re.Match) -> Optional[_Range]:
    """Build a _Range from a _RANGE_PAREN_RE / _RANGE_BARE_RE match (None if garbled)."""
    text = m.group("core")
    if m.group("low") is not None:
        low, high = _to_float(m.group("low")), _to_float(m.group("high"))
        if low > high:
            return None  # e.g. OCR garbage like "50-3.5": don't trust it
        return _Range(low, high, text)
    cmp, bound = m.group("cmp"), _to_float(m.group("bound"))
    if cmp in _LESS_THAN:
        return _Range(None, bound, text, high_inclusive=(cmp != "<"))
    return _Range(bound, None, text, low_inclusive=(cmp != ">"))


def _decimals(x: float) -> int:
    text = f"{x:g}"
    return len(text.split(".")[1]) if "." in text else 0


def _format_typical(test: LabTest) -> str:
    """'70-99', '0.4-4.0', '<200' or '>=60'."""
    low, high = test.typical_low, test.typical_high
    if low is not None and high is not None:
        d = max(_decimals(low), _decimals(high))  # so TSH shows "0.4-4.0", not "0.4-4"
        return f"{low:.{d}f}-{high:.{d}f}"
    if high is not None:
        return f"<{high:g}"
    return f">={low:g}"


def _unit_key(unit: str) -> str:
    """Normalize unit spelling for comparison: 'x10*3/µL' -> '10^3/ul', 'K/uL' -> 'k/ul'."""
    u = unit.lower().replace(" ", "")
    u = u.replace("µ", "u").replace("μ", "u").replace("mcg", "ug").replace("²", "2").replace("*", "^")
    u = re.sub(r"10e(?=\d)", "10^", u)
    return u[1:] if u.startswith("x") else u


def _typical_range(test: LabTest, unit: Optional[str]) -> Optional[_Range]:
    """Our fallback range for this test, or None if it isn't safe to use one."""
    if test.typical_low is None and test.typical_high is None:
        return None
    if unit is None:
        if not test.unit_optional:
            return None  # e.g. "Glucose 6.2" might be mmol/L: comparing with 70-99 mg/dL would be wrong
    else:
        accepted = {_unit_key(u) for u in (test.typical_unit, *test.other_units)}
        if _unit_key(unit) not in accepted:
            return None
    # Reuse the range parser so "<200" / ">=60" / "70-99" mean the same thing everywhere.
    return _range_from_match(_RANGE_BARE_RE.fullmatch(_format_typical(test)))


def _is_above(x: float, rng: _Range) -> bool:
    return rng.high is not None and (x > rng.high if rng.high_inclusive else x >= rng.high)


def _is_below(x: float, rng: _Range) -> bool:
    return rng.low is not None and (x < rng.low if rng.low_inclusive else x <= rng.low)


def _flag_against_range(value: float, comparator: Optional[str], rng: _Range) -> str:
    """Compare a value (maybe printed as "<0.01" or ">60") with a range."""
    above, below = _is_above(value, rng), _is_below(value, rng)
    if comparator is None:
        return "high" if above else "low" if below else "normal"

    # "<0.01" means "somewhere below 0.01". Only flag it if EVERY such value gets the same flag.
    if comparator in _LESS_THAN:
        if below:
            return "low"
        nothing_can_be_low = rng.low is None or rng.low <= 0  # lab values can't be negative
        return "normal" if not above and nothing_can_be_low else "unknown"
    # ">60" means "somewhere above 60".
    if above:
        return "high"
    return "normal" if not below and rng.high is None else "unknown"


# ---------------------------------------------------------------------------
# Line parsing
# ---------------------------------------------------------------------------

@dataclass
class _Tail:
    """What we found after the value on a line."""
    marker: Optional[str] = None      # "high" | "low" | "abnormal"
    unit: Optional[str] = None
    rng: Optional[_Range] = None
    range_unit: Optional[str] = None  # unit printed inside the range brackets: "(3.5-5.0 mmol/L)"
    range_unreadable: bool = False    # a range was printed but made no sense


def _marker_direction(marker: str) -> str:
    letters = marker.strip("()[]*").upper()
    if letters in ("H", "HH", "HI", "HIGH"):
        return "high"
    if letters in ("L", "LL", "LO", "LOW"):
        return "low"
    return "abnormal"


def _parse_tail(tail: str) -> Optional[_Tail]:
    """Read the marker / unit / range that follow a value, in any order.

    Returns None when the text right after the value is something else
    (e.g. "weeks" or "hr post meal"): then the line isn't a table row.
    """
    found = _Tail()
    pos = 0
    recognised_anything = False
    skipped_previous_results = False
    while pos < len(tail):
        if tail[pos].isspace():
            pos += 1
            continue
        m = None
        if found.marker is None:
            m = _MARKER_RE.match(tail, pos)
            if m:
                found.marker = _marker_direction(m.group(0))
        if m is None and found.unit is None:
            m = _UNIT_RE.match(tail, pos)
            if m:
                found.unit = m.group(0).rstrip(".")  # "x10^3/uL." at the end of a sentence
        if m is None and found.rng is None:
            m = _RANGE_PAREN_RE.match(tail, pos) or _RANGE_BARE_RE.match(tail, pos)
            if m:
                found.rng = _range_from_match(m)
                found.range_unit = m.groupdict().get("unit")
                if found.rng is None:
                    found.range_unreadable = True
                    break
        if m is None and found.unit is None and found.rng is None:
            # Reports that show trend columns print earlier results right after
            # the current one: "7.8 H   8.6 H   %   (4.0-5.6)". The first value is
            # today's; skip the older ones (only kept if a unit/range follows).
            # Columns are separated by a wide gap; a single space is more likely
            # an OCR-split number ("9 9" for "99"), which must not be guessed at.
            # (Ranges are tried first, so "104   44 - 147" keeps its range.)
            prev = _PREVIOUS_RESULT_RE.match(tail, pos)
            if prev and tail[max(0, pos - 2):pos] == "  ":
                skipped_previous_results = True
                pos = prev.end()
                continue
        if m is None:
            break  # something we don't recognise (a date, a comment...): stop reading
        recognised_anything = True
        pos = m.end()

    if not recognised_anything and tail[pos:].strip():
        return None
    if skipped_previous_results and found.unit is None and found.rng is None:
        return None  # several bare numbers in a row: not a recognisable result row
    return found


def _merge_result_blocks(lines: list[str]) -> list[str]:
    """Join "name / Result: ... / Reference range: ..." blocks into one table-like line."""
    merged: list[str] = []
    i = 0
    while i < len(lines):
        result = _RESULT_LINE_RE.match(lines[i])
        if result and merged and _NAME_ONLY_RE.match(merged[-1]):
            line = f"{merged.pop().strip()}  {result.group('rest').strip()}"
            if i + 1 < len(lines):
                ref = _RANGE_LINE_RE.match(lines[i + 1])
                if ref:
                    line += f"  {ref.group('rest').strip()}"
                    i += 1
            merged.append(line)
        else:
            merged.append(lines[i])
        i += 1
    return merged


def _clean_line(line: str) -> str:
    line = line.strip()
    # Dot leaders: "Total Cholesterol ........ 224 mg/dL" -> "Total Cholesterol  224 mg/dL"
    line = re.sub(r"\s*\.{2,}\s*", "  ", line)
    # Pipe tables: "| Potassium | 5.4 | mmol/L | 3.5-5.1 |" -> "Potassium  5.4  mmol/L  3.5-5.1"
    # (header rows have no number and separator rows "|---|---|" have no letters, so both fall out later)
    if "|" in line:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        line = "  ".join(cell for cell in cells if cell)
    return line


def _is_plausible_unknown_test(name: str, source: Optional[str], unit: Optional[str]) -> bool:
    """Should we keep a test that isn't in LAB_TESTS?

    Only if the document itself gives us something to go on (a printed range,
    or an H/L marker next to a unit) and the "name" doesn't read like a sentence.
    """
    if source not in ("report", "report_flag"):
        return False
    if source == "report_flag" and unit is None:
        return False  # in "Urine output 1.5 L" the L means liters, not "low"
    words = _normalize_name(name).split()
    return 0 < len(words) <= 5 and not (set(words) & _NARRATIVE_WORDS)


def _parse_line(line: str) -> Optional[LabValue]:
    m = _LINE_RE.match(line)
    if not m:
        return None
    tail = _parse_tail(line[m.end():])
    if tail is None:
        return None

    name = m.group("name").rstrip(",;.")
    canonical = canonical_test_key(name)
    comparator = m.group("cmp")
    value = _to_float(m.group("num"))
    unit = "%" if m.group("pct") else (tail.unit or tail.range_unit)
    marker = _marker_direction(m.group("gmark")) if m.group("gmark") else tail.marker

    # Decide where the flag comes from (see the module docstring for the order).
    rng: Optional[_Range] = None
    source: Optional[str] = None
    if tail.rng is not None:
        rng, source = tail.rng, "report"
    elif marker in ("high", "low"):
        source = "report_flag"
    elif canonical is not None and not tail.range_unreadable:
        rng = _typical_range(LAB_TESTS[canonical], unit)
        source = "typical" if rng else None

    if rng is not None:
        flag = _flag_against_range(value, comparator, rng)
        if source == "typical" and marker == "abnormal" and flag == "normal":
            flag = "unknown"  # the lab marked it abnormal ("*"): don't contradict it with a generic range
    elif source == "report_flag":
        flag = marker
    else:
        flag = "unknown"

    if canonical is None and not _is_plausible_unknown_test(name, source, unit):
        return None

    return LabValue(
        name=name,
        canonical=canonical,
        value=value,
        value_text=m.group("value"),
        unit=unit,
        ref_low=rng.low if rng else None,
        ref_high=rng.high if rng else None,
        range_text=rng.text if rng else None,
        range_source=source,
        flag=flag,
    )


def parse_lab_values(text: str) -> list[LabValue]:
    """
    Best-effort structured extraction of lab lines from raw text.
    This is intentionally conservative: it's fine to miss an oddly
    formatted line (it'll just be explained as prose by the LLM),
    but it should not fabricate a value that isn't in the text.
    """
    results: list[LabValue] = []
    seen: set[tuple[str, str]] = set()
    for raw_line in _merge_result_blocks((text or "").splitlines()):
        line = _clean_line(raw_line)
        if not line or len(line) > _MAX_LINE_LENGTH:
            continue
        lab = _parse_line(line)
        if lab is None:
            continue
        key = (lab.canonical or lab.name.lower(), lab.value_text)
        if key in seen:  # the same result printed twice (e.g. summary + detail table)
            continue
        seen.add(key)
        results.append(lab)
    return results


# ---------------------------------------------------------------------------
# Prompt context
# ---------------------------------------------------------------------------

def _describe_range(lab: LabValue) -> str:
    if lab.range_source == "report":
        return f"range on report: {lab.range_text}"
    if lab.range_source == "typical":
        return f"typical adult range {lab.range_text}; not printed on the report"
    if lab.range_source == "report_flag":
        return f"no range; report marks it {lab.flag}"
    return "no reference range available"


def flags_to_prompt_context(labs: list[LabValue]) -> str:
    """Turn flagged values into ground-truth context injected into the LLM prompt.

    Uses value_text / range_text so the model sees numbers exactly as printed
    ("0.60", not "0.6").
    """
    if not labs:
        return "No structured lab values were detected in this document."
    lines = []
    for lab in labs:
        if lab.unit == "%":
            value = f"{lab.value_text}%"
        elif lab.unit:
            value = f"{lab.value_text} {lab.unit}"
        else:
            value = lab.value_text
        lines.append(f"- {lab.name}: {value} ({_describe_range(lab)}) -> FLAG: {lab.flag.upper()}")
    return "\n".join(lines)
