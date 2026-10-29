"""Tests for backend/flagging.py: deterministic lab parsing and flagging."""

from dataclasses import asdict
from pathlib import Path

import pytest

from flagging import (
    LAB_TESTS,
    LabValue,
    canonical_test_key,
    flags_to_prompt_context,
    parse_lab_values,
)

SAMPLE_REPORT = Path(__file__).resolve().parent.parent / "samples" / "sample_lab_report.txt"


def parse_one(line: str) -> LabValue:
    """Parse a single line and insist it yields exactly one lab value."""
    labs = parse_lab_values(line)
    assert len(labs) == 1, f"expected one lab from {line!r}, got {labs}"
    return labs[0]


# ---------------------------------------------------------------------------
# Public API shape
# ---------------------------------------------------------------------------

def test_lab_tests_has_exactly_the_agreed_canonical_keys():
    assert set(LAB_TESTS) == {
        "glucose", "sodium", "potassium", "chloride", "co2", "bun", "creatinine", "egfr",
        "calcium", "magnesium", "phosphorus", "total_protein", "albumin", "total_bilirubin",
        "alk_phos", "alt", "ast", "total_cholesterol", "ldl", "hdl", "triglycerides", "a1c",
        "tsh", "free_t4", "wbc", "rbc", "hemoglobin", "hematocrit", "mcv", "platelets",
        "ferritin", "inr", "psa", "vitamin_d", "bnp", "uacr", "anion_gap",
    }


def test_lab_value_serializes_to_api_contract_shape():
    lab = parse_one("Potassium 5.8 mmol/L (3.5-5.0)")
    assert list(asdict(lab)) == [
        "name", "canonical", "value", "value_text", "unit", "ref_low", "ref_high",
        "range_text", "range_source", "flag",
    ]


@pytest.mark.parametrize("key", [
    "hemoglobin", "hematocrit", "rbc", "ferritin", "creatinine", "inr", "vitamin_d", "bnp", "psa",
])
def test_sex_age_or_therapy_dependent_tests_have_no_typical_range(key):
    test = LAB_TESTS[key]
    assert test.typical_low is None and test.typical_high is None


@pytest.mark.parametrize("name, key", [
    ("Potassium", "potassium"),
    ("K", "potassium"),
    ("POTASSIUM, SERUM", "potassium"),
    ("Blood Urea Nitrogen", "bun"),
    ("Urea Nitrogen", "bun"),
    ("HbA1c", "a1c"),
    ("Hgb A1c", "a1c"),
    ("Glycohemoglobin", "a1c"),
    ("Hgb", "hemoglobin"),
    ("White Blood Cell Count", "wbc"),
    ("Estimated GFR", "egfr"),
    ("eGFR (CKD-EPI)", "egfr"),
    ("Bicarbonate", "co2"),
    ("HCO3", "co2"),
    ("Alkaline Phosphatase", "alk_phos"),
    ("ALT (SGPT)", "alt"),
    ("LDL-C", "ldl"),
    ("Glucose, Fasting", "glucose"),
    ("Urine Albumin/Creatinine Ratio", "uacr"),
    ("Microalbumin/Creatinine Ratio", "uacr"),
])
def test_canonical_test_key_aliases(name, key):
    assert canonical_test_key(name) == key


@pytest.mark.parametrize("name", [
    "Bilirubin, Direct",      # a different test from total bilirubin
    "NT-proBNP",              # different ranges from BNP
    "Calcium, Ionized",
    "her potassium was",      # short aliases must not match inside sentences
    "Kidney",
    "",
])
def test_canonical_test_key_rejects_non_matches(name):
    assert canonical_test_key(name) is None


# ---------------------------------------------------------------------------
# The bundled sample report
# ---------------------------------------------------------------------------

def test_sample_report_finds_all_14_labs_with_correct_flags():
    labs = parse_lab_values(SAMPLE_REPORT.read_text())
    flags = {lab.canonical: lab.flag for lab in labs}
    assert flags == {
        "glucose": "high", "sodium": "normal", "potassium": "high", "chloride": "normal",
        "bun": "high", "creatinine": "normal", "calcium": "normal",
        "total_cholesterol": "high", "ldl": "high", "hdl": "low", "triglycerides": "high",
        "hemoglobin": "low", "wbc": "normal", "platelets": "normal",
    }
    assert len(labs) == 14
    assert all(lab.range_source == "report" for lab in labs)


def test_sample_report_values_and_ranges_are_as_printed():
    labs = {lab.canonical: lab for lab in parse_lab_values(SAMPLE_REPORT.read_text())}
    potassium = labs["potassium"]
    assert (potassium.value, potassium.value_text, potassium.unit) == (5.8, "5.8", "mmol/L")
    assert (potassium.ref_low, potassium.ref_high, potassium.range_text) == (3.5, 5.0, "3.5-5.0")
    assert labs["wbc"].unit == "x10^3/uL"
    assert labs["total_cholesterol"].name == "Total Cholesterol"


def test_sample_report_ignores_header_and_physician_notes():
    labs = parse_lab_values(SAMPLE_REPORT.read_text())
    values = {lab.value for lab in labs}
    # DOB 01/01/1970, age 55, collected 07/10/2026, lisinopril 10mg, "in 2 weeks", "in 1 month"
    assert values.isdisjoint({1, 2, 7, 10, 55, 1970, 2026})
    assert all(lab.canonical is not None for lab in labs)


# ---------------------------------------------------------------------------
# Table layouts
# ---------------------------------------------------------------------------

def test_column_table_with_flag_before_unit_and_bare_range():
    lab = parse_one("Hemoglobin   9.8   L   g/dL   12.0-15.5")
    assert (lab.canonical, lab.value, lab.unit) == ("hemoglobin", 9.8, "g/dL")
    assert (lab.range_text, lab.range_source, lab.flag) == ("12.0-15.5", "report", "low")


def test_column_table_with_flag_after_unit():
    lab = parse_one("Glucose\t\t112\tmg/dL\tH\t70-99")
    assert (lab.canonical, lab.unit, lab.range_text, lab.flag) == ("glucose", "mg/dL", "70-99", "high")


@pytest.mark.parametrize("line, value, flag", [
    ("Potassium 5.9H mmol/L 3.5-5.1", 5.9, "high"),
    ("Potassium 3.1 L mmol/L 3.5-5.1", 3.1, "low"),
    ("Potassium 5.9 (H) mmol/L (3.5-5.1)", 5.9, "high"),
    ("Potassium 5.9* mmol/L 3.5-5.1", 5.9, "high"),
    ("Potassium 5.9 mmol/L 3.5-5.1 HH", 5.9, "high"),
])
def test_markers_glued_or_separate_with_range(line, value, flag):
    lab = parse_one(line)
    assert (lab.canonical, lab.value, lab.unit, lab.flag) == ("potassium", value, "mmol/L", flag)
    assert lab.range_source == "report"


def test_pipe_table_skips_header_and_separator_rows():
    text = (
        "| Test | Result | Units | Reference Range |\n"
        "|------|--------|-------|-----------------|\n"
        "| Potassium | 5.4 | mmol/L | 3.5-5.1 |\n"
        "| Sodium | 139 | mmol/L | 135-145 |\n"
    )
    labs = parse_lab_values(text)
    assert [(lab.canonical, lab.value_text, lab.range_text, lab.flag) for lab in labs] == [
        ("potassium", "5.4", "3.5-5.1", "high"),
        ("sodium", "139", "135-145", "normal"),
    ]


# ---------------------------------------------------------------------------
# Ranges and comparators
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line, range_text, low, high, flag", [
    ("Total Cholesterol 238 mg/dL <200", "<200", None, 200, "high"),
    ("Total Cholesterol 180 mg/dL < 200", "< 200", None, 200, "normal"),
    ("HDL 38 mg/dL >40", ">40", 40, None, "low"),
    ("HDL 40 mg/dL >40", ">40", 40, None, "low"),      # ">40" excludes 40 itself
    ("eGFR 60 mL/min/1.73m2 >=60", ">=60", 60, None, "normal"),
    ("eGFR 45 mL/min/1.73m2 ≥60", "≥60", 60, None, "low"),
    ("Hemoglobin A1c 6.1 % Ref: <5.7", "<5.7", None, 5.7, "high"),
    ("Triglycerides 140 mg/dL (ref <150)", "<150", None, 150, "normal"),
])
def test_one_sided_ranges(line, range_text, low, high, flag):
    lab = parse_one(line)
    assert (lab.range_text, lab.ref_low, lab.ref_high, lab.flag) == (range_text, low, high, flag)
    assert lab.range_source == "report"


@pytest.mark.parametrize("line, value_text, flag", [
    ("Troponin I <0.01 ng/mL (<0.04)", "<0.01", "normal"),
    ("Troponin I <0.01 ng/mL (0.00-0.04)", "<0.01", "normal"),
    ("Lipase <5 U/L (10-60)", "<5", "low"),          # every value below 5 is below 10
    ("Lipase <15 U/L (10-60)", "<15", "unknown"),    # could be 9 (low) or 12 (normal)
    ("Lipase >600 U/L (10-60)", ">600", "high"),
    ("eGFR >90 mL/min/1.73m2 (>59)", ">90", "normal"),
])
def test_comparator_values(line, value_text, flag):
    lab = parse_one(line)
    assert (lab.value_text, lab.flag) == (value_text, flag)
    assert lab.value == float(value_text.lstrip("<>"))


@pytest.mark.parametrize("line, range_text", [
    ("Potassium 5.8 mmol/L (3.5–5.0)", "3.5–5.0"),   # en dash
    ("Potassium 5.8 mmol/L (3.5—5.0)", "3.5—5.0"),   # em dash
    ("Potassium 5.8 mmol/L 3.5 - 5.0", "3.5 - 5.0"),
    ("Potassium 5.8 mmol/L (3.5 to 5.0)", "3.5 to 5.0"),
])
def test_range_separators_are_parsed_and_preserved(line, range_text):
    lab = parse_one(line)
    assert (lab.ref_low, lab.ref_high, lab.range_text, lab.flag) == (3.5, 5.0, range_text, "high")


def test_value_and_range_text_keep_document_formatting():
    lab = parse_one("Creatinine 0.60 mg/dL (0.50-1.10)")
    assert (lab.value, lab.value_text, lab.range_text) == (0.6, "0.60", "0.50-1.10")
    big = parse_one("Ferritin 1,250 ng/mL (15-150)")
    assert (big.value, big.value_text, big.flag) == (1250.0, "1,250", "high")


def test_garbled_range_is_not_trusted_and_does_not_fall_back_to_typical():
    lab = parse_one("Potassium 5.8 mmol/L (5.0-3.5)")
    assert (lab.flag, lab.range_source, lab.range_text) == ("unknown", None, None)


# ---------------------------------------------------------------------------
# Units and OCR noise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line, unit", [
    ("Potassium 5.8 mmol/l (3.5-5.0)", "mmol/l"),
    ("TSH 2.10 µIU/mL (0.45-4.50)", "µIU/mL"),
    ("TSH 2.10 uIU/mL (0.45-4.50)", "uIU/mL"),
    ("WBC 7.2 x10^3/uL (4.5-11.0)", "x10^3/uL"),
    ("WBC 7.2 K/uL (4.5-11.0)", "K/uL"),
    ("Platelets 260 10*3/uL (150-450)", "10*3/uL"),
    ("eGFR 85 mL/min/1.73m2 (>59)", "mL/min/1.73m2"),
    ("Hemoglobin A1c 5.4 % (4.0-5.6)", "%"),
    ("Hemoglobin A1c 5.4% (4.0-5.6)", "%"),
    ("MCV 88 fL (80-100)", "fL"),
    ("PSA 1.2 ng/mL (0.0-4.0)", "ng/mL"),
    ("Vitamin B12 450 pg/mL (200-900)", "pg/mL"),
    ("ALT 30 U/L (7-56)", "U/L"),
    ("AST 25 IU/L (10-40)", "IU/L"),
    ("Sodium 140 mEq/L (135-145)", "mEq/L"),
    ("CRP 3.2 mg/L (<10)", "mg/L"),
    ("Potassium 5.8 (3.5-5.0 mmol/L)", "mmol/L"),
])
def test_units_are_captured_as_printed(line, unit):
    lab = parse_one(line)
    assert lab.unit == unit
    assert lab.range_source == "report"


def test_ocr_typos_never_crash_or_fabricate():
    text = (
        "Potass1um 5.8 mmol/L (3.5-5.0)\n"   # misspelled name: kept only as an unrecognised test
        "Sodium 14O mmol/L (135-145)\n"      # letter O inside the number: skipped
        "Chloride 1O1 mmol/L\n"
        "Glucose 9 9 mg/dL\n"
        "\x0c |||\n"
    )
    labs = parse_lab_values(text)
    assert all(lab.canonical != "potassium" for lab in labs)
    assert all(lab.canonical not in ("sodium", "chloride", "glucose") for lab in labs)
    for lab in labs:
        assert lab.value_text in text  # whatever we return was really printed


def test_hemoglobin_a1c_is_not_read_as_hemoglobin_1():
    lab = parse_one("Hemoglobin A1c 6.8 % (4.0-5.6)")
    assert (lab.canonical, lab.name, lab.value, lab.flag) == ("a1c", "Hemoglobin A1c", 6.8, "high")


# ---------------------------------------------------------------------------
# Narrative text must not produce lab values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line", [
    "Follow up in 2-4 weeks",
    "BP 148/92",
    "Her potassium was 3.1 today",
    "her potassium was 3.1 today",
    "Potassium 3.1 today, recheck",
    "Metformin 500 mg twice daily",
    "metformin 500 mg twice daily",
    "Continue lisinopril 10mg daily.",
    "Glucose 2 hr post meal 140 mg/dL",
    "DOB: 01/01/1970 (Age 55)",
    "COLLECTED: 07/10/2026",
    "Page 1 of 2",
    "Repeat labs in 3 months (2-4 weeks if worse)",
    "Urine output 1.5 L",
])
def test_narrative_lines_produce_nothing(line):
    assert parse_lab_values(line) == []


def test_short_alias_works_at_line_start_only():
    assert parse_one("K 5.9 mmol/L (3.5-5.0)").canonical == "potassium"
    assert parse_one("Potassium: 5.9 mmol/L").canonical == "potassium"
    assert parse_lab_values("K was 5.9 this morning") == []


# ---------------------------------------------------------------------------
# Where the flag comes from when no range is printed
# ---------------------------------------------------------------------------

def test_marker_without_range_gives_report_flag():
    lab = parse_one("Troponin I 0.08 ng/mL H")
    assert (lab.canonical, lab.flag, lab.range_source, lab.range_text) == (None, "high", "report_flag", None)
    # The lab's own marker beats our typical range.
    lab = parse_one("Potassium 3.4 mmol/L L")
    assert (lab.flag, lab.range_source) == ("low", "report_flag")


def test_typical_range_used_only_when_report_prints_none():
    lab = parse_one("Glucose 112 mg/dL")
    assert (lab.flag, lab.range_source, lab.range_text, lab.ref_low, lab.ref_high) == (
        "high", "typical", "70-99", 70, 99)
    assert parse_one("TSH 2.1 µIU/mL").range_text == "0.4-4.0"
    assert parse_one("WBC 12.5 K/uL").flag == "high"           # K/uL == x10^3/uL
    assert parse_one("Platelets 120 10*3/uL").flag == "low"


@pytest.mark.parametrize("line, flag, range_text", [
    ("HDL 75 mg/dL", "normal", ">=40"),            # high HDL is never flagged with the typical range
    ("HDL 35 mg/dL", "low", ">=40"),
    ("Total Cholesterol 200 mg/dL", "high", "<200"),
    ("LDL 99 mg/dL", "normal", "<100"),
    ("Triglycerides 150 mg/dL", "high", "<150"),
    ("eGFR 60 mL/min/1.73m2", "normal", ">=60"),
    ("eGFR 59 mL/min/1.73m2", "low", ">=60"),
    ("eGFR >60", "normal", ">=60"),
])
def test_one_sided_typical_ranges(line, flag, range_text):
    lab = parse_one(line)
    assert (lab.flag, lab.range_source, lab.range_text) == (flag, "typical", range_text)


@pytest.mark.parametrize("line", [
    "Hemoglobin 11.0 g/dL",
    "Creatinine 1.4 mg/dL",
    "INR 2.5",
    "Ferritin 12 ng/mL",
    "PSA 4.5 ng/mL",
    "Glucose 6.2 mmol/L",   # different unit system: 70-99 mg/dL would be wrong
    "Glucose 112",          # no unit: can't be sure it's mg/dL
])
def test_no_safe_typical_range_gives_unknown(line):
    lab = parse_one(line)
    assert lab.canonical is not None
    assert (lab.flag, lab.range_source, lab.range_text, lab.ref_low, lab.ref_high) == (
        "unknown", None, None, None, None)


def test_unit_optional_tests_use_typical_range_without_unit():
    lab = parse_one("Potassium 5.8")
    assert (lab.unit, lab.flag, lab.range_source) == (None, "high", "typical")


def test_star_marker_is_not_contradicted_by_typical_range():
    lab = parse_one("Glucose 95 * mg/dL")
    assert (lab.flag, lab.range_source) == ("unknown", "typical")


def test_unrecognised_test_needs_a_printed_range_or_marker():
    lab = parse_one("Lipase 45 U/L (13-60)")
    assert (lab.name, lab.canonical, lab.flag, lab.range_source) == ("Lipase", None, "normal", "report")
    assert parse_lab_values("Lipase 45 U/L") == []
    assert parse_lab_values("Lipase 45 U/L *") == []           # "*" alone gives no direction
    assert parse_lab_values("Some Very Long Unknown Test Name 45 U/L (13-60)") == []


def test_duplicate_results_are_reported_once():
    text = "Potassium 5.8 mmol/L (3.5-5.0)\nSUMMARY\nPotassium 5.8 mmol/L H\nPotassium 4.9 mmol/L (3.5-5.0)"
    labs = parse_lab_values(text)
    assert [(lab.canonical, lab.value_text) for lab in labs] == [("potassium", "5.8"), ("potassium", "4.9")]


def test_empty_and_blank_input():
    assert parse_lab_values("") == []
    assert parse_lab_values("   \n\n\t") == []


# ---------------------------------------------------------------------------
# Prompt context
# ---------------------------------------------------------------------------

def test_flags_to_prompt_context_formats():
    labs = parse_lab_values(
        "Potassium 5.8 mmol/L (3.5-5.0)\n"
        "Glucose 112 mg/dL\n"
        "Troponin I 0.08 ng/mL H\n"
        "INR 2.5\n"
        "Creatinine 0.60 mg/dL (0.50-1.10)\n"
        "Hemoglobin A1c 6.8% (4.0-5.6)\n"
    )
    assert flags_to_prompt_context(labs).splitlines() == [
        "- Potassium: 5.8 mmol/L (range on report: 3.5-5.0) -> FLAG: HIGH",
        "- Glucose: 112 mg/dL (typical adult range 70-99; not printed on the report) -> FLAG: HIGH",
        "- Troponin I: 0.08 ng/mL (no range; report marks it high) -> FLAG: HIGH",
        "- INR: 2.5 (no reference range available) -> FLAG: UNKNOWN",
        "- Creatinine: 0.60 mg/dL (range on report: 0.50-1.10) -> FLAG: NORMAL",
        "- Hemoglobin A1c: 6.8% (range on report: 4.0-5.6) -> FLAG: HIGH",
    ]


def test_flags_to_prompt_context_empty():
    assert flags_to_prompt_context([]) == "No structured lab values were detected in this document."


# --- formats added during integration ------------------------------------------------

def test_dot_leaders_are_treated_as_column_gaps():
    labs = parse_lab_values("Total Cholesterol ........ 224 mg/dL     Ref: <200      HIGH\n"
                            "HDL Cholesterol .......... 46 mg/dL      Ref: >40")
    by_key = {lab.canonical: lab for lab in labs}
    assert by_key["total_cholesterol"].flag == "high"
    assert by_key["total_cholesterol"].range_text == "<200"
    assert by_key["hdl"].flag == "normal"


def test_previous_result_columns_are_skipped():
    labs = parse_lab_values("Hemoglobin A1c                   7.8 H        8.6 H        %                (4.0-5.6)\n"
                            "Glucose, Fasting                 146 H        171 H        mg/dL            (70-99)\n"
                            "Estimated Average Glucose        177          200          mg/dL")
    assert [(lab.canonical, lab.value_text, lab.flag) for lab in labs] == [
        ("a1c", "7.8", "high"),
        ("glucose", "146", "high"),
    ]


def test_bare_number_runs_are_not_results():
    assert parse_lab_values("Potassium 5.8 4.9 3.7") == []


def test_multiline_result_blocks():
    text = (
        "TSH (Thyroid Stimulating Hormone)\n"
        "  Result: 11.8 mIU/L        HIGH\n"
        "  Reference range: 0.45 - 4.50 mIU/L\n"
        "\n"
        "Free T4 (Free Thyroxine)\n"
        "  Result: 0.62 ng/dL        LOW\n"
        "  Reference range: 0.82 - 1.77 ng/dL\n"
    )
    labs = parse_lab_values(text)
    assert [(lab.canonical, lab.value_text, lab.flag, lab.range_source) for lab in labs] == [
        ("tsh", "11.8", "high", "report"),
        ("free_t4", "0.62", "low", "report"),
    ]


def test_unit_followed_by_sentence_period():
    labs = parse_lab_values("WBC 14.2 x10^3/uL. CT abdomen/pelvis with IV contrast: dilated appendix")
    assert labs[0].unit == "x10^3/uL"
    assert labs[0].flag == "high" and labs[0].range_source == "typical"


def test_bracketed_qualifier_with_digits_in_name():
    labs = parse_lab_values("eGFR (CKD-EPI 2021)              89           86           mL/min/1.73m2    (>=60)")
    assert [(lab.canonical, lab.value_text, lab.flag) for lab in labs] == [("egfr", "89", "normal")]


def test_spaced_range_after_value_is_not_mistaken_for_a_trend_column():
    labs = parse_lab_values("Alkaline Phosphatase       104             44 - 147         U/L\n"
                            "ALT (SGPT)                 142     HIGH    7 - 56           U/L")
    assert [(lab.canonical, lab.value_text, lab.range_text, lab.flag) for lab in labs] == [
        ("alk_phos", "104", "44 - 147", "normal"),
        ("alt", "142", "7 - 56", "high"),
    ]
