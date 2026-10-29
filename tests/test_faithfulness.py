from faithfulness import explanation_numbers, normalize_number, numeric_faithfulness


def test_sentence_ending_period_is_not_part_of_the_number():
    # "22." used to be extracted as "22." and wrongly flagged as unsupported.
    result = numeric_faithfulness("BUN 22 mg/dL", "Your BUN was 22.")
    assert result["unsupported_numbers"] == []
    assert result["numeric_faithfulness_score"] == 1.0


def test_equivalent_number_formats_match():
    assert normalize_number("5.0") == normalize_number("5")
    assert normalize_number("07") == "7"
    assert normalize_number("1,000") == "1000"
    assert normalize_number("100") == "100"
    assert normalize_number("0.60") == "0.6"


def test_citation_markers_are_not_numbers():
    assert explanation_numbers("Potassium helps your heart [2]. See also [1, 3] and [4-6].") == set()


def test_invented_number_is_flagged():
    result = numeric_faithfulness("Potassium 5.8 mmol/L (3.5-5.0)", "Your potassium was 5.9, a bit above 5.0.")
    assert result["unsupported_numbers"] == ["5.9"]
    assert result["total_numbers_in_explanation"] == 2
    assert result["numeric_faithfulness_score"] == 0.5


def test_typical_range_numbers_are_allowed_but_reported_separately():
    source = "Glucose 112 mg/dL"
    lab_context = "- Glucose: 112 mg/dL (typical adult range 70-99; not printed on the report) -> FLAG: HIGH"
    result = numeric_faithfulness(source, "Your glucose of 112 is above the usual 70 to 99.", lab_context)
    assert result["unsupported_numbers"] == []
    assert result["numbers_from_reference_ranges"] == ["70", "99"]


def test_no_numbers_scores_perfectly():
    assert numeric_faithfulness("Normal chest X-ray.", "Your chest X-ray looked normal.")["numeric_faithfulness_score"] == 1.0


def test_dates_rewritten_in_words_still_trace_back():
    result = numeric_faithfulness("COLLECTED: 07/10/2026", "Your blood was drawn on July 10, 2026.")
    assert result["unsupported_numbers"] == []


def test_numbered_list_markers_are_not_values():
    result = numeric_faithfulness("Take lisinopril 10 mg daily.", "1. Take lisinopril 10 mg daily.\n2. Call your doctor.")
    assert result["unsupported_numbers"] == []


def test_comma_separated_readings_in_source():
    result = numeric_faithfulness("Glucose checks overnight: 180,210,240", "Your overnight readings were 180, 210 and 240.")
    assert result["unsupported_numbers"] == []
