"""Tests for the MedlinePlus retrieval layer (backend/medlineplus.py).

No network: every test runs with medlineplus._get_json replaced by a fake. Drug
lookups are served from canned RxNav / MedlinePlus Connect responses.
"""

import json
import os
import time
import urllib.parse
from types import SimpleNamespace

import pytest

import medlineplus as mp

ROOT = os.path.join(os.path.dirname(__file__), "..")
SAMPLE_PATH = os.path.join(ROOT, "samples", "sample_lab_report.txt")

LAB_KEYS = [
    "glucose", "sodium", "potassium", "chloride", "co2", "bun", "creatinine", "egfr", "calcium", "magnesium",
    "phosphorus", "total_protein", "albumin", "total_bilirubin", "alk_phos", "alt", "ast", "total_cholesterol",
    "ldl", "hdl", "triglycerides", "a1c", "tsh", "free_t4", "wbc", "rbc", "hemoglobin", "hematocrit", "mcv",
    "platelets", "ferritin", "inr", "psa", "vitamin_d", "bnp", "uacr", "anion_gap",
]

SAMPLE_TEXT = """PATIENT: [SYNTHETIC TEST PATIENT]
COMPREHENSIVE METABOLIC PANEL
Glucose        112 mg/dL     (70-99)
Sodium         138 mmol/L    (135-145)
Potassium      5.8 mmol/L    (3.5-5.0)
Calcium        9.4 mg/dL     (8.5-10.5)
LIPID PANEL
Total Cholesterol   238 mg/dL   (0-200)
LDL                  158 mg/dL   (0-100)
CBC
Hemoglobin      12.1 g/dL    (13.5-17.5)
PHYSICIAN NOTES:
Patient presents for routine follow-up of hypertension and reports mild fatigue.
Continue lisinopril 10mg daily. Recommend dietary counseling given elevated lipid
panel and repeat potassium check in 2 weeks given mild hyperkalemia, likely related
to ACE inhibitor therapy. Follow up in 1 month.
"""


def lab(name, canonical, flag, value=None, low=None, high=None):
    return SimpleNamespace(name=name, canonical=canonical, flag=flag, value=value, ref_low=low, ref_high=high)


SAMPLE_LABS = [
    lab("Glucose", "glucose", "high", 112, 70, 99),
    lab("Sodium", "sodium", "normal", 138, 135, 145),
    lab("Potassium", "potassium", "high", 5.8, 3.5, 5.0),
    lab("Calcium", "calcium", "normal", 9.4, 8.5, 10.5),
    lab("LDL", "ldl", "high", 158, 0, 100),
    lab("Hemoglobin", "hemoglobin", "normal", 13.9, 13.5, 17.5),
]

LISINOPRIL_URL = "https://medlineplus.gov/druginfo/meds/a692051.html"
REAL_GET_JSON = mp._get_json  # captured before the autouse fixture replaces it


# --------------------------------------------------------------------------- fakes

class FakeNet:
    """Stands in for medlineplus._get_json: canned RxNav + Connect answers."""

    def __init__(self, drugs=None, fail=False, feeds=None):
        # feeds: drug name (lower) -> (rxcui, [(title, url), ...]) for an rxcui whose
        # Connect answer lists several drug pages (route variants, ambiguous salts)
        self.feeds = feeds or {}
        # drug name (lower) -> (rxcui, title, url, summary)
        self.drugs = drugs if drugs is not None else {
            "lisinopril": ("29046", "Lisinopril", LISINOPRIL_URL + "?utm_source=mplusconnect&utm_medium=service",
                           "Lisinopril is used alone or in combination with other medications to treat high blood "
                           "pressure. It is in a class of medications called ACE inhibitors."),
            "metformin": ("6809", "Metformin", "https://medlineplus.gov/druginfo/meds/a696005.html?utm_source=x",
                          "Metformin is used to treat type 2 diabetes."),
        }
        self.fail = fail
        self.calls = []

    def __call__(self, url, timeout):
        self.calls.append(url)
        if self.fail:
            return None
        parts = urllib.parse.urlsplit(url)
        q = dict(urllib.parse.parse_qsl(parts.query))
        if "rxnav" in parts.netloc:
            hit = self.drugs.get(q.get("name", "").lower()) or self.feeds.get(q.get("name", "").lower())
            return {"idGroup": {"rxnormId": [hit[0]]} if hit else {}}
        if "connect.medlineplus.gov" in parts.netloc:
            code = q.get("mainSearchCriteria.v.c")
            for rxcui, pages in self.feeds.values():
                if rxcui == code:
                    return {"feed": {"entry": [
                        {"title": {"_value": title}, "link": [{"href": url}],
                         "summary": {"_value": f"{title} is a medicine."}} for title, url in pages]}}
            for _rxcui, title, durl, summary in (d for d in self.drugs.values() if d[0] == code):
                return {"feed": {"entry": [
                    {"title": {"_value": "Blood Pressure Medicines"},
                     "link": [{"href": "https://medlineplus.gov/bloodpressuremedicines.html?utm_source=x"}],
                     "summary": {"_value": "<p>Topic page, not a drug page.</p>"}},
                    {"title": {"_value": title}, "link": [{"href": durl}], "summary": {"_value": summary}},
                ]}}
            return {"feed": {"entry": []}}
        raise AssertionError(f"unexpected URL {url}")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Default: no drug is known and nothing touches the network."""
    fake = FakeNet(drugs={})
    monkeypatch.setattr(mp, "_get_json", fake)
    mp._DRUG_CACHE.clear()
    yield fake
    mp._DRUG_CACHE.clear()


@pytest.fixture
def net(monkeypatch):
    fake = FakeNet()
    monkeypatch.setattr(mp, "_get_json", fake)
    return fake


def titles(refs):
    return [r.title for r in refs]


def read_sample(name):
    with open(os.path.join(ROOT, "samples", name), encoding="utf-8") as fh:
        return fh.read()


def rxnav_names(fake):
    """Medication names the fake RxNav was asked about, in order."""
    return [dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(u).query))["name"]
            for u in fake.calls if "rxnav" in u]


def topic_title(idx):
    return mp._load_index()["topics"][idx]["title"] if idx is not None else None


# --------------------------------------------------------------------------- index

def test_index_loads_with_topics_and_all_lab_tests():
    stats = mp.index_stats()
    assert stats["topics"] >= 900
    assert stats["lab_tests"] == len(LAB_KEYS)
    assert len(stats["source_date"]) == 10 and stats["source_date"][4] == "-"

    tests = mp._load_index()["lab_tests"]
    for key in LAB_KEYS:
        entry = tests[key]
        assert entry["url"].startswith("https://medlineplus.gov/lab-tests/"), key
        assert entry["title"] and entry["description"] and len(entry["text"]) > 200, key
        assert "utm_" not in entry["url"]
    assert tests["potassium"]["title"] == "Potassium Blood Test"
    assert tests["a1c"]["title"].startswith("Hemoglobin A1C")


def test_topic_records_are_clean():
    topics = mp._load_index()["topics"]
    for t in topics:
        assert t["title"] and t["url"].startswith("https://medlineplus.gov/")
        assert "utm_" not in t["url"]
        assert len(t["text"]) <= 950
        assert not t["description"].endswith(("Learn more.", "Read more.", "Find out more."))
        assert "NIH: National" not in t["text"]
    hbp = next(t for t in topics if t["title"] == "High Blood Pressure")
    assert "Hypertension" in hbp["synonyms"]


def test_curated_titles_exist_in_index():
    """Guards against MedlinePlus renaming a topic that a curated list refers to."""
    known = {t["title"] for t in mp._load_index()["topics"]}
    curated = set(mp.STOP_TOPICS) | set(mp.EXTRA_SYNONYMS) | set(mp.CONCEPT_SYNONYMS)
    curated |= {t for titles_ in mp.LAB_ANALYTE_TOPICS.values() for t in titles_}
    assert curated - known == set()


def test_lazy_structures_build_quickly():
    for fn in (mp._load_index, mp._topic_by_title, mp._stop_topic_ids, mp._phrase_candidates,
               mp._phrase_tables, mp._bm25, mp._lab_alias_map, mp._lab_words):
        fn.cache_clear()
    start = time.perf_counter()
    mp._phrase_tables()
    mp._bm25()
    mp._lab_alias_map()
    assert time.perf_counter() - start < 2.0  # ~0.35 s on a laptop; generous for CI


# --------------------------------------------------------------------------- synonyms

@pytest.mark.parametrize("term, expected", [
    ("hypertension", "High Blood Pressure"),
    ("Hypertension", "High Blood Pressure"),
    ("HTN", "High Blood Pressure"),
    ("hyperkalemia", "Potassium"),
    ("A1C", "A1C"),
    ("HbA1c", "A1C"),
    ("hemoglobin A1c", "A1C"),
    ("Hgb A1C", "A1C"),
    ("glycated hemoglobin", "A1C"),
    ("Alzheimers disease", "Alzheimer's Disease"),
    ("Abdominal pain", "Abdominal Pain"),
    ("chronic kidney disease", "Chronic Kidney Disease"),
    ("mild hyperkalemia", "Potassium"),
])
def test_exact_synonym_mapping(term, expected):
    assert topic_title(mp._exact_concept_topic(term)) == expected


@pytest.mark.parametrize("name, key", [
    ("HbA1c", "a1c"), ("Hemoglobin A1c", "a1c"), ("A1C", "a1c"), ("K", "potassium"),
    ("BUN", "bun"), ("eGFR", "egfr"), ("WBC", "wbc"), ("LDL Cholesterol", "ldl"),
])
def test_lab_aliases(name, key):
    assert mp._lab_key(SimpleNamespace(name=name, canonical=None)) == key


def test_ambiguous_and_generic_phrases_are_not_matched():
    doc, _, _ = mp._phrase_tables()
    for phrase in ("blood pressure", "joint", "ice", "period", "pain", "medications", "children", "health",
                   "metabolic", "family history", "allergies", "ms", "md", "ed", "all"):
        assert tuple(mp._tokens(phrase)) not in doc, phrase


# --------------------------------------------------------------------------- find_references

def test_find_references_on_sample_report(net):
    refs = mp.find_references(SAMPLE_TEXT, SAMPLE_LABS)
    kinds = [r.kind for r in refs]

    assert len(refs) <= 8
    assert len({r.url for r in refs}) == len(refs)
    # abnormal labs first, most abnormal (LDL 58% over) before potassium (53%) and glucose (45%)
    assert titles(refs)[:3] == ["Cholesterol Levels", "Potassium Blood Test", "Blood Glucose Test"]
    assert kinds[:3] == ["lab_test"] * 3
    assert kinds == sorted(kinds, key=["lab_test", "topic", "drug"].index)

    by_title = {r.title: r for r in refs}
    assert by_title["High Blood Pressure"].kind == "topic"
    assert by_title["High Blood Pressure"].matched == "hypertension"
    assert by_title["Lisinopril"].kind == "drug"
    assert by_title["Lisinopril"].url == LISINOPRIL_URL  # utm_* stripped
    assert by_title["Lisinopril"].matched == "lisinopril"
    # superseded by the Potassium Blood Test page / normal labs listed in the table
    for junk in ("Potassium", "Sodium", "Calcium", "Blood Glucose", "Cholesterol", "Blood Count Tests",
                 "Hemoglobin Test"):
        assert junk not in by_title, junk
    for r in refs:
        assert r.snippet and r.text and r.matched


@pytest.mark.skipif(not os.path.exists(SAMPLE_PATH), reason="sample report not present")
def test_find_references_on_real_sample_file_with_full_panel(net):
    with open(SAMPLE_PATH, encoding="utf-8") as fh:
        text = fh.read()
    labs = SAMPLE_LABS + [
        lab("BUN", "bun", "high", 22, 7, 20), lab("Total Cholesterol", "total_cholesterol", "high", 238, 0, 200),
        lab("HDL", "hdl", "low", 38, 40, 60), lab("Triglycerides", "triglycerides", "high", 190, 0, 150),
        lab("WBC", "wbc", "normal", 7.2, 4.5, 11), lab("Platelets", "platelets", "normal", 260, 150, 450),
    ]
    refs = mp.find_references(text, labs)
    kinds = [r.kind for r in refs]
    assert len(refs) == 8
    # 6 distinct abnormal-lab pages compete; topics and the drug keep their reserved slots
    assert kinds.count("lab_test") >= 4
    assert "High Blood Pressure" in titles(refs) and "Lisinopril" in titles(refs)
    assert titles(refs)[0] == "Cholesterol Levels"


def test_lab_pages_merge_labs_sharing_a_page():
    labs = [lab("Total Cholesterol", "total_cholesterol", "high", 238, 0, 200),
            lab("LDL", "ldl", "high", 158, 0, 100), lab("HDL", "hdl", "low", 38, 40, 60)]
    refs = mp.find_references("", labs)
    assert titles(refs) == ["Cholesterol Levels"]
    assert refs[0].matched == "LDL, Total Cholesterol, HDL"


def test_labs_without_canonical_key_and_dict_labs():
    refs = mp.find_references("", [{"name": "HbA1c", "canonical": None, "flag": "high"},
                                   SimpleNamespace(name="Mystery", canonical=None, flag="high")])
    assert titles(refs) == ["Hemoglobin A1C (HbA1c) Test"]


def test_hyperkalemia_maps_to_potassium_when_no_lab_page():
    refs = mp.find_references("Assessment: mild hyperkalemia, likely dietary.", [])
    assert "Potassium" in titles(refs)
    assert refs[0].matched == "hyperkalemia"


def test_analyte_name_in_table_does_not_pull_topic_but_condition_does():
    labs = [lab("Potassium", "potassium", "normal", 4.1, 3.5, 5.0)]
    assert mp.find_references("Potassium 4.1 mmol/L (3.5-5.0)", labs) == []
    refs = mp.find_references("Potassium 4.1 mmol/L. History of hyperkalemia.", labs)
    assert titles(refs) == ["Potassium"]


def test_generic_words_produce_no_junk_topics():
    text = """Patient seen in clinic today. Vital signs stable. Medications reviewed with patient.
Allergies: NKDA. Family history reviewed. Insurance: Medicare. Pediatrics department.
Discussed diet, exercise and stress management. Labs drawn, results normal. Pain 0/10.
Apply ice to the joint for 20 minutes. Period of observation: 2 hours. Blood pressure 122/78.
Signed: Robert Walker, MD, ED attending. Follow up in 2 weeks. Comprehensive metabolic panel pending."""
    assert mp.find_references(text, []) == []


def test_context_words_block_misleading_topics():
    refs = mp.find_references("ECG: 1 mm ST depression in V4-V6. Respiratory depression after opioids. "
                              "Stress test scheduled. Bowel prep given.", [])
    assert "Depression" not in titles(refs)
    assert "Stress" not in titles(refs)
    assert "HIV: PrEP and PEP" not in titles(refs)
    assert "Heart Health Tests" in titles(refs)


def test_negated_and_family_history_mentions_are_skipped():
    text = ("No evidence of pneumonia. Denies chest pain. Family history of breast cancer. "
            "Atrial fibrillation was ruled out. Diagnosis: asthma.")
    assert titles(mp.find_references(text, [])) == ["Asthma"]


def test_longest_match_wins():
    refs = mp.find_references("History of pulmonary hypertension and type 2 diabetes.", [])
    assert titles(refs) == ["Pulmonary Hypertension", "Diabetes Type 2"]


def test_medication_lines_do_not_create_topics(net):
    text = "DISCHARGE MEDICATIONS:\n1. Metformin 500 mg twice daily\n2. Atorvastatin calcium 40 mg nightly\n"
    refs = mp.find_references(text, [])
    assert "Calcium" not in titles(refs)
    assert titles(refs) == ["Metformin"]


def test_allocation_reserves_slots_for_topics_and_drugs(net):
    labs = [lab(k, k, "high") for k in ("glucose", "potassium", "sodium", "bun", "creatinine", "calcium",
                                        "alt", "ast", "tsh", "ferritin")]
    text = "Hypertension and asthma. Continue lisinopril 10 mg daily."
    refs = mp.find_references(text, labs, max_refs=8)
    kinds = [r.kind for r in refs]
    assert len(refs) == 8
    assert kinds.count("lab_test") == 5
    assert kinds.count("topic") == 2
    assert kinds.count("drug") == 1
    assert mp.find_references(text, labs, max_refs=0) == []
    assert len(mp.find_references(text, labs, max_refs=3)) == 3


# --------------------------------------------------------------------------- medications

def test_medication_candidates_from_doses_and_med_section():
    text = """Continue lisinopril 10mg daily. Start metformin 500 mg twice daily.
Hemoglobin 12.1 g/dL. Creatinine 1.1 mg/dL. Sodium 138 mEq/L.
CURRENT MEDICATIONS:
- Atorvastatin
- Aspirin EC 81 mg
- Insulin glargine 20 units at bedtime
ALLERGIES: penicillin
"""
    groups = mp._medication_candidates(text)
    flat = [name.lower() for g in groups for name in g]
    assert groups[0] == ["lisinopril"]
    assert ["metformin"] in [[n.lower() for n in g] for g in groups]
    assert "atorvastatin" in flat and "aspirin" in flat and "insulin glargine" in flat
    for not_drug in ("continue", "start", "hemoglobin", "creatinine", "sodium", "penicillin", "allergies"):
        assert not_drug not in flat, not_drug


def test_med_section_with_crlf_line_endings(net):
    text = "Assessment: asthma.\r\nMEDICATIONS:\r\n- Metformin 500 mg daily\r\n- Atorvastatin calcium 40 mg\r\n"
    refs = mp.find_references(text, [])
    assert titles(refs) == ["Asthma", "Metformin"]  # no Calcium topic from the med line


def test_drug_lookups_are_capped_per_document(no_network):
    text = "\n".join(f"Drug{chr(97 + i)}xyz {i + 1}0 mg daily" for i in range(20))
    mp.find_references(text, [])
    rxnav_calls = [u for u in no_network.calls if "rxnav" in u]
    assert 0 < len(rxnav_calls) <= mp.MAX_DRUG_LOOKUPS


def test_lookup_drug_success_and_cache(net):
    ref = mp.lookup_drug("Lisinopril")
    assert ref.kind == "drug" and ref.title == "Lisinopril" and ref.url == LISINOPRIL_URL
    assert "ACE inhibitors" in ref.text and ref.snippet
    n_calls = len(net.calls)
    again = mp.lookup_drug("lisinopril")
    assert again.url == LISINOPRIL_URL and len(net.calls) == n_calls  # served from cache


def test_lookup_drug_unknown_name_returns_none(net):
    assert mp.lookup_drug("blorfinazole") is None
    assert mp.lookup_drug("") is None
    assert mp.lookup_drug(None) is None


def test_lookup_drug_network_failure_returns_none_and_is_not_cached(monkeypatch):
    down = FakeNet(fail=True)
    monkeypatch.setattr(mp, "_get_json", down)
    assert mp.lookup_drug("lisinopril") is None
    monkeypatch.setattr(mp, "_get_json", FakeNet())  # network back
    assert mp.lookup_drug("lisinopril").url == LISINOPRIL_URL


def test_lookup_drug_never_raises(monkeypatch):
    def boom(url, timeout):
        raise RuntimeError("socket exploded")

    monkeypatch.setattr(mp, "_get_json", boom)
    assert mp.lookup_drug("lisinopril") is None


def test_get_json_returns_none_on_errors(monkeypatch):
    import io
    import socket
    import urllib.request

    url = "https://rxnav.nlm.nih.gov/REST/rxcui.json?name=x"

    def refuse(*args, **kwargs):
        raise OSError("connection refused")

    def slow(*args, **kwargs):
        raise socket.timeout("timed out")

    def not_json(*args, **kwargs):
        return io.BytesIO(b"<html>maintenance</html>")

    for fake_urlopen in (refuse, slow, not_json):
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        assert REAL_GET_JSON(url, 1.0) is None


def test_find_references_survives_network_failure(monkeypatch):
    monkeypatch.setattr(mp, "_get_json", FakeNet(fail=True))
    refs = mp.find_references(SAMPLE_TEXT, SAMPLE_LABS)
    assert "Potassium Blood Test" in titles(refs)
    assert "High Blood Pressure" in titles(refs)
    assert all(r.kind != "drug" for r in refs)


# --------------------------------------------------------------------------- search / learn more

def test_search_topics_bm25():
    results = mp.search_topics("high blood pressure", limit=3)
    assert len(results) == 3
    assert results[0][1]["title"] == "High Blood Pressure"
    assert all(isinstance(score, float) for score, _ in results)
    assert results[0][0] >= results[1][0] >= results[2][0]
    assert mp.search_topics("", limit=3) == []
    assert mp.search_topics("zzyzx qwertyuiop", limit=3) == []


@pytest.mark.parametrize("term, expected", [
    ("shortness of breath", "Breathing Problems"),
    ("kidney function", "Kidney Tests"),
    ("blood clot in the lung", "Pulmonary Embolism"),
    ("iron deficiency", "Anemia"),
])
def test_bm25_accepts_good_matches(term, expected):
    assert topic_title(mp._bm25_concept(term)) == expected


@pytest.mark.parametrize("term", [
    "dietary counseling", "follow up", "routine", "normal", "therapy", "panel", "patient education",
    "1 month", "discharge", "blorfin zzyzx", "medication adherence",
])
def test_bm25_rejects_junk(term):
    assert mp._bm25_concept(term) is None


def test_learn_more_threshold_behaviour(net):
    out = mp.learn_more_for_concepts([
        {"term": "ACE inhibitor", "kind": "medication"},
        {"term": "blorfin zzyzx", "kind": "condition"},
        {"term": "dietary counseling", "kind": "procedure"},
        {"term": "shortness of breath", "kind": "condition"},
    ], exclude_urls=set())
    found = {r.matched: r for r in out}
    assert "blorfin zzyzx" not in found and "dietary counseling" not in found
    if "ACE inhibitor" in found:  # something reasonable or nothing
        assert found["ACE inhibitor"].title in ("Blood Pressure Medicines", "Lisinopril")
    assert found["shortness of breath"].title == "Breathing Problems"


def test_learn_more_skips_covered_terms_and_excluded_urls(net):
    refs = mp.find_references(SAMPLE_TEXT, SAMPLE_LABS)
    exclude = {r.url for r in refs}
    out = mp.learn_more_for_concepts([
        {"term": "hyperkalemia", "kind": "condition"},     # explained by Potassium Blood Test
        {"term": "hypertension", "kind": "condition"},     # already a reference
        {"term": "lisinopril", "kind": "medication"},      # already a reference
        {"term": "metformin", "kind": "medication"},       # live lookup (mocked)
        {"term": "A1C", "kind": "test"},                   # lab-test page
        {"term": "a1c", "kind": "test"},                   # duplicate term
    ], exclude_urls=exclude)
    assert titles(out) == ["Metformin", "Hemoglobin A1C (HbA1c) Test"]
    assert out[0].kind == "drug" and out[0].matched == "metformin"
    assert out[1].kind == "lab_test" and out[1].matched == "A1C"
    assert not (exclude & {r.url for r in out})


def test_learn_more_caps_and_handles_bad_input(net):
    concepts = [{"term": t, "kind": "condition"} for t in
                ("asthma", "gout", "migraine", "psoriasis", "glaucoma", "anemia", "insomnia", "acne")]
    assert len(mp.learn_more_for_concepts(concepts, set(), max_items=6)) == 6
    assert mp.learn_more_for_concepts(concepts, set(), max_items=0) == []
    assert mp.learn_more_for_concepts([{"term": None}, {}, {"kind": "test"}], set()) == []


def test_learn_more_medication_falls_back_when_lookup_fails(monkeypatch):
    monkeypatch.setattr(mp, "_get_json", FakeNet(fail=True))
    out = mp.learn_more_for_concepts([{"term": "lisinopril", "kind": "medication"},
                                      {"term": "statin", "kind": "medication"}], set())
    assert titles(out) == ["Statins"]


# --------------------------------------------------------------------------- review fixes (retrieval)

def meds_page(slug):
    return f"https://medlineplus.gov/druginfo/meds/{slug}.html"


def drug(rxcui, title, slug):
    return (rxcui, title, meds_page(slug), f"{title} is a medicine.")


# 0
def test_ckd_epi_equation_name_is_not_a_ckd_diagnosis():
    text = "eGFR (CKD-EPI 2021) 89 mL/min/1.73m2 (>=60)\nKidney function is normal."
    assert mp.find_references(text, []) == []
    out = mp.learn_more_for_concepts([{"term": "CKD-EPI equation", "kind": "test"}], set())
    assert "Chronic Kidney Disease" not in titles(out)
    # a stated CKD diagnosis next to the equation name still counts
    assert titles(mp.find_references("Assessment: CKD stage 3a. eGFR by CKD-EPI 48.", [])) == [
        "Chronic Kidney Disease"]


# 1
def test_lymph_nodes_heading_is_not_lymphatic_disease():
    assert mp.find_references("LYMPH NODES: No enlarged lymph nodes.\nLIVER: Normal.", []) == []
    assert titles(mp.find_references("Enlarged lymph nodes in the left neck.", [])) == ["Lymphatic Diseases"]
    assert titles(mp.find_references("Cervical lymphadenopathy.", [])) == ["Lymphatic Diseases"]
    for sample in ("radiology_thyroid_ultrasound_nodule.txt", "radiology_ct_abdomen_kidney_stone.txt"):
        assert "Lymphatic Diseases" not in titles(mp.find_references(read_sample(sample), [])), sample


# 2
DRUG_FEEDS = {
    "docusate sodium": ("71722", [("Senna", meds_page("a601112")), ("Stool Softeners", meds_page("a601113"))]),
    "docusate": ("82003", [("Stool Softeners", meds_page("a601113"))]),
    "lasix": ("202991", [("Furosemide", meds_page("a682858")), ("Furosemide Injection", meds_page("a618026"))]),
    "ferrous sulfate": ("4450", [("Iron Supplements", meds_page("a682778"))]),
}


def test_rxcui_naming_different_drugs_is_not_guessed(monkeypatch):
    fake = FakeNet(drugs={}, feeds=DRUG_FEEDS)
    monkeypatch.setattr(mp, "_get_json", fake)
    assert mp.lookup_drug("Docusate sodium") is None          # not Senna
    assert titles(mp._confirm_drugs([["Docusate sodium", "Docusate"]])) == ["Stool Softeners"]
    assert mp.lookup_drug("Lasix").title == "Furosemide"       # route variants of one drug
    assert mp.lookup_drug("ferrous sulfate").title == "Iron Supplements"   # a single page
    # learn more retries the medication without its salt word
    out = mp.learn_more_for_concepts([{"term": "docusate sodium", "kind": "medication"}], set())
    assert titles(out) == ["Stool Softeners"]


def test_combination_brand_products_keep_an_ingredient_page(monkeypatch):
    """Percocet / Hyzaar have no other name to try: their Connect pages are the ingredients."""
    feeds = {
        "percocet": ("42844", [("Acetaminophen", meds_page("a681004")), ("Oxycodone", meds_page("a682132"))]),
        "hyzaar": ("261962", [("Hydrochlorothiazide", meds_page("a682571")), ("Losartan", meds_page("a695008"))]),
    }
    monkeypatch.setattr(mp, "_get_json", FakeNet(drugs={}, feeds=feeds))
    assert mp.lookup_drug("Percocet").title == "Acetaminophen"
    assert mp.lookup_drug("Hyzaar").title == "Hydrochlorothiazide"
    out = mp.learn_more_for_concepts([{"term": "Percocet", "kind": "medication"}], set())
    assert titles(out) == ["Acetaminophen"]


# 3
def test_return_precautions_are_not_the_patients_findings():
    text = ("DIAGNOSIS: Acute appendicitis\n"
            "RETURN TO THE ED FOR: severe shortness of breath, chest pain, or fainting.")
    assert titles(mp.find_references(text, [])) == ["Appendicitis"]
    header = ("CALL THE CLINIC OR RETURN TO THE ED FOR:\n- Fever or shaking chills\n"
              "- Sudden shortness of breath\n\nDIAGNOSIS: asthma")
    assert titles(mp.find_references(header, [])) == ["Asthma"]
    # an inline precaution covers its own sentence only: the next line is still scanned
    assert titles(mp.find_references("- Call 911 for chest pain.\n- Limit sodium to 2,000 mg per day.", [])) == ["Sodium"]
    assert titles(mp.find_references("Call 911 for chest pain.\nLimit sodium to 2,000 mg per day.", [])) == ["Sodium"]
    assert "Chest Pain" not in titles(mp.find_references(read_sample("discharge_copd_exacerbation.txt"), []))
    assert "Breathing Problems" not in titles(mp.find_references(read_sample("discharge_appendectomy.txt"), []))


def test_unpunctuated_precaution_does_not_swallow_the_next_lines():
    """Without a full stop the inline span continues only onto wrapped lines, and never
    past a new heading."""
    text = ("Call 911 for chest pain or trouble breathing\nHypertension - continue lisinopril\n"
            "Type 2 diabetes - continue metformin")
    assert titles(mp.find_references(text, [])) == ["High Blood Pressure", "Diabetes Type 2"]
    text = ("Return to clinic for follow-up in 1 week\nDischarge Diagnosis: Community-acquired pneumonia\n"
            "Secondary: hypertension")
    assert titles(mp.find_references(text, [])) == ["Pneumonia", "High Blood Pressure"]
    assert titles(mp.find_references("Seek care if symptoms worsen\nProblem list: asthma, GERD", [])) == [
        "Asthma", "GERD"]
    # wrapped lines (lowercase start / line ending in "or") still belong to the precaution
    text = "Call 911 if you have chest pain or\nShortness of breath\nHypertension - continue lisinopril"
    assert titles(mp.find_references(text, [])) == ["High Blood Pressure"]
    # a heading-style list also ends at a new "Heading: value" line
    text = "RETURN PRECAUTIONS:\nchest pain, shortness of breath\nDischarge Diagnosis: pneumonia"
    assert titles(mp.find_references(text, [])) == ["Pneumonia"]


# 4
def test_interpretation_legend_rows_do_not_create_topics():
    refs = titles(mp.find_references(read_sample("lab_diabetes_a1c_followup.txt"), []))
    assert "Prediabetes" not in refs and "Diabetes Type 2" in refs
    for row in ("  5.7-6.4%   Prediabetes", ">=6.5%\tDiabetes",
                "  <30 mg/g normal | 30-300 mg/g moderately increased | >300 mg/g severely increased"):
        assert mp._LEGEND_ROW_RE.match(row), row
    for prose in ("4 days of productive cough, fever", "10-14 days of amoxicillin for pneumonia",
                  "Glucose  112 mg/dL (70-99)"):
        assert not mp._LEGEND_ROW_RE.match(prose), prose


# 5
def test_negative_test_results_are_negated():
    assert titles(mp.find_references("Influenza, COVID-19, and RSV tests negative. Diagnosis: asthma.", [])) == [
        "Asthma"]
    for text in ("COVID-19 PCR: not detected.", "HIV test was negative.", "COVID-19 PCR negative.",
                 "Influenza A and B antigens negative.", "Hepatitis B surface antigen negative."):
        assert mp.find_references(text, []) == [], text
    assert titles(mp.find_references("Pneumonia with blood cultures negative.", [])) == ["Pneumonia"]
    assert titles(mp.find_references("Pneumonia, blood cultures negative.", [])) == ["Pneumonia"]
    assert titles(mp.find_references("COVID-19 PCR positive.", [])) == ["COVID-19 (Coronavirus Disease 2019)"]
    assert titles(mp.find_references("Diabetes: negative for retinopathy.", [])) == ["Diabetes"]


@pytest.mark.parametrize("text, expected", [
    ("CXR showed right lower lobe pneumonia and blood cultures were negative.", "Pneumonia"),
    ("Admitted for pneumonia and blood cultures negative.", "Pneumonia"),
    ("Diagnosed with UTI and urine culture was negative.", "Urinary Tract Infections"),
    ("Treated for cellulitis and wound culture negative.", "Cellulitis"),
])
def test_a_negative_result_after_and_is_not_about_the_diagnosis(text, expected):
    """The list tail only negates the term when a test word follows it ("... and RSV tests")."""
    assert titles(mp.find_references(text, [])) == [expected]


def test_negative_test_lists_still_negate_every_item():
    for text in ("Legionella and pneumococcal antigens negative.", "Influenza A/B and RSV PCR negative.",
                 "Urine Legionella\nand pneumococcal antigens negative."):
        assert mp.find_references(text, []) == [], text


# 6
def test_negation_carries_over_hard_line_wraps():
    text = "Knee pain for a week. No chest pain,\npalpitations, headache, or vision changes."
    assert "Headache" not in titles(mp.find_references(text, []))
    assert mp.find_references("Family history of\nbreast cancer.", []) == []
    # a capitalised next line is a new item, not a wrap
    assert titles(mp.find_references("IMPRESSION\nNo acute fracture\nPneumonia in the right lower lobe", [])) == [
        "Pneumonia"]
    # so is a lowercase one under a short line that already names its own finding
    assert titles(mp.find_references("IMPRESSION:\nno acute fracture\npneumonia in the right lower lobe", [])) == [
        "Pneumonia"]
    # ...but a short line that names nothing yet still wraps (narrow OCR columns)
    assert mp.find_references("No evidence of acute\nfracture.", []) == []
    assert mp.find_references("Patient denies\nchest pain.", []) == []
    assert mp.find_references("No chest pain,\r\npalpitations, headache.", []) == []   # CRLF


# 7
def test_stated_diagnoses_rank_ahead_of_frequent_mentions(monkeypatch):
    dka = mp.find_references(read_sample("discharge_diabetic_ketoacidosis.txt"), [], max_refs=3)
    assert titles(dka) == ["Diabetes Type 1", "Gastroenteritis", "Dehydration"]
    pna = titles(mp.find_references(read_sample("discharge_pneumonia.txt"), []))
    assert pna[:3] == ["Pneumonia", "Respiratory Failure", "High Blood Pressure"]
    # heart failure with its BNP page and all five discharge drugs known: 5 topic slots
    hf_drugs = {
        "furosemide": drug("4603", "Furosemide", "a682858"),
        "metoprolol succinate": drug("221124", "Metoprolol", "a682864"),
        "lisinopril": drug("29046", "Lisinopril", "a692051"),
        "potassium chloride": drug("8591", "Potassium Chloride", "a601085"),
        "atorvastatin": drug("83367", "Atorvastatin", "a600045"),
    }
    monkeypatch.setattr(mp, "_get_json", FakeNet(drugs=hf_drugs))
    hf = mp.find_references(read_sample("discharge_heart_failure.txt"), [lab("BNP", "bnp", "high", 1840, 0, 100)])
    assert [r.kind for r in hf].count("drug") == 2
    assert "Cardiomyopathy" in titles(hf) and titles(hf)[1] == "Heart Failure"


def test_topic_matched_only_by_a_confirmed_drug_name_is_dropped(monkeypatch):
    text = "HOSPITAL COURSE:\nIV insulin drip overnight, then insulin glargine 20 units at bedtime.\n"
    assert "Diabetes Medicines" in titles(mp.find_references(text, []))      # drug unknown: keep
    glargine = {"insulin glargine": drug("274783", "Insulin Glargine (rDNA origin) Injection", "a600027")}
    monkeypatch.setattr(mp, "_get_json", FakeNet(drugs=glargine))
    mp._DRUG_CACHE.clear()   # forget the "unknown drug" answer cached above
    assert titles(mp.find_references(text, [])) == ["Insulin Glargine (rDNA origin) Injection"]


# 8
def test_discharge_list_drugs_come_before_one_off_inpatient_doses(monkeypatch):
    text = read_sample("discharge_appendectomy.txt")
    groups, _, _ = mp._medication_mentions(text, prefer_lists=True)
    first = [g[0].lower() for g in groups]
    assert first[:4] == ["acetaminophen", "ibuprofen", "oxycodone", "docusate sodium"]
    assert set(first[4:]) == {"ceftriaxone", "metronidazole"}
    assert mp._medication_candidates(text)[0] == ["ceftriaxone"]      # default stays in text order
    known = {name: drug(str(i), name.title(), f"a{i}") for i, name in enumerate(
        ("ceftriaxone", "metronidazole", "acetaminophen", "ibuprofen", "oxycodone", "docusate sodium"))}
    monkeypatch.setattr(mp, "_get_json", FakeNet(drugs=known))
    refs = mp.find_references(text, [lab("WBC", "wbc", "high", 14.2, 4.0, 11.0)], max_refs=6)
    assert [r.title for r in refs if r.kind == "drug"] == ["Acetaminophen", "Ibuprofen"]


# 9
def test_dose_regex_handles_wraps_formulation_words_and_insulin_regimen():
    assert mp._medication_candidates("Start ferrous\nsulfate 325 mg daily.") == [["ferrous sulfate", "ferrous"]]
    assert mp._medication_candidates("Start potassium chloride ER 20 mEq daily.") == [["potassium chloride"]]
    assert mp._medication_candidates("Glucagon nasal powder 3 mg as needed.") == [["Glucagon"]]
    assert mp._medication_candidates("Aspirin EC 81 mg daily.") == [["Aspirin"]]
    assert mp._medication_candidates("Start metformin ER 500 mg daily.") == [["metformin"]]
    regimen = ("DISCHARGE INSULIN REGIMEN:\n1. Insulin glargine 22 units at bedtime\n"
               "2. Insulin lispro (rapid-acting): inject before meals\n")
    flat = [n.lower() for g in mp._medication_candidates(regimen) for n in g]
    assert "insulin glargine" in flat and "insulin lispro" in flat
    # the word right before a dose must be a name, not a filler word
    for junk in ("inhale contents of 1 capsule", "(INCREASED from 20 units)", "lunch with 60 g carbs"):
        assert mp._medication_candidates(junk) == [], junk


# 10
def drug_entries(*page_titles):
    return [{"title": {"_value": t}, "link": [{"href": meds_page(f"a{i}")}]} for i, t in enumerate(page_titles)]


def test_route_picks_the_right_per_route_drug_page(monkeypatch):
    def pick(entries, name, hint=()):
        return mp._pick_drug_entry(entries, name, frozenset(hint))["title"]["_value"]

    ipratropium = drug_entries("Ipratropium Nasal Spray", "Ipratropium Oral Inhalation")
    assert pick(ipratropium, "ipratropium") == "Ipratropium Oral Inhalation"
    assert pick(ipratropium, "ipratropium", {"inhalation"}) == "Ipratropium Oral Inhalation"
    assert pick(ipratropium, "ipratropium", {"nasal"}) == "Ipratropium Nasal Spray"
    nicotine = drug_entries("Nicotine Gum", "Nicotine Nasal Spray", "Nicotine Oral Inhalation",
                            "Nicotine Transdermal Patch")
    assert pick(nicotine, "Nicotine", {"transdermal"}) == "Nicotine Transdermal Patch"
    assert pick(drug_entries("Insulin Human Inhalation", "Insulin Human Injection"), "insulin") == \
        "Insulin Human Injection"
    assert pick(drug_entries("Epinephrine Oral Inhalation", "Epinephrine Injection"), "epinephrine") == \
        "Epinephrine Injection"
    fluticasone = drug_entries("Fluticasone Nasal Spray", "Fluticasone Oral Inhalation", "Fluticasone Topical")
    assert pick(fluticasone, "fluticasone", {"inhalation"}) == "Fluticasone Oral Inhalation"

    # the hint comes from the medication's own list item / line
    copd = read_sample("discharge_copd_exacerbation.txt")
    groups, _, hints = mp._medication_mentions(copd)
    by_name = {g[0]: h for g, h in zip(groups, hints)}
    assert by_name["Albuterol"] == {"inhalation"} and by_name["Nicotine"] == {"transdermal"}

    # learn more: document lines that mention the medicine give the route
    feeds = {"ipratropium": ("7213", [("Ipratropium Nasal Spray", meds_page("a614006")),
                                      ("Ipratropium Oral Inhalation", meds_page("a695021"))])}
    monkeypatch.setattr(mp, "_get_json", FakeNet(drugs={}, feeds=feeds))
    concept = [{"term": "ipratropium", "kind": "medication"}]
    assert titles(mp.learn_more_for_concepts(concept, set(), document_text=copd)) == ["Ipratropium Oral Inhalation"]
    nose = "Ipratropium nasal spray, 2 sprays in each nostril twice daily for runny nose."
    assert titles(mp.learn_more_for_concepts(concept, set(), document_text=nose)) == ["Ipratropium Nasal Spray"]


# 11
def test_learn_more_rejects_misleading_bm25_pages():
    assert mp.learn_more_for_concepts([{"term": "blood gas", "kind": "test"}], set()) == []
    assert mp.learn_more_for_concepts([{"term": "arterial blood gas", "kind": "test"}], set()) == []
    for term, expected in (("knee pain", "Knee Injuries and Disorders"), ("hip pain", "Hip Injuries and Disorders"),
                           ("foot pain", "Foot Injuries and Disorders")):
        assert titles(mp.learn_more_for_concepts([{"term": term, "kind": "condition"}], set())) == [expected]
    assert topic_title(mp._bm25_concept("weight loss")) != "Weight Loss Surgery"
    assert topic_title(mp._bm25_concept("kidney transplant")) == "Kidney Transplantation"
    assert topic_title(mp._bm25_concept("knee replacement")) == "Knee Replacement"
    # concept-only synonyms never affect document scanning ("shoulder pain" from surgical gas)
    assert "Shoulder Injuries and Disorders" not in titles(
        mp.find_references(read_sample("discharge_appendectomy.txt"), []))


# 12
def test_ocr_garbled_lab_names_still_get_lab_pages():
    labs = [lab("Gluc0se", None, "high", 132, 70, 99), lab("Potass1um", None, "high", 5.9, 3.5, 5.1),
            lab("Sod ium", None, "normal", 141, 136, 145), lab("eGFR (CKD-EPI)", "egfr", "normal", 82, 60, None)]
    refs = mp.find_references(read_sample("tricky_ocr_noisy_bmp.txt"), labs)
    tests = mp._load_index()["lab_tests"]
    assert [r.url for r in refs] == [tests["glucose"]["url"], tests["potassium"]["url"]]  # most abnormal first
    for name, key in (("Sod ium", "sodium"), ("C02", "co2"), ("Ca1cium", "calcium"), ("Hemog1obin A1c", "a1c")):
        assert mp._lab_key(SimpleNamespace(name=name, canonical=None)) == key, name
    for name in ("Bilirub1n, Direct", "her potass1um was"):
        assert mp._lab_key(SimpleNamespace(name=name, canonical=None)) is None, name


# 13 / 17
WRAPPED_MED_LIST = """DISCHARGE MEDICATIONS:
1. Aspirin 81 mg by mouth daily, take with
   breakfast every morning.
2. Atorvastatin 80 mg at bedtime; avoid grapefruit juice for 3
   months.
3. Metoprolol tartrate 25 mg twice daily; hold if pulse
   below 55.
4. Clopidogrel 75 mg daily for 12 months, do not stop unless
   your cardiologist says so.
5. Lisinopril 5 mg daily.
6. Nitroglycerin 0.4 mg under the tongue as needed for
   chest pain, may repeat every 5 minutes up to 3 doses.
7. Pantoprazole 40 mg daily while on two
   blood thinners.
"""


def test_wrapped_med_list_lines_are_not_new_drugs(no_network):
    groups = mp._medication_candidates(WRAPPED_MED_LIST)
    assert [g[0] for g in groups] == ["Aspirin", "Atorvastatin", "Metoprolol tartrate", "Clopidogrel",
                                      "Lisinopril", "Nitroglycerin", "Pantoprazole"]
    mp.find_references(WRAPPED_MED_LIST, [])
    assert "Pantoprazole" in rxnav_names(no_network)       # still within MAX_DRUG_LOOKUPS
    # the wrapped line stays inside the item's skip span: no Chest Pain / Blood Thinners topics
    assert mp.find_references(WRAPPED_MED_LIST, []) == []
    appendectomy = [n.lower() for g in mp._medication_candidates(read_sample("discharge_appendectomy.txt")) for n in g]
    for junk in ("needed", "severe pain", "pain", "severe"):
        assert junk not in appendectomy, junk


def test_med_list_ends_at_the_next_heading():
    text = ("Home Medications: metformin 500 mg twice daily\n"
            "Past Medical History: hypertension, type 2 diabetes\n"
            "Hospital Course: admitted for community-acquired pneumonia.")
    assert mp._medication_candidates(text) == [["metformin"]]
    assert titles(mp.find_references(text, [])) == ["High Blood Pressure", "Diabetes Type 2", "Pneumonia"]
    # a sentence about medications is not a medication-list header
    assert mp._medication_candidates("Patient tolerating medications well\nAsthma flare today.") == []
    assert mp._medication_candidates("Home Medications\nAmlodipine\n") == [["Amlodipine"]]


# 14
def test_dka_and_thyroid_nodule_get_topics():
    assert "Diabetes Type 1" in titles(mp.find_references(read_sample("discharge_diabetic_ketoacidosis.txt"), []))
    assert "Thyroid Diseases" in titles(mp.find_references(read_sample("radiology_thyroid_ultrasound_nodule.txt"), []))
    assert titles(mp.learn_more_for_concepts([{"term": "DKA", "kind": "condition"}], set())) == ["Diabetes Type 1"]
    assert titles(mp.learn_more_for_concepts([{"term": "thyroid nodule", "kind": "condition"}], set())) == [
        "Thyroid Diseases"]


# 15
@pytest.mark.parametrize("text", [
    "a" + "1" * 19999, "a-" * 10000, "a1-" * 6666, "ab" + " " * 19998, "ab\n" * 6666,
    "Metformin" + "5" * 19991,
])
def test_dose_regex_is_fast_on_pathological_input(text):
    """These took 10-40 s with the old unbounded pattern (quadratic backtracking)."""
    start = time.perf_counter()
    list(mp._DOSE_RE.finditer(text))
    mp.find_references(text, [])
    assert time.perf_counter() - start < 0.5


def test_bounded_dose_regex_still_finds_normal_doses():
    assert mp._medication_candidates("Metformin 500 mg twice daily") == [["Metformin"]]
    assert mp._medication_candidates("Metformin: 500 mg") == [["Metformin"]]
    assert mp._medication_candidates("Levothyroxine 0.075 mg daily") == [["Levothyroxine"]]


# 16
def test_allergy_section_does_not_swallow_following_headings():
    text = ("Allergies: Penicillin (rash)\nDischarge Diagnosis: Community-acquired pneumonia\n"
            "Secondary: Type 2 diabetes, hypertension\nHospital Course: improved on antibiotics.")
    assert {"Pneumonia", "Diabetes Type 2", "High Blood Pressure"} <= set(titles(mp.find_references(text, [])))
    text = "Allergies: NKDA\nChief Complaint: shortness of breath\nReason for admission: heart failure exacerbation"
    assert titles(mp.find_references(text, [])) == ["Breathing Problems", "Heart Failure"]
    problems = "Allergic rhinitis - seasonal\nAsthma, mild intermittent\nHypertension\nType 2 diabetes"
    assert titles(mp.find_references(problems, [])) == ["Hay Fever", "Asthma", "High Blood Pressure", "Diabetes Type 2"]
    text = "Allergies: penicillin\nPast Medical History: hypertension, asthma"
    assert titles(mp.find_references(text, [])) == ["High Blood Pressure", "Asthma"]
    # reactions inside the allergy list are still not the patient's conditions
    assert mp.find_references("Allergies:\nPenicillin: hives\nSulfa: rash", []) == []
    assert mp.find_references("ALLERGIES\nPenicillin - hives\nSulfa - rash", []) == []
    assert titles(mp.find_references("Allergic to sulfa (rash).\nHypertension.", [])) == ["High Blood Pressure"]


def test_allergy_headers_with_a_dash_or_column_gap():
    text = "Allergies - Penicillin (rash), sulfa (hives)\nDiagnosis: pneumonia"
    assert titles(mp.find_references(text, [])) == ["Pneumonia"]
    assert titles(mp.find_references("ALLERGIES    Penicillin (hives)\nDIAGNOSIS: pneumonia", [])) == ["Pneumonia"]
    assert titles(mp.find_references("ALLERGIES\tPenicillin (hives)\nAsthma.", [])) == ["Asthma"]
    # an inline allergy list covers its wrapped / indented lines, not the next row
    assert titles(mp.find_references("ALLERGIES    Penicillin (hives)\nPROBLEMS     Asthma", [])) == ["Asthma"]
    assert titles(mp.find_references("Allergies: Penicillin (rash),\nsulfa (hives)\nAsthma", [])) == ["Asthma"]
    assert titles(mp.find_references("Allergies: Penicillin (rash)\n    Sulfa (hives)\nAsthma", [])) == ["Asthma"]
    assert titles(mp.find_references("Allergic rhinitis - seasonal\nAsthma", [])) == ["Hay Fever", "Asthma"]


# 18
def test_negation_scope_stops_at_termination_terms():
    assert titles(mp.find_references("No fever but CT shows appendicitis.", [])) == ["Appendicitis"]
    assert "Pneumonia" in titles(mp.find_references("Denies chest pain but has pneumonia on chest x-ray.", []))
    assert "COPD" in titles(mp.find_references("Patient denies smoking, now with COPD exacerbation.", []))
    assert "High Blood Pressure" in titles(mp.find_references(
        "Type 2 diabetes mellitus without complications, essential hypertension.", []))
    assert "GERD" in titles(mp.find_references("Asthma without status asthmaticus, GERD, osteoporosis.", []))
    # negations themselves still work
    assert mp.find_references("CT shows no appendicitis. No fever, chills, or chest pain.", []) == []
    assert mp.find_references("No reported chest pain.", []) == []


def test_without_negates_a_whole_or_list():
    """'without X, Y, or Z' negates every item; an ICD 'without ...' qualifier does not."""
    assert titles(mp.find_references("Abdominal pain without nausea, vomiting, or diarrhea.", [])) == [
        "Abdominal Pain"]
    assert titles(mp.find_references("Cough without fever, headache, or chest pain.", [])) == ["Cough"]
    assert titles(mp.find_references("Fever without cough, chills or night sweats.", [])) == ["Fever"]
    # problem lists join new diagnoses with "and": those are not negated
    assert titles(mp.find_references(
        "Type 2 diabetes mellitus without complications, hypertension and hyperlipidemia.", [])) == [
        "Diabetes Type 2", "High Blood Pressure", "Cholesterol"]


# 19
def test_salt_names_try_the_base_drug_not_the_salt():
    assert mp._medication_candidates("Continue metoprolol succinate ER 50 mg daily.") == [
        ["metoprolol succinate", "metoprolol"]]
    assert mp._medication_candidates("Docusate sodium 100 mg twice daily") == [["Docusate sodium", "Docusate"]]
    assert mp._medication_candidates("magnesium oxide 400 mg daily") == [["magnesium oxide"]]
    groups = mp._medication_candidates(read_sample("lab_cbc_iron_deficiency.txt"))
    assert groups == [["ferrous sulfate", "ferrous"]]


def test_line_wrapped_name_keeps_the_topic_on_the_line_above():
    """A pair across a line break must not hide a topic that ends the previous line."""
    assert "Asthma" in titles(mp.find_references("History of asthma\nalbuterol 2 puffs as needed.", []))


UNNUMBERED_MED_LIST = """DISCHARGE MEDICATIONS
Acetaminophen 500 mg every 6 hours as needed for pain
Oxycodone 5 mg every 4 hours as needed for severe pain
Albuterol 2 puffs every 4 hours as needed for wheezing
Fluticasone 50 mcg nasal spray daily
Senna 8.6 mg nightly as needed for constipation
Pantoprazole 40 mg daily for reflux
Apixaban 5 mg twice daily
Metoprolol 25 mg twice daily
"""
UNNUMBERED_DRUGS = ["Acetaminophen", "Oxycodone", "Albuterol", "Fluticasone", "Senna", "Pantoprazole", "Apixaban",
                    "Metoprolol"]


def test_one_drug_per_line_list_does_not_glue_the_line_above(no_network):
    """'...as needed for pain\\nOxycodone 5 mg' is Oxycodone, not 'pain Oxycodone'."""
    assert mp._medication_candidates(UNNUMBERED_MED_LIST) == [[d] for d in UNNUMBERED_DRUGS]
    # same list in lowercase: the line above has its own dose, so it is a separate medicine
    assert mp._medication_candidates(UNNUMBERED_MED_LIST.lower()) == [[d.lower()] for d in UNNUMBERED_DRUGS]
    # every drug fits in the lookup budget now (junk names used to crowd out the last three)
    mp.find_references(UNNUMBERED_MED_LIST, [])
    assert rxnav_names(no_network) and set(rxnav_names(no_network)) == set(UNNUMBERED_DRUGS)
    # a name wrapped in the middle is still joined
    assert mp._medication_candidates("Start ferrous\r\nsulfate 325 mg daily.") == [["ferrous sulfate", "ferrous"]]


# 20
def test_decomposed_unicode_text_matches_topics():
    import unicodedata

    nfd = unicodedata.normalize("NFD", "Ménière's disease")
    hits = mp._scan_topics(nfd)
    assert [topic_title(h.topic) for h in hits] == ["Meniere's Disease"]
    assert hits[0].matched == nfd


# 21
def test_index_builder_accepts_a_bare_output_filename(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_index", os.path.join(ROOT, "scripts",
                                                                              "build_medlineplus_index.py"))
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    xml = tmp_path / "mplus_topics_2026-01-02.xml"
    xml.write_text(
        '<health-topics date-generated="01/02/2026 03:04:05">'
        '<health-topic title="Asthma" url="https://medlineplus.gov/asthma.html" id="1" language="English"'
        ' meta-desc="Asthma is a chronic disease that affects your airways and makes breathing harder.">'
        '<also-called>Reactive airway disease</also-called>'
        '<full-summary>&lt;p&gt;Asthma is a chronic lung disease.&lt;/p&gt;</full-summary>'
        '</health-topic></health-topics>', encoding="utf-8")
    monkeypatch.setattr(build, "build_lab_tests", lambda topics: {})   # no network
    monkeypatch.chdir(tmp_path)
    assert build.main(["--xml", str(xml), "--out", "index.json"]) == 0
    with open(tmp_path / "index.json", encoding="utf-8") as fh:
        assert [t["title"] for t in json.load(fh)["topics"]] == ["Asthma"]
