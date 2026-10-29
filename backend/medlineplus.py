"""MedlinePlus retrieval layer (the "R" in Clearchart's RAG).

Clearchart explains clinical documents in plain language. To keep explanations
grounded, every request is paired with short passages from MedlinePlus (U.S.
National Library of Medicine) that the LLM can cite, and the UI shows the same
pages as "learn more" cards.

Data
    backend/data/medlineplus_index.json is built offline by
    scripts/build_medlineplus_index.py: ~1,000 English health topics (title,
    synonyms, description, ~900 chars of summary) plus one MedlinePlus lab-test page
    per canonical lab key (found via MedlinePlus Connect + LOINC). Medication pages
    are looked up live (RxNav name -> RxCUI, then MedlinePlus Connect) and cached.

Retrieval design
    * Lexical, not embeddings. Clinical documents name things precisely
      ("hyperkalemia", "lisinopril 10 mg", "LDL 158"), and MedlinePlus ships curated
      synonyms (Hypertension -> High Blood Pressure, Hyperkalemia -> Potassium). Exact
      phrase matching plus BM25 is precise on such terms, needs no extra API key or
      model download, is deterministic (the same document always yields the same
      references, which keeps citations and tests stable) and runs offline in
      milliseconds. Embeddings would add a dependency and tend to return
      "semantically close" but wrong pages for short medical terms.
    * find_references() runs BEFORE the LLM call and picks the passages the model
      may cite, in priority order:
        1. lab-test pages for abnormal labs (most abnormal first; labs that share a
           page, e.g. LDL/HDL/total cholesterol, are merged),
        2. health topics whose title/synonym appears in the document: word-boundary
           n-gram match on normalised tokens (case, accents, punctuation, plurals),
           leftmost-longest, no overlaps. Precision guards: generic topics
           (STOP_TOPICS), common-word/slang/ambiguous synonyms (STOP_PHRASES, short
           acronyms unless allowlisted), context blocks ("ST depression", "CKD-EPI"),
           a NegEx-style check ("denies chest pain", "family history of", "tests
           negative"), medication/allergy lines, return precautions ("call 911 for
           chest pain") and legend rows ("5.7-6.4%  Prediabetes") skipped, and topics
           made redundant by a lab page ("Potassium" when "Potassium Blood Test" is
           included) dropped. Topics named under a DIAGNOSIS/IMPRESSION heading rank
           first,
        3. medications found next to a dose ("metformin 500 mg", not "12 g/dL") or
           under a MEDICATIONS header (discharge list first), confirmed through RxNav
           + MedlinePlus Connect (concurrent, capped, with timeouts).
      A few slots are reserved for topics and drugs so a long lab panel cannot
      crowd out the diagnosis and medication pages.
    * learn_more_for_concepts() runs AFTER the LLM call on the concepts the model
      extracted: lab page / exact synonym match -> live drug lookup (medications)
      -> BM25 over topics (title x3, synonyms x2, description, text) with a
      coordination requirement, a calibrated score threshold and a "query terms
      must hit the title/synonyms" rule -> a known phrase inside the term. An
      unknown or generic term returns nothing rather than a random page.

All network access goes through _get_json(), which never raises (returns None on
any failure) and is the single seam tests monkeypatch.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import re
import ssl
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

INDEX_PATH = Path(__file__).resolve().parent / "data" / "medlineplus_index.json"

RXNAV_RXCUI_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
CONNECT_URL = "https://connect.medlineplus.gov/service"
RXNORM_OID = "2.16.840.1.113883.6.88"

DRUG_TIMEOUT_S = 4.0          # per HTTP request
DRUG_PHASE_BUDGET_S = 9.0     # wall-clock cap for all drug lookups in one call
MAX_DRUG_LOOKUPS = 12         # per document
DRUG_TEXT_CHARS = 700


@dataclass
class Reference:
    title: str
    url: str
    snippet: str      # 1-2 sentences for UI cards
    text: str         # grounding passage (~600-900 chars) passed to the LLM
    kind: str         # "lab_test" | "topic" | "drug"
    matched: str      # the term from the document/concept that triggered it


# =============================================================================
# Curated lists (built by reading the MedlinePlus topic list; see module docstring)
# =============================================================================

# Topics never returned: too generic to help a patient understand a specific
# document, or about demographics/admin rather than a condition, test or treatment.
# Their phrases ("medications", "pediatrics", "vital signs", "family history",
# "insurance", "pain") appear in almost every clinical document.
STOP_TOPICS = frozenset({
    # admin / health-care system
    "Medicines", "Health Checkup", "Baby Health Checkup", "Women's Health Checkup",
    "Health Insurance", "Medicare", "Medicaid", "Managed Care", "Health Facilities",
    "Health Occupations", "Choosing a Doctor or Health Care Service", "Talking With Your Doctor",
    "Patient Rights", "Patient Safety", "Personal Health Records", "Telehealth",
    "Evaluating Health Information", "Understanding Medical Research", "Medical Ethics",
    "Financial Assistance", "Health Literacy", "Health Statistics", "Health Disparities",
    "Health Fraud", "Health Screening", "Laboratory Tests", "Diagnostic Imaging", "Vital Signs",
    "Family History", "Drug Safety", "Medication Errors", "Rare Diseases", "Clinical Trials",
    # demographic / life-stage umbrella topics
    "Women's Health", "Men's Health", "Children's Health", "Teen Health", "Toddler Health",
    "Older Adult Health", "Healthy Aging", "College Health", "School Health", "Rural Health Concerns",
    "Global Health", "Environmental Health", "Occupational Health",
    "Occupational Health for Health Care Providers", "Veterans and Military Health",
    "Veterans and Military Family Health", "Child Development", "Teen Development",
    "Toddler Development", "Infant and Newborn Development", "Child Nutrition", "Toddler Nutrition",
    "Infant and Newborn Nutrition", "Nutrition for Older Adults", "Exercise for Children",
    "Exercise for Older Adults", "Child Safety", "Child Mental Health", "Teen Mental Health",
    "Older Adult Mental Health", "Sexual Health", "Teen Sexual Health",
    # generic umbrella topics
    "Healthy Living", "Nutrition", "Diets", "Dietary Proteins", "Vitamins", "Minerals", "Hormones",
    "Anatomy", "Body Weight", "Weight Control", "Safety", "Disabilities", "Caregivers",
    "Caregiver Health", "Parenting", "Family Issues", "Mental Health", "Mental Disorders",
    "How to Improve Mental Health", "Pain", "Blood", "Surgery", "Pregnancy", "Infectious Diseases",
    "Noise",
})

# Individual phrases (normalised) never used for matching: everyday words, slang
# and words with a different common clinical meaning ("joint" -> Cannabis, "ice" ->
# Methamphetamine, "stress" in "stress test", "gas" in "blood gas", "falls within").
STOP_PHRASES = frozenset({
    # street names
    "joint", "grass", "pot", "weed", "bud", "dope", "hash", "blunt", "reefer", "ganja", "mary jane",
    "blow", "coke", "crack", "flake", "snow", "soda cot", "coca", "big h", "black tar", "chiva",
    "hell dust", "horse", "junk", "smack", "thunder", "chalk", "crystal", "glass", "ice", "speed",
    "tina", "meth",
    # everyday words / admin words
    "delivery", "labor", "period", "water", "sex", "statistics", "insurance", "operation", "shots",
    "fitness", "driving", "accidents", "accident", "abuse", "violence", "teeth", "veins", "shoes",
    "pets", "fertility", "privacy", "careers", "proteins", "lipids", "fiber", "bulk", "drinking",
    "imaging", "pollution", "prevention", "aging", "temperature", "pulse", "sugar", "stool", "poop",
    "pee", "feces", "urine", "urination", "ulcer", "ulcers", "cramps", "fungi", "cutting", "dip",
    "makeup", "nursing", "battery", "injuries", "side effects", "medications", "drugs", "screening",
    "prescription drugs", "drug information", "blood tests", "blood cells", "second opinion",
    "informed consent", "oncology", "tumor", "neoplasms", "thirst", "gas", "stress", "tears",
    "memory", "falls", "rehab", "germs", "hygiene", "palsy", "aids", "the runs", "the trots",
    "the clap", "clap", "crabs", "seniors", "swelling", "lumbago", "cold", "hot", "trich", "endo",
    "fibro", "crypto", "mono", "strep", "staph", "pelvis", "patella", "meniscus", "coccyx",
    "lymph node",  # anatomy / radiology heading ("LYMPH NODES: none enlarged"), not a disease
    "semen", "donors", "transplantation", "stoma", "colitis", "dermatitis", "neuropathy",
    "hemangioma", "nevus", "posture", "pediatrics", "hospitals", "prep", "braces", "glasses",
    "abnormalities", "ophthalmology", "surrogate", "walker", "walkers", "children", "health",
    "youth", "recreational", "allergies",  # "Allergies: NKDA" heads almost every note
})

# Multi-word expressions that must not produce a topic even though a shorter phrase
# inside them would ("ST depression" on an ECG is not Depression). They take part in
# leftmost-longest matching and then map to nothing.
BLOCK_PHRASES = frozenset({
    "st depression", "st segment depression", "respiratory depression", "cns depression",
    "bowel prep",
    "ckd epi",  # eGFR equation name ("eGFR (CKD-EPI 2021) 89"), not a CKD diagnosis
    # a lab test, not Gangrene (BM25 would match "gas" in the synonym "Gas Gangrene")
    "blood gas", "arterial blood gas", "venous blood gas", "capillary blood gas",
})

# Short (< 4 char) single-token synonyms are skipped in documents ("MS" = milliseconds,
# "MD" = doctor, "ED" = emergency department, "ALL" = all) except these.
SHORT_ALLOWLIST = frozenset({
    "flu", "mri", "uti", "htn", "hbp", "chf", "ckd", "dvt", "tia", "cva", "hiv", "hpv", "ibs",
    "aml", "cll", "cml", "tb", "cbc", "icu", "aaa", "rsv", "sle", "hcv", "hbv", "hav", "hsv",
    "cmv", "mi", "bph", "osa", "ocd", "pid", "pms", "rls", "tbi", "tmj", "std", "sti", "hrt",
    "ivf", "iud", "afm", "cjd", "mci", "mds", "pku", "hdl", "ldl", "a1c", "pci", "ekg", "ecg", "egd",
    "dka",
})

# Extra clinical shorthand -> topic title (MedlinePlus synonyms are consumer-oriented).
EXTRA_SYNONYMS: dict[str, list[str]] = {
    "Diabetes Type 2": ["t2dm", "dm2", "dm type 2", "type 2 dm", "niddm", "type ii dm"],
    # DKA is explained on the Diabetes Type 1 page (Diabetes Complications never mentions it)
    "Diabetes Type 1": ["t1dm", "dm1", "type 1 dm", "iddm", "diabetic ketoacidosis", "dka"],
    "Thyroid Diseases": ["thyroid nodule", "ti rads"],
    # findings, not the anatomy word "lymph nodes" (a STOP_PHRASE); "no enlarged lymph
    # nodes" is still caught by the negation check
    "Lymphatic Diseases": ["lymphadenopathy", "enlarged lymph node", "swollen lymph node"],
    "Sleep Apnea": ["osa", "obstructive sleep apnea"],
    "Cholesterol": ["hyperlipidemia", "dyslipidemia", "hld", "high cholesterol", "hypercholesterolemia",
                    "familial hypercholesterolemia", "familial combined hyperlipidemia"],
    "Cholesterol Levels: What You Need to Know": ["lipid panel", "lipid profile"],
    "Atrial Fibrillation": ["a fib", "afib"],
    "Heart Attack": ["stemi", "nstemi"],
    "Heart Failure": ["hfref", "hfpef", "chf"],
    "Pneumonia": ["pna"],
    "Sodium": ["hypernatremia"],
    "Hypothyroidism": ["hypothyroid"],
    "Hyperthyroidism": ["hyperthyroid"],
    "A1C": ["glycated hemoglobin", "glycosylated hemoglobin", "hgb a1c", "hb a1c", "hemoglobin a1c"],
    "Blood Pressure Medicines": [
        "ace inhibitor", "angiotensin converting enzyme inhibitor", "arb", "angiotensin receptor blocker",
        "angiotensin ii receptor blocker", "beta blocker", "calcium channel blocker", "antihypertensive",
        "antihypertensive medication", "water pill", "diuretic",
    ],
    "Blood Thinners": ["antiplatelet", "antiplatelet medication", "anticoagulation"],
    "Pain Relievers": ["nsaid", "nonsteroidal anti inflammatory drug"],
    "Statins": ["statin therapy", "hmg coa reductase inhibitor"],
    "Peripheral Nerve Disorders": ["peripheral neuropathy", "neuropathy"],
    "Acute Bronchitis": ["bronchitis"],
    "Eczema": ["atopic dermatitis"],
    "GERD": ["acid reflux", "reflux"],
    "Deep Vein Thrombosis": ["venous thrombosis"],
    "Obesity": ["overweight"],
    "Sepsis": ["septic shock"],
    "Hay Fever": ["allergic rhinitis"],
    "Osteoporosis": ["bone loss"],
    "Low Blood Pressure": ["orthostatic hypotension"],
    "Heart Valve Diseases": ["heart murmur"],
    "Muscle Cramps": ["leg cramps"],
    "Gluten Sensitivity": ["gluten intolerance"],
    "Heart Diseases": ["cardiovascular disease"],
    "Prediabetes": ["insulin resistance"],
    # common test / procedure names
    "Mammography": ["mammogram", "screening mammogram", "diagnostic mammogram"],
    "Heart Health Tests": ["ekg", "ecg", "electrocardiogram", "echocardiogram", "stress test",
                           "cardiac stress test", "holter monitor", "cardiac catheterization"],
    "Endoscopy": ["egd", "upper endoscopy", "esophagogastroduodenoscopy"],
    "Cervical Cancer Screening": ["pap smear", "pap test"],
    "Bone Density": ["dexa scan", "dxa scan", "bone density scan", "bone density test"],
    # ambiguous MedlinePlus synonyms resolved to the most useful page
    "Adrenal Gland Disorders": ["adrenal insufficiency"],
    "Limb Loss": ["amputation"],
    "Collapsed Lung": ["pneumothorax"],
    "Drug Use and Addiction": ["substance use"],
    "Peripheral Arterial Disease": ["claudication", "intermittent claudication"],
    "Vaccines": ["vaccination"],
    "Abdominal Pain": ["stomach ache"],
    "Hearing Disorders and Deafness": ["deafness"],
    "Nursing Homes": ["long term care"],
    "Carotid Artery Disease": ["carotid endarterectomy"],
    "Pituitary Disorders": ["acromegaly"],
    "Yeast Infections": ["vaginal yeast infection"],
    "Preterm Labor": ["premature labor"],
    "Primary Ovarian Insufficiency": ["ovarian insufficiency"],
}

# Synonyms used only for LLM concepts, never for scanning documents: "shoulder pain"
# after laparoscopy (gas pain) is no reason to cite Shoulder Injuries and Disorders, but
# a "knee pain" concept should land on the knee page rather than Knee Replacement.
CONCEPT_SYNONYMS: dict[str, list[str]] = {
    "Foot Injuries and Disorders": ["foot pain"],
    "Knee Injuries and Disorders": ["knee pain"],
    "Hip Injuries and Disorders": ["hip pain"],
    "Shoulder Injuries and Disorders": ["shoulder pain"],
    "Ankle Injuries and Disorders": ["ankle pain"],
    "Hand Injuries and Disorders": ["hand pain"],
}

# When a phrase belongs to several topics, these extra synonyms win; anything else
# that stays ambiguous (e.g. "blood pressure" -> High/Low Blood Pressure/Vital Signs)
# is dropped.
_OVERRIDE_LEVEL = -1
_BLOCK_LEVEL = -2   # BLOCK_PHRASES: matched like any phrase, but map to no topic
_BLOCKED = -1       # phrase-table value for a blocked phrase

# Lab key -> topics that only restate what the lab-test page explains. Skipped when
# that lab's page is included, or when the lab is listed (any flag) and the topic was
# matched by the analyte's own name (e.g. "Calcium 9.4" -> Calcium topic).
LAB_ANALYTE_TOPICS: dict[str, tuple[str, ...]] = {
    "glucose": ("Blood Glucose",),
    "a1c": ("A1C",),
    "sodium": ("Sodium",),
    "potassium": ("Potassium",),
    "calcium": ("Calcium",),
    "chloride": ("Fluid and Electrolyte Balance",),
    "co2": ("Fluid and Electrolyte Balance",),
    "anion_gap": ("Fluid and Electrolyte Balance",),
    "total_cholesterol": ("Cholesterol", "Cholesterol Levels: What You Need to Know"),
    "ldl": ('LDL: The "Bad" Cholesterol', "Cholesterol", "Cholesterol Levels: What You Need to Know"),
    "hdl": ('HDL: The "Good" Cholesterol', "Cholesterol", "Cholesterol Levels: What You Need to Know"),
    "triglycerides": ("Triglycerides", "Cholesterol Levels: What You Need to Know"),
    "ferritin": ("Iron",),
    "vitamin_d": ("Vitamin D",),
    "bun": ("Kidney Tests",),
    "creatinine": ("Kidney Tests",),
    "egfr": ("Kidney Tests",),
    "uacr": ("Kidney Tests", "Urinalysis"),
    "tsh": ("Thyroid Tests",),
    "free_t4": ("Thyroid Tests",),
    "wbc": ("Blood Count Tests",),
    "rbc": ("Blood Count Tests",),
    "hemoglobin": ("Blood Count Tests",),
    "hematocrit": ("Blood Count Tests",),
    "mcv": ("Blood Count Tests",),
    "platelets": ("Blood Count Tests",),
}



# =============================================================================
# Text normalisation
# =============================================================================

# Combining accents (Unicode category Mn) are not \w in Python's re, so decomposed (NFD)
# text, where an accent is a separate character after its letter (PDF text layers, macOS
# copy/paste), would split "Meniere" at every accent. Letting marks continue a word keeps
# spans exact on the original text; _fold() strips the marks from the token itself.
_COMBINING_MARKS = r"\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f"
_WORD_CHAR = rf"[^\W_][{_COMBINING_MARKS}]*"
_WORD_RE = re.compile(rf"(?:{_WORD_CHAR})+(?:['’](?:{_WORD_CHAR})+)?", re.UNICODE)


def _fold(text: str) -> str:
    """Lower-case and strip accents (Sjögren -> sjogren)."""
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower()


def _stem(tok: str) -> str:
    """Drop possessive/apostrophes and strip plurals lightly (input is already folded)."""
    if "'" in tok or "’" in tok:
        tok = re.sub(r"['’]s$", "", tok).replace("'", "").replace("’", "")
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith(("ches", "shes", "sses", "xes", "zes")):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def _tokens_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Normalised tokens with their character spans in the original text."""
    out = []
    for m in _WORD_RE.finditer(text or ""):
        tok = _stem(_fold(m.group(0)))
        if tok:
            out.append((tok, m.start(), m.end()))
    return out


def _tokens(text: str) -> list[str]:
    """Normalised tokens; same result as _tokens_with_spans but folds the text in one go."""
    return [t for t in (_stem(w) for w in _WORD_RE.findall(_fold(text or ""))) if t]


def _word_set(words: str) -> frozenset[str]:
    """A whitespace-separated word list, normalised the same way as document tokens."""
    return frozenset(_stem(_fold(w)) for w in words.split())


_BM25_STOPWORDS = _word_set("""
a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has
have having he her here hers him his how i if in into is it its itself just me more most my no nor not
now of off on once only or other our out over own same she should so some such than that the their
them then there these they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours patient patients
""")

# Words that are never (part of) a medication name in dose / med-list extraction.
_DRUG_STOPWORDS = _word_set("""
a an and the of to for from into than via with without by on at in or as per plus then also is was be
take takes taking took give given start started starting continue continued continuing resume resumed
stop stopped hold
held increase increased decrease decreased reduce reduced change changed switch switched add added
discontinue discontinued restart prescribed prescription refill refills dispense sig qty quantity
dose doses dosage daily twice once nightly weekly monthly every each day days week weeks hour hours
morning evening night bedtime meals meal mouth oral orally po iv im sc subq subcutaneous bid tid qid
qd qhs prn q4h q6h q8h q12h ac pc hs tablet tablets tab tabs capsule capsules cap caps pill pills
puff puffs drop drops unit units mg mcg ml meq gram grams patient patients pt new home current
currently active none no not other others use using apply inject inhale mix total about approx
approximately around over under up down until after before x recommend recommended consider trial
medication medications meds medicine medicines list allergies allergy instructions follow return
level levels value values result results range normal high low elevated repeat check checked
output intake urine fluid fluids bolus received blood loss estimated ebl weight water
food breakfast lunch dinner juice correction example sliding scale
""")

# Formulation words: skipped as a name, not combined into two-word names, and allowed
# between a name and its dose ("potassium chloride ER 20 mEq", "Glucagon nasal powder 3 mg").
_FORM_WORDS_RAW = """
er xr sr dr cr xl la ec odt hcl hct er24 inhaler injection solution suspension cream ointment gel
patch spray nasal ophthalmic topical extended release delayed chewable liquid syrup oral
powder nebulizer hfa inhalation tablet tablets tab tabs capsule capsules cap caps
"""
_FORM_WORDS = _word_set(_FORM_WORDS_RAW)

# Salt / counter-ion words: never looked up on their own (RxNav knows "sulfate", but
# MedlinePlus has no page for it), and when one ends a name the base drug is tried next
# ("metoprolol succinate" -> "metoprolol").
_SALT_WORDS = _word_set("""
sulfate sulphate succinate tartrate chloride hydrochloride citrate besylate maleate mesylate
fumarate gluconate acetate phosphate carbonate bicarbonate oxide hyclate sodium potassium
calcium magnesium
""")


def _norm(text: str) -> str:
    return " ".join(_tokens(text))


def _content_tokens(text: str) -> list[str]:
    """Tokens for BM25: stopwords and single letters dropped, digits kept ("type 2")."""
    return [t for t in _tokens(text) if t.isdigit() or (len(t) > 1 and t not in _BM25_STOPWORDS)]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _first_sentences(text: str, n: int = 2, max_chars: int = 260) -> str:
    flat = " ".join((text or "").replace("• ", " ").split())
    out = ""
    for sent in _SENT_SPLIT.split(flat)[:n]:
        cand = (out + " " + sent).strip()
        if out and len(cand) > max_chars:
            break
        out = cand
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return out


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s)|\n", cut) if m.end() >= limit * 0.45]
    if ends:
        return cut[: max(ends)].rstrip()
    return cut.rsplit(" ", 1)[0].rstrip(",;:") + "…"


def _strip_tracking(url: str) -> str:
    parts = urllib.parse.urlsplit(url or "")
    query = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith("utm_")]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


class _TextExtractor(HTMLParser):
    """Minimal HTML -> text: paragraphs on new lines, list items as "• " bullets."""

    _BLOCK = {"p", "div", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "table", "tr", "br", "section"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6", "script", "style"):
            self._skip += 1
        if tag == "li":
            self.parts.append("\n• ")
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6", "script", "style"):
            self._skip = max(0, self._skip - 1)
        if tag in self._BLOCK or tag == "li":
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _html_to_text(html: str) -> str:
    if "<" not in (html or ""):
        # Connect's drug summaries are plain text whose lists survive only as runs of
        # spaces ("is used to:   reduce fever  relieve pain"); keep them readable.
        text = re.sub(r":[ \t]{2,}", ": ", (html or "").strip())
        return " ".join(re.sub(r"[ \t]{2,}", "; ", text).split())
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup: fall back to tag stripping
        return " ".join(re.sub(r"<[^>]+>", " ", html).split())
    lines = [" ".join(line.split()) for line in "".join(parser.parts).split("\n")]
    return "\n".join(line for line in lines if line and line != "•")


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# =============================================================================
# Index loading
# =============================================================================

@lru_cache(maxsize=1)
def _load_index() -> dict:
    try:
        with open(INDEX_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        log.warning("MedlinePlus index unavailable (%s): references will be limited to drug lookups", exc)
        data = {}
    data.setdefault("metadata", {})
    data.setdefault("topics", [])
    data.setdefault("lab_tests", {})
    return data


@lru_cache(maxsize=1)
def _topic_by_title() -> dict[str, int]:
    return {t["title"]: i for i, t in enumerate(_load_index()["topics"])}


def _topic_ids(titles: Iterable[str]) -> set[int]:
    by_title = _topic_by_title()
    return {by_title[t] for t in titles if t in by_title}


@lru_cache(maxsize=1)
def _stop_topic_ids() -> frozenset[int]:
    return frozenset(_topic_ids(STOP_TOPICS))


def index_stats() -> dict:
    idx = _load_index()
    return {
        "topics": len(idx["topics"]),
        "lab_tests": len(idx["lab_tests"]),
        "source_date": idx["metadata"].get("source_xml_date", ""),
    }


# =============================================================================
# Phrase tables (exact / n-gram matching)
# =============================================================================

def _useful_parenthetical(inner: str) -> bool:
    """'(BPH)', '(PCI)', '(Swine Flu)' name the thing; '(Metabolic)', '(Temperature)' don't."""
    acronym = re.fullmatch(r"[A-Z][A-Za-z0-9-]{1,7}", inner) and sum(c.isupper() for c in inner) >= 2
    return bool(acronym) or len(inner.split()) >= 2


def _title_variants(title: str, unique_words: frozenset[str]) -> list[str]:
    """'HDL: The "Good" Cholesterol' -> HDL; 'Enlarged Prostate (BPH)' -> both parts;
    'Cancer--Living with Cancer' -> parts; 'Dizziness and Vertigo' -> each word, but
    only when neither word occurs in another title (so 'Medicines and Children'
    does not claim the word "children")."""
    out: list[str] = []
    if ":" in title:
        out.append(title.split(":", 1)[0])
    if "--" in title:
        out += title.split("--")
    m = re.match(r"^(.*?)\s*\((.+)\)\s*$", title)
    if m:
        out.append(m.group(1))
        if _useful_parenthetical(m.group(2)):
            out.append(m.group(2))
    parts = re.split(r"\s+and\s+", title)
    if len(parts) == 2 and all(len(p.split()) == 1 and _norm(p) in unique_words for p in parts):
        out += parts
    return [o.strip() for o in out if o.strip()]


def _synonym_variants(syn: str) -> list[str]:
    """Un-invert MeSH-style 'Pain, Abdominal' -> 'Abdominal Pain'; split '(PCI)'."""
    out = []
    m = re.match(r"^(.*?)\s*\((.+)\)\s*$", syn)
    if m:
        if _useful_parenthetical(m.group(2)):
            out.append(m.group(2))
        syn = m.group(1)
    parts = [p.strip() for p in syn.split(",")]
    if 2 <= len(parts) <= 3 and " and " not in syn.lower() and all(parts):
        out.append(" ".join(reversed(parts)))
    else:
        out.append(syn)
    return [o for o in out if o.strip()]


@lru_cache(maxsize=1)
def _phrase_candidates() -> dict[tuple[str, ...], list[tuple[int, int]]]:
    """phrase tokens -> [(level, topic index)]; level 0 title, 1 title variant, 2 synonym."""
    cands: dict[tuple[str, ...], list[tuple[int, Optional[int]]]] = defaultdict(list)
    stop_ids = _stop_topic_ids()
    stop_phrases = {_norm(p) for p in STOP_PHRASES}
    topics = _load_index()["topics"]
    title_word_counts = Counter(w for t in topics for w in set(_tokens(t["title"])))
    unique_words = frozenset(w for w, c in title_word_counts.items() if c == 1)

    def add(phrase: str, level: int, idx: Optional[int]) -> None:
        toks = tuple(_tokens(phrase))
        if not toks or " ".join(toks) in stop_phrases or all(t.isdigit() for t in toks):
            return
        cands[toks].append((level, idx))

    for i, topic in enumerate(topics):
        if i in stop_ids:
            continue
        add(topic["title"], 0, i)
        for v in _title_variants(topic["title"], unique_words):
            add(v, 1, i)
        for syn in topic.get("synonyms", []):
            for v in _synonym_variants(syn):
                add(v, 2, i)
    by_title = _topic_by_title()
    for title, phrases in EXTRA_SYNONYMS.items():
        if title in by_title and by_title[title] not in stop_ids:
            for p in phrases:
                add(p, _OVERRIDE_LEVEL, by_title[title])
    for p in BLOCK_PHRASES:
        add(p, _BLOCK_LEVEL, None)
    return cands


def _resolve(cands: list[tuple[int, Optional[int]]]) -> Optional[int]:
    """Pick the topic for a phrase: best level wins; ties between topics -> ambiguous."""
    best = min(level for level, _ in cands)
    if best == _BLOCK_LEVEL:
        return _BLOCKED
    topics = {idx for level, idx in cands if level == best}
    return topics.pop() if len(topics) == 1 else None


@lru_cache(maxsize=1)
def _phrase_tables() -> tuple[dict[tuple[str, ...], int], dict[tuple[str, ...], int], int]:
    """Returns (document phrase table, concept phrase table, max phrase length).

    The document table is strict (no short tokens outside the allowlist) because it
    scans free text; the concept table also accepts short acronyms, since an LLM
    concept like "MS" or "CAD" is the whole term.
    """
    doc: dict[tuple[str, ...], int] = {}
    concept: dict[tuple[str, ...], int] = {}
    for toks, cands in _phrase_candidates().items():
        idx = _resolve(cands)
        if idx is None:
            continue
        if idx != _BLOCKED:
            concept[toks] = idx
        if len(toks) == 1 and len(toks[0]) < 4 and toks[0] not in SHORT_ALLOWLIST:
            continue
        doc[toks] = idx
    by_title, stop_ids = _topic_by_title(), _stop_topic_ids()
    for title, phrases in CONCEPT_SYNONYMS.items():
        idx = by_title.get(title)
        if idx is not None and idx not in stop_ids:
            for p in phrases:
                concept[tuple(_tokens(p))] = idx
    max_len = max((len(k) for k in doc), default=1)
    return doc, concept, max_len


@lru_cache(maxsize=1)
def _blocked_concepts() -> frozenset[tuple[str, ...]]:
    """BLOCK_PHRASES as token tuples, so LLM concepts like "blood gas" are rejected too."""
    return frozenset(tuple(_tokens(p)) for p in BLOCK_PHRASES)


# Cues that the patient does not have the finding: negation ("no pneumonia", "denies
# chest pain", "atrial fibrillation ruled out") or someone else has it ("family
# history of breast cancer").
_PRE_NEGATION_RE = re.compile(
    r"\b(no|not|denies|denied|deny|denying|negative for|without|never|free of|ruled out|absence of"
    r"|family history|fhx|fh)\b", re.I)

# Words that end a negation's scope (NegEx "termination terms"): "denies chest pain BUT
# has pneumonia", "denies smoking, NOW with COPD", "no fever but CT SHOWS appendicitis".
# ("reported" is left out on purpose: "no reported allergies" is still a negation.)
_NEG_TERMINATION_RE = re.compile(
    r"\b(?:but|however|although|though|except|aside from|apart from|now|presents?|presented|presenting"
    r"|complains?|complained|reports?|reveals?|revealed|shows?|showed|demonstrates?)\b", re.I)

# Negation that follows the term: "COVID-19 PCR: not detected", "HIV test was negative",
# "Influenza, COVID-19, and RSV tests negative".
# The rest of a coordinated list after the term (", COVID-19, and RSV"). It needs and/or,
# so "Pneumonia, blood cultures negative" does NOT negate pneumonia.
_NEG_LIST_ITEM = r"[\w/-]+(?:\s+[\w/-]+)?"
_NEG_LIST_TAIL = rf"(?:\s*,\s*{_NEG_LIST_ITEM})*\s*,?\s+(?:and|or)\s+{_NEG_LIST_ITEM}"
# Test words allowed between the term and the result ("PCR", "surface antigen", "A/B").
_NEG_TEST_WORD = (r"(?:\s+(?:rapid|tests?|testing|pcr|naat|antigens?|antibod(?:y|ies)|ag|ab|igm|igg"
                  r"|screen(?:ing)?|swabs?|panel|assay|serology|surface|core|a/b|a|b))")
_NEG_TEST_WORDS = rf"{_NEG_TEST_WORD}*"
# "Diabetes: negative for retinopathy" is about the retinopathy, hence the (?!...for).
_NEG_RESULT = (r"(?:negative|neg|non-?reactive|undetected|ruled out|excluded|absent"
               r"|not (?:seen|present|found|identified|detected|demonstrated))\b(?!\s+for\b)")
# After a list tail at least one test word must follow: in "Influenza and RSV tests negative"
# the result belongs to the whole list, but in "pneumonia and blood cultures were negative"
# it is only about the cultures, so the pneumonia stays.
_POST_NEGATION_RE = re.compile(
    rf"^{_NEG_TEST_WORDS}(?:{_NEG_LIST_TAIL}{_NEG_TEST_WORD}+)?\s*"   # "Influenza A and B antigens"
    rf"(?::\s*|\s+(?:is|was|were|are|has been|have been|came back|returned)\s+)?{_NEG_RESULT}", re.I)

# A line break followed by a lowercase word can be a hard wrap in the middle of a sentence
# ("No chest pain,\npalpitations, headache"), not a clause break. It counts as one only
# when the line above looks unfinished (see _is_soft_wrap).
_SOFT_WRAP_RE = re.compile(r"(?<=[,\w-])[ \t]*\r?\n(?=[ \t]*[a-z(])")
# A line ending like this is not a complete thought ("No chest pain,", "Family history of",
# "Patient denies").
_UNFINISHED_LINE_RE = re.compile(
    r"(?:[,(/\-]|\b(?:and|or|nor|with|without|to|for|of|the|a|an|as|by|in|on|at|from|than|not|no"
    r"|denies|denied|deny|never|negative for))[ \t]*$", re.I)
_WRAPPED_LINE_CHARS = 50   # prose hard-wrapped at ~70-80 columns fills most of the line
_CLAUSE_SPLIT_RE = re.compile(r"[.;:!?\n]")
# "without X, Y, or Z" lists. Only or/nor: with "and", "diabetes without complications,
# hypertension and hyperlipidemia" (a problem list) would lose two real diagnoses.
# The rest of the list after a term (", or diarrhea"), and a conjunction before it.
_LIST_REST_RE = re.compile(rf"^(?:\s*,\s*{_NEG_LIST_ITEM})*\s*,?\s+(?:or|nor)\b", re.I)
_CONJUNCTION_RE = re.compile(r"\b(?:or|nor)\b", re.I)


def _names_finding(line: str) -> bool:
    """Does the line contain a topic phrase ("fracture", "pleural effusion")?"""
    table, _, max_len = _phrase_tables()
    words = _tokens(line)
    return any(table.get(tuple(words[i:i + n]), _BLOCKED) != _BLOCKED
               for i in range(len(words)) for n in range(1, min(max_len, len(words) - i) + 1))


def _is_soft_wrap(text: str, newline_pos: int) -> bool:
    """Does the line ending at newline_pos continue on the next line? Yes when it ends
    with a comma or a word like "of" / "denies", when it is a long line of wrapped prose,
    or when it names no finding yet ("No evidence of acute" + "fracture"). A short line
    that is complete ("no acute fracture" above "pneumonia in the right lower lobe") is a
    list item of its own, so its negation must not carry over."""
    line = text[text.rfind("\n", 0, newline_pos) + 1:newline_pos].rstrip()
    return (len(line) >= _WRAPPED_LINE_CHARS or bool(_UNFINISHED_LINE_RE.search(line))
            or not _names_finding(line))


def _is_negated(text: str, start: int, end: int) -> bool:
    """NegEx-lite: a cue among the few words before the match (or right after it), in the
    same clause and with no termination term ("but", "now", "shows") in between."""
    lo = max(0, start - 60)
    # Replace soft wraps with spaces of the same length, so offsets still line up.
    window = _SOFT_WRAP_RE.sub(
        lambda m: " " * len(m.group()) if _is_soft_wrap(text, lo + m.start()) else m.group(),
        text[lo:end + 80])
    # ':' is not a clause break after the term ("COVID-19: not detected"); the regex only
    # allows it right before the result word.
    after = re.split(r"[.;!?\n]", window[end - lo:])[0]
    before = _CLAUSE_SPLIT_RE.split(window[:start - lo])[-1]
    terms = list(_NEG_TERMINATION_RE.finditer(before))
    if terms:
        before = before[terms[-1].end():]
    words = " ".join(before.split()[-5:])
    for cue in _PRE_NEGATION_RE.finditer(words):
        # ICD-style "diabetes without complications, essential hypertension": "without"
        # negates only its own comma segment ("no X, Y" lists still carry over commas)...
        between = words[cue.end():]
        if cue.group(1).lower() == "without" and "," in between:
            # ...unless the term is part of an "or" list: "without nausea, vomiting, or
            # diarrhea" negates all three (or/nor before the term or after it).
            if not (_CONJUNCTION_RE.search(between) or _LIST_REST_RE.match(after)):
                continue
        return True
    return bool(_POST_NEGATION_RE.match(after))


@dataclass
class _TopicHit:
    topic: int
    matched: str                 # surface text of the first occurrence
    first_pos: int
    count: int = 1
    forms: list[str] = field(default_factory=list)   # every surface form seen
    in_dx: bool = False          # mentioned under a DIAGNOSIS / IMPRESSION / ASSESSMENT heading
    core: int = 0                # mentions outside instruction / follow-up sections


def _in_spans(pos: int, spans: Iterable[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in spans)


def _scan_topics(text: str, skip_spans: Iterable[tuple[int, int]] = (),
                 dx_spans: Iterable[tuple[int, int]] = (),
                 instr_spans: Iterable[tuple[int, int]] = ()) -> list[_TopicHit]:
    """Leftmost-longest phrase matching over normalised tokens.

    Returns non-negated hits. Hits starting inside skip_spans (medication mentions,
    which get their own drug pages) are ignored, so "atorvastatin calcium 40 mg" does
    not yield the Calcium topic.

    Order: topics named in a diagnosis section (dx_spans) first, then by how often
    they are mentioned outside instruction sections (instr_spans), then overall. A
    plainly stated diagnosis mentioned once must beat "insulin" repeated six times in
    the hospital course. Without spans this is plain most-frequent-first.
    """
    table, _, max_len = _phrase_tables()
    spans = sorted(skip_spans)
    dx, instr = list(dx_spans), list(instr_spans)
    toks = _tokens_with_spans(text)
    words = [t for t, _, _ in toks]
    hits: dict[int, _TopicHit] = {}
    i = 0
    while i < len(words):
        for n in range(min(max_len, len(words) - i), 0, -1):
            idx = table.get(tuple(words[i:i + n]))
            if idx is None:
                continue
            start, end = toks[i][1], toks[i + n - 1][2]
            if idx != _BLOCKED and not _in_spans(start, spans) and not _is_negated(text, start, end):
                surface = " ".join(text[start:end].split())   # "chronic\nkidney disease" -> one line
                if idx in hits:
                    hits[idx].count += 1
                else:
                    hits[idx] = _TopicHit(idx, surface, start)
                hit = hits[idx]
                hit.forms.append(surface)
                hit.in_dx = hit.in_dx or _in_spans(start, dx)
                hit.core += not _in_spans(start, instr)
            i += n
            break
        else:
            i += 1
    return sorted(hits.values(), key=lambda h: (not h.in_dx, -h.core, -h.count, h.first_pos))


# =============================================================================
# BM25 over topics
# =============================================================================

class _BM25:
    K1 = 1.2
    B = 0.75
    FIELD_WEIGHTS = (("title", 3.0), ("synonyms", 2.0), ("description", 1.0), ("text", 1.0))

    def __init__(self, topics: list[dict]) -> None:
        self.n_docs = len(topics)
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.strong: list[frozenset[str]] = []   # title + synonym tokens per topic
        self.title_terms: list[frozenset[str]] = []
        lengths = []
        for i, topic in enumerate(topics):
            tf: Counter = Counter()
            strong: set[str] = set()
            for name, weight in self.FIELD_WEIGHTS:
                value = topic.get(name) or ""
                if isinstance(value, list):
                    value = " ; ".join(value)
                toks = _content_tokens(value)
                if name in ("title", "synonyms"):
                    strong.update(toks)
                for tok in toks:
                    tf[tok] += weight
            self.strong.append(frozenset(strong))
            self.title_terms.append(frozenset(_content_tokens(topic.get("title", ""))))
            lengths.append(sum(tf.values()))
            for tok, f in tf.items():
                self.postings[tok].append((i, f))
        self.lengths = lengths
        self.avg_len = (sum(lengths) / len(lengths)) if lengths else 1.0
        self.idf = {
            tok: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for tok, p in self.postings.items()
        }

    def search(self, query_tokens: list[str], limit: int, min_coverage: float = 0.0,
               exclude: frozenset[int] = frozenset()) -> list[tuple[float, float, int]]:
        """Returns [(score, normalised score 0..1, topic index)] best first.

        min_coverage is the fraction of distinct query tokens a topic must contain.
        The normalised score divides by the best score any document could get for
        this query, which makes one threshold work for short and long queries.
        A topic whose title is exactly the query gets a bonus, so "high blood
        pressure" ranks High Blood Pressure above How to Prevent High Blood Pressure.
        """
        terms = list(dict.fromkeys(query_tokens))
        if not terms:
            return []
        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, int] = defaultdict(int)
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc, f in self.postings[term]:
                norm = f * (self.K1 + 1) / (f + self.K1 * (1 - self.B + self.B * self.lengths[doc] / self.avg_len))
                scores[doc] += idf * norm
                matched[doc] += 1
        max_possible = sum(self.idf.get(t, 0.0) for t in terms) * (self.K1 + 1) or 1.0
        query_set = frozenset(terms)
        for doc in scores:
            if self.title_terms[doc] == query_set:
                scores[doc] += 0.5 * max_possible
        need = math.ceil(min_coverage * len(terms) - 1e-9)
        ranked = [
            (s, min(1.0, s / max_possible), d) for d, s in scores.items()
            if matched[d] >= need and d not in exclude
        ]
        ranked.sort(key=lambda r: (-r[0], r[2]))
        return ranked[:limit]


@lru_cache(maxsize=1)
def _bm25() -> _BM25:
    return _BM25(_load_index()["topics"])


def search_topics(query: str, limit: int = 3) -> list[tuple[float, dict]]:
    """Plain BM25 search over health topics: [(score, topic dict)], best first."""
    topics = _load_index()["topics"]
    return [(round(score, 4), topics[i]) for score, _, i in _bm25().search(_content_tokens(query), limit)]


# BM25 acceptance for learn_more. A topic is accepted only if
#   * it contains every query token (>= 75% for queries of 4+ tokens),
#   * its score is >= BM25_MIN_NORM_SCORE of the best score possible for the query,
#   * at least half of the query tokens (and one specific, non-generic token) occur
#     in its title or synonyms - body-text-only matches were the main source of junk
#     ("dietary counseling" -> Chiropractic, "discharge" -> Breast Diseases),
#   * a one-word query is rare enough (idf >= BM25_MIN_SINGLE_IDF) to be specific.
# Calibrated on ~60 concept-like terms: accepted e.g. "shortness of breath" ->
# Breathing Problems, "kidney function" -> Kidney Tests, "blood clot in the lung" ->
# Pulmonary Embolism, "physical therapy" -> Rehabilitation; rejected "follow up",
# "routine", "normal", "therapy", "panel", "patient education", "1 month", nonsense.
BM25_MIN_NORM_SCORE = 0.5
BM25_MIN_STRONG_COVERAGE = 0.5
BM25_MIN_SINGLE_IDF = 3.5

# Modifiers dropped from concept terms before matching ("mild hyperkalemia").
_CONCEPT_MODIFIERS = _word_set("""
mild moderate severe elevated increased decreased borderline abnormal suspected possible probable
likely history hx known new worsening stable uncontrolled controlled repeat recurrent persistent
slightly significant significantly
""")

# Words too generic to justify a match on their own.
_GENERIC_WORDS = _word_set("""
panel test testing level result value therapy treatment management medication medicine drug care
check follow routine normal abnormal history risk change lifestyle counseling education plan visit
appointment month week day year exam examination evaluation assessment status problem condition
health healthy procedure finding range report diet high low disease disorder
""")


# Surgery pages are only right when the concept itself names a procedure: "knee pain"
# must not land on Knee Replacement, nor "weight loss" on Weight Loss Surgery.
_PROCEDURE_WORDS = _word_set("replacement surgery transplant transplantation arthroplasty amputation")


def _concept_tokens(term: str) -> list[str]:
    return [t for t in dict.fromkeys(_content_tokens(term)) if t not in _CONCEPT_MODIFIERS]


def _bm25_concept(term: str) -> Optional[int]:
    """Best topic for a free-text concept, or None if nothing clears the bar."""
    toks = _concept_tokens(term)
    specific = [t for t in toks if t not in _GENERIC_WORDS]
    if not specific:
        return None
    bm = _bm25()
    if len(toks) == 1 and bm.idf.get(toks[0], 0.0) < BM25_MIN_SINGLE_IDF:
        return None
    coverage = 1.0 if len(toks) <= 3 else 0.75
    procedural_query = any(t in _PROCEDURE_WORDS for t in toks)
    for _score, norm, doc in bm.search(toks, 3, min_coverage=coverage, exclude=_stop_topic_ids()):
        if norm < BM25_MIN_NORM_SCORE:
            break
        if not procedural_query and bm.title_terms[doc] & _PROCEDURE_WORDS:
            continue
        strong = bm.strong[doc]
        if (sum(t in strong for t in toks) >= BM25_MIN_STRONG_COVERAGE * len(toks)
                and any(t in strong for t in specific)):
            return doc
    return None


# =============================================================================
# Lab tests
# =============================================================================

# MedlinePlus "Other names" that are too broad to identify one lab test.
_LAB_SYNONYM_DENY = frozenset(_norm(s) for s in (
    "urine", "serum electrolytes", "CBC with differential", "PFT", "kidney function", "renal function",
    "urine creatinine", "ALK", "PHOS", "Crit",
))


@lru_cache(maxsize=1)
def _lab_alias_map() -> dict[str, str]:
    """normalised lab name / synonym -> canonical key (curated names win over synonyms)."""
    out: dict[str, str] = {}
    tests = _load_index()["lab_tests"]
    for key, lab in tests.items():
        for name in [key.replace("_", " "), lab.get("title", "")] + list(lab.get("names", [])):
            out.setdefault(_norm(name), key)
    for key, lab in tests.items():
        for name in lab.get("synonyms", []):
            n = _norm(re.sub(r"\s*\([^)]*\)", "", name))
            if n and n not in _LAB_SYNONYM_DENY and len(n) >= 3:
                out.setdefault(n, key)
    out.pop("", None)
    return out


# A 0 or 1 between two letters is an OCR misread of o / i / l ("Gluc0se", "Potass1um").
# Digits next to a non-letter are kept, so "A1c", "T4" and "B12" survive.
_OCR_DIGIT_IN_WORD = re.compile(r"(?<=[a-z])[01](?=[a-z])")


def _ocr_name_candidates(name: str) -> list[str]:
    """Spellings to retry for an OCR-garbled lab name: 'Gluc0se', 'Potass1um', 'Sod ium', 'C02'."""
    words = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", _fold(name)).split()
    if not words or len(words) > 4:
        return []
    options = []
    for w in words:
        w = re.sub(r"^c0(?=2$)", "co", w)                                  # C02 -> co2
        options.append(sorted({
            w,                                                             # as printed ('a1c')
            _OCR_DIGIT_IN_WORD.sub(lambda x: "o" if x.group() == "0" else "i", w),
            _OCR_DIGIT_IN_WORD.sub(lambda x: "o" if x.group() == "0" else "l", w),
        }))
    out: list[str] = []
    for combo in itertools.product(*options):        # per word, so 'Hemog1obin A1c' works
        norm = _norm(" ".join(combo))
        out += [norm, norm.replace(" ", "")]           # 'sod ium' -> 'sodium'
    return list(dict.fromkeys(c for c in out if c))


def _lab_key(lab: Any) -> Optional[str]:
    tests = _load_index()["lab_tests"]
    key = _attr(lab, "canonical")
    if key in tests:
        return key
    alias = _lab_alias_map()
    name = str(_attr(lab, "name", "") or "")
    hit = alias.get(_norm(name))
    if hit:
        return hit
    # OCR fallback. Still an exact whole-name match, so no fuzzy false positives.
    return next((alias[c] for c in _ocr_name_candidates(name) if c in alias), None)


def _severity(lab: Any) -> float:
    """How far outside its range a lab is, in range-widths (0 when unknown)."""
    try:
        value = float(_attr(lab, "value"))
        low, high = _attr(lab, "ref_low"), _attr(lab, "ref_high")
        flag = _attr(lab, "flag")
        width = (float(high) - float(low)) if low is not None and high is not None and float(high) > float(low) else None
        if flag == "high" and high is not None:
            return (value - float(high)) / (width or abs(float(high)) or 1.0)
        if flag == "low" and low is not None:
            return (float(low) - value) / (width or abs(float(low)) or 1.0)
    except (TypeError, ValueError):
        pass
    return 0.0


def _lab_reference(key: str, matched: str) -> Reference:
    lab = _load_index()["lab_tests"][key]
    return Reference(
        title=lab["title"], url=lab["url"],
        snippet=lab.get("description") or _first_sentences(lab.get("text", "")),
        text=lab.get("text", ""), kind="lab_test", matched=matched,
    )


def _topic_reference(idx: int, matched: str) -> Reference:
    topic = _load_index()["topics"][idx]
    return Reference(
        title=topic["title"], url=topic["url"],
        snippet=topic.get("description") or _first_sentences(topic.get("text", "")),
        text=topic.get("text", ""), kind="topic", matched=matched,
    )


# =============================================================================
# Drugs (live RxNav + MedlinePlus Connect)
# =============================================================================

def _ssl_context() -> ssl.SSLContext:
    # python.org macOS builds have no system CA bundle; certifi is already installed
    # (dependency of anthropic/httpx). Fall back to the default store otherwise.
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_SSL_CTX = _ssl_context()


def _get_json(url: str, timeout: float) -> Optional[dict]:
    """GET a JSON document. Never raises: returns None on any network/parse error."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Clearchart/1.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log.info("MedlinePlus/RxNav request failed (%s): %s", url, exc)
        return None


_DRUG_CACHE: dict[tuple[str, frozenset[str]], Optional[Reference]] = {}
_DRUG_CACHE_LOCK = threading.Lock()
_DRUG_CACHE_MAX = 2048


def _connect_drug_url(rxcui: str) -> str:
    return CONNECT_URL + "?" + urllib.parse.urlencode({
        "mainSearchCriteria.v.cs": RXNORM_OID,
        "mainSearchCriteria.v.c": rxcui,
        "knowledgeResponseType": "application/json",
        "informationRecipient.languageCode.c": "en",
    })


_ROUTE_WORDS = frozenset({"injection", "rectal", "inhalation", "nasal", "ophthalmic", "otic", "topical",
                          "transdermal", "intravenous", "vaginal", "implant"})


# Words near a medication that say which route it is taken by. MedlinePlus has one page
# per route for some drugs (Ipratropium Nasal Spray / Ipratropium Oral Inhalation), so
# "albuterol/ipratropium nebulizer" should pick the inhalation page.
_ROUTE_CUES = {
    "inhalation": _word_set("inhalation inhaler inhalers inhaled inhale nebulizer nebulizers nebulized "
                            "neb nebs hfa mdi dpi puff puffs diskus respimat handihaler ellipta"),
    "nasal": _word_set("nasal intranasal nostril nostrils"),
    "transdermal": _word_set("transdermal patch patches"),
    "injection": _word_set("injection injections inject injected subcutaneous subq sq intramuscular "
                           "pen autoinjector syringe"),
    "topical": _word_set("topical cream ointment lotion"),
    "ophthalmic": _word_set("ophthalmic eye eyes"),
    "rectal": _word_set("rectal suppository enema"),
    "vaginal": _word_set("vaginal"),
}
# Tie-break only, when nothing in the document names a route: lower wins. Keeps insulin
# and epinephrine on the injection page, and ipratropium on the inhaler, not the nose spray.
_ROUTE_PRIOR = {"injection": 0, "inhalation": 1, "transdermal": 2, "nasal": 3}


def _route_hint(context: str) -> frozenset[str]:
    """Routes named in the text around a medication ("2 puffs" -> {"inhalation"})."""
    toks = set(_tokens(context))
    return frozenset(route for route, cues in _ROUTE_CUES.items() if toks & cues)


def _pick_drug_entry(entries: list, name: str, route_hint: frozenset[str] = frozenset()) -> Optional[dict]:
    """Prefer /druginfo/meds/ pages (then /druginfo/natural/) whose title best matches
    the name. A route the name didn't ask for ("Potassium Chloride Injection" for
    "potassium chloride") counts against a page, a route named in the document
    (route_hint) counts for it.

    An rxcui whose /druginfo/meds/ pages name different drugs, none of them the name, is
    ambiguous for a salt name (docusate sodium -> [Senna, Stool Softeners]): None, so the
    caller tries the base drug next ("Docusate"). A brand name with such pages is a
    combination product whose pages are its ingredients (Percocet -> [Acetaminophen,
    Oxycodone]; Hyzaar, Janumet): the best ingredient page is kept, since no other name is
    left to try. Route variants of one drug (Lasix -> [Furosemide, Furosemide Injection])
    and a single page (ferrous sulfate -> Iron Supplements) are always accepted."""
    wanted = set(_tokens(name))
    cue_toks = set().union(*(_ROUTE_CUES[r] for r in route_hint)) if route_hint else set()
    best, best_key = None, None
    meds_toks: list[set[str]] = []
    for pos, entry in enumerate(entries or []):
        try:
            href = (entry.get("link") or [{}])[0].get("href", "")
            title = (entry.get("title") or {}).get("_value", "")
        except (AttributeError, IndexError, TypeError):
            continue
        if "/druginfo/meds/" in href:
            tier = 0
        elif "/druginfo/natural/" in href:
            tier = 1
        else:
            continue
        title_toks = set(_tokens(title))
        if tier == 0:
            meds_toks.append(title_toks)
        hinted = title_toks & route_hint
        extra = title_toks - wanted - hinted - cue_toks
        score = (len(wanted & title_toks) + 1.0 * len(hinted)
                 - 0.3 * len(extra) - 1.0 * len(extra & _ROUTE_WORDS))
        prior = min((_ROUTE_PRIOR.get(r, 9) for r in title_toks & _ROUTE_WORDS), default=0)
        key = (tier, -score, prior, pos)
        if best_key is None or key < best_key:
            best, best_key = entry, key
    if len(meds_toks) > 1 and wanted & _SALT_WORDS and not any(wanted & t for t in meds_toks):
        if not set.intersection(*(t - _ROUTE_WORDS for t in meds_toks)):
            return None  # a salt name listing different drugs: let the base name decide
    return best


def _lookup_drug_uncached(name: str, timeout: float,
                          route_hint: frozenset[str] = frozenset()) -> tuple[Optional[Reference], bool]:
    """Returns (reference or None, definitive). Not definitive = network trouble."""
    data = _get_json(RXNAV_RXCUI_URL + "?" + urllib.parse.urlencode({"name": name, "search": 2}), timeout)
    if not isinstance(data, dict):
        return None, False
    ids = ((data.get("idGroup") or {}).get("rxnormId") or [])[:2]
    if not ids:
        return None, True
    for rxcui in ids:
        feed = _get_json(_connect_drug_url(str(rxcui)), timeout)
        if not isinstance(feed, dict):
            return None, False
        entry = _pick_drug_entry((feed.get("feed") or {}).get("entry") or [], name, route_hint)
        if entry is None:
            continue
        title = (entry.get("title") or {}).get("_value", "").strip() or name
        url = _strip_tracking(entry["link"][0]["href"])
        text = _truncate(_html_to_text((entry.get("summary") or {}).get("_value", "")), DRUG_TEXT_CHARS)
        return Reference(title=title, url=url, snippet=_first_sentences(text), text=text,
                         kind="drug", matched=name), True
    return None, True


def lookup_drug(name: str, timeout: float = DRUG_TIMEOUT_S,
                route_hint: frozenset[str] = frozenset()) -> Optional[Reference]:
    """MedlinePlus drug page for a medication name, or None. Cached; never raises.

    route_hint (see _route_hint) picks between per-route pages of the same drug."""
    try:
        clean = " ".join(str(name or "").split()).strip(" .,;:")
        norm = _norm(clean)
        if len(norm) < 3:
            return None
        key = (norm, frozenset(route_hint))   # the hint can change which page is picked
        with _DRUG_CACHE_LOCK:
            if key in _DRUG_CACHE:
                hit = _DRUG_CACHE[key]
                return replace(hit, matched=clean) if hit else None
        ref, definitive = _lookup_drug_uncached(clean, timeout, frozenset(route_hint))
        if definitive:
            with _DRUG_CACHE_LOCK:
                if len(_DRUG_CACHE) >= _DRUG_CACHE_MAX:
                    _DRUG_CACHE.clear()
                _DRUG_CACHE[key] = ref
        return ref
    except Exception:  # defensive: retrieval must never break a translation
        log.exception("lookup_drug failed")  # no name: it comes from the user's document
        return None


# ---- medication candidates ---------------------------------------------------

_W = r"[A-Za-z][A-Za-z0-9\-]+"
# The dose regex scans the whole document, so every part of it is bounded: a word of
# at most 40 characters, a number of at most 7 digits, and a gap before the number that
# can be matched only one way. With an unbounded word followed by an optional gap and
# \d+, a long run like "a111...1" makes the engine try every split point (quadratic),
# and Python's re holds the GIL while it does, freezing the whole server for ~40 s.
_DOSE_W = r"[A-Za-z][A-Za-z0-9\-]{1,39}"
# Spaces, or one line break before a lowercase word: a name wrapped in the middle
# ("Start ferrous\nsulfate 325 mg"). A capitalised next line is a new list item
# ("...as needed for pain\nOxycodone 5 mg" is not "pain Oxycodone"). (?-i:...) keeps
# [a-z] lowercase-only although _DOSE_RE is case-insensitive.
_NAME_SEP = r"(?:[ \t]+|[ \t]*\r?\n[ \t]*(?=(?-i:[a-z])))"
_FORM_ALT = "|".join(sorted(_FORM_WORDS_RAW.split(), key=len, reverse=True))
_DOSE_UNIT = r"(?:mg|mcg|µg|ug|g|gm|grams?|units?|iu|ml|meq|tabs?|tablets?|caps?|capsules?|puffs?)"
_DOSE_RE = re.compile(
    rf"(?:\b(?P<w1>{_DOSE_W}){_NAME_SEP})?\b(?P<w2>{_DOSE_W})"
    rf"(?:[ \t]+(?:{_FORM_ALT})\b){{0,2}}"          # "ER", "nasal powder" before the dose
    r"[ \t]*(?:[:,][ \t]*)?"
    r"(?P<num>\d{1,7}(?:\.\d{1,3})?)[ \t]?"
    rf"(?P<unit>{_DOSE_UNIT})\b"
    r"(?![ \t]*/)",
    re.I,
)
_HAS_DOSE_RE = re.compile(rf"\d[ \t]?{_DOSE_UNIT}\b", re.I)   # "500 mg" somewhere on a line
_MED_HEADER_RE = re.compile(
    r"^\s*(?P<head>(?:[A-Za-z()&/-]+[ \t]+){0,3}?"
    r"(?:medications?|meds|medicines|prescriptions?|insulin[ \t]+regimens?)\b"
    r"[A-Za-z()&/ \t-]{0,30}?)[ \t]*(?::[ \t]*(?P<rest>.*))?$",
    re.I,
)
# Medicines given during the stay ("MEDICATIONS GIVEN IN ED:") rank after the discharge list.
_INPATIENT_LIST_RE = re.compile(
    r"\b(?:inpatient|hospital|administered|given|received|pre-?op\w*|intra-?op\w*|ed|er)\b", re.I)
_TAKE_HOME_LIST_RE = re.compile(r"\b(?:discharge|home|current|outpatient)\b", re.I)
_BULLET_RE = re.compile(r"^[\s\-\*•·–>]*(?:\(?\d{1,2}[.)]\s*)?")
_LEAD_WORDS_RE = re.compile(rf"^({_W})(?:[ \t]+({_W}))?")
_ITEM_MARKER_RE = re.compile(r"^[ \t]*(?:[-*•·–>]+|\(?\d{1,2}[.)](?!\d))[ \t]*")
# A line ending like this continues on the next one ("take with\nbreakfast").
_DANGLING_RE = re.compile(
    r"(?:[,(/\-]|\b(?:and|or|with|to|for|of|the|a|an|as|by|in|on|at|every|per|if|then|before|after"
    r"|until|while|from|plus|than|not))[ \t]*$", re.I)
# First words that make "<...> medications" a sentence, not a heading.
_HEADER_VERBS = frozenset({
    "continue", "continued", "resume", "resumed", "take", "taking", "takes", "tolerating", "tolerated",
    "tolerates", "start", "started", "stop", "stopped", "hold", "held", "review", "reviewed", "reconcile",
    "reconciled", "refill", "refilled", "no", "all", "any", "his", "her", "their", "on", "off", "denies",
})
_SMALL_WORDS = frozenset({"of", "on", "at", "to", "and", "for", "the"})


@lru_cache(maxsize=1)
def _lab_words() -> frozenset[str]:
    words = set(_lab_alias_map())
    words.update({"hemoglobin", "hematocrit", "platelet", "glucose", "cholesterol", "ldl", "hdl"})
    return frozenset(words)


def _single_ok(word: str) -> bool:
    w = _norm(word)
    return (len(w) >= 4 and w not in _DRUG_STOPWORDS and w not in _FORM_WORDS
            and w not in _lab_words() and not w.isdigit())


def _pair_ok(w1: str, w2: str) -> bool:
    a, b = _norm(w1), _norm(w2)
    return (len(a) >= 3 and a not in _DRUG_STOPWORDS and a not in _FORM_WORDS
            and b not in _DRUG_STOPWORDS and b not in _FORM_WORDS)


def _variants(w1: Optional[str], w2: Optional[str]) -> list[str]:
    """Names to try for 'w1 w2 <dose>': the two-word name, then each word alone. A salt
    word is never tried alone, and after a salt the base drug comes first
    ("Docusate sodium" -> ["Docusate sodium", "Docusate"])."""
    out = []
    if w1 and w2 and _pair_ok(w1, w2):
        out.append(f"{w1} {w2}")
    singles = (w1, w2) if w2 and _norm(w2) in _SALT_WORDS else (w2, w1)
    for w in singles:
        if w and _single_ok(w) and _norm(w) not in _SALT_WORDS:
            out.append(w)
    return out


def _dose_name_words(m: re.Match) -> list[tuple[str, int]]:
    """The name words (word, offset) of a dose match, trailing formulation words dropped
    ("Aspirin EC 81 mg" -> Aspirin). Empty when the word right before the dose is a filler
    word: "contents of 1 capsule", "(INCREASED from 20 units)", "lunch with 60 g"."""
    words = [(m.group(g), m.start(g)) for g in ("w1", "w2") if m.group(g)]
    if len(words) == 2 and "\n" in m.string[words[0][1]:words[1][1]]:
        # w1 ends the line above. A line with its own dose is a separate medicine in an
        # all-lowercase list ("...for pain\noxycodone 5 mg"), not the first half of a name.
        line_start = m.string.rfind("\n", 0, words[0][1]) + 1
        if _HAS_DOSE_RE.search(m.string, line_start, words[0][1]):
            words = words[1:]
    while words and _norm(words[-1][0]) in _FORM_WORDS:
        words.pop()
    if not words or _norm(words[-1][0]) in _DRUG_STOPWORDS:
        return []
    return words


_CAPS_HEADING_RE = re.compile(r"^\s*[A-Z][A-Z /&()-]{2,40}:")   # "INSTRUCTIONS: Take ..."
_SECTION_RE = re.compile(
    r"^\s*(?:allerg\w*|instructions?|plan|follow[- ]?up|diet|activity|diagnos\w*|assessment|impression"
    r"|labs?|laboratory|vitals?|vital signs|history|hpi|exam\w*|procedures?|disposition|signed"
    r"|return precautions|patient instructions|discharge instructions)\b[^:\n]{0,25}:",
    re.I,
)
# A mixed-case "Heading: value" line that starts a new section: "Discharge Diagnosis: ...",
# "Past Medical History: ...", "Hospital Course: ...". The heading must contain a section
# word, so "Penicillin: hives" (allergy list) or "Lisinopril: take daily" (med list) does not.
_SECTION_WORD_RE = re.compile(
    r"^\s*(?:[A-Za-z][A-Za-z /&()'-]{0,40}?\s)?"
    r"(?:allerg\w*|diagnos\w*|complaint|course|history|hx|pmh|psh|hpi|exam\w*|reason|results?|imaging"
    r"|findings|impression|assessment|plan|problems?|medications?|meds|admi\w*|discharge|procedures?"
    r"|instructions?|follow[- ]?up|vitals?|labs?|secondary|primary|condition|disposition|precautions"
    r"|consults?|signed|diet|activity|systems)\b[^:\n]{0,20}:[ \t]*(?P<value>.*)$",
    re.I,
)


def _is_section_header(line: str) -> bool:
    """End of a medication list: blank line, a new "Heading:" or an ALL-CAPS heading."""
    s = line.strip()
    if not s:
        return True
    if (s.endswith(":") and len(s) <= 45) or _CAPS_HEADING_RE.match(s) or _SECTION_RE.match(s):
        return True
    letters = re.sub(r"[^A-Za-z]", "", s)
    return len(s) <= 45 and len(letters) >= 4 and s.upper() == s and not re.search(r"\d", s)


def _starts_new_section(line: str) -> bool:
    """A mixed-case 'Heading: value' line with a section word ('Hospital Course: ...'),
    but not a list item ('1. ...') or 'Aspirin (history of MI): 81 mg'."""
    if _ITEM_MARKER_RE.match(line):
        return False
    m = _SECTION_WORD_RE.match(line.rstrip("\r\n"))
    return bool(m) and not re.match(r"\d", m.group("value"))


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _plain_med_header(line: str, head: str) -> bool:
    """A medication header without a colon must look like a heading ("DISCHARGE
    MEDICATIONS", "Home Medications"), not a sentence ("Patient tolerating medications well")."""
    s = line.strip()
    if s != head.strip():
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", s)
    if not words or words[0].lower() in _HEADER_VERBS:
        return False
    if s.upper() == s or all(w[0].isupper() for w in words if w.lower() not in _SMALL_WORDS):
        return True
    return len(words) <= 3 and re.search(
        r"(?:medications?|meds|medicines|prescriptions?|regimens?)(?:[ \t]+list)?$", s, re.I) is not None


def _header_spans(text: str, header_re: re.Pattern, max_lines: int = 15) -> list[tuple[int, int]]:
    """Spans of the sections whose first line matches header_re: the header line plus the
    lines under it, up to a blank line or the next heading."""
    spans = []
    lines = text.splitlines(keepends=True)
    pos = i = 0
    while i < len(lines):
        if not header_re.match(lines[i]):
            pos += len(lines[i])
            i += 1
            continue
        start, end, j = pos, pos + len(lines[i]), i + 1
        while j < len(lines) and j - i <= max_lines and not _is_section_header(lines[j]):
            end += len(lines[j])
            j += 1
        spans.append((start, end))
        pos, i = end, j
    return spans


# A bullet or numbered item at the start of a line ("- Fever", "2) ...").
_BULLET_START_RE = re.compile(r"^\s*(?:[-*•·–>]|\(?\d{1,2}[.)])\s+")


def _continues_sentence(prev: str, line: str) -> bool:
    """Is `line` the wrapped rest of the sentence or list on `prev`? Only when it is not a
    new bullet, prev did not end the sentence, and either prev ends mid-phrase ("worsening
    shortness of", "penicillin (rash),") or the line starts in lowercase ("  or 5 lb in
    one week"). An unpunctuated plan line ("Hypertension - continue lisinopril") is new
    content. Used for return precautions and allergy lines that start on a heading line."""
    prev = prev.rstrip("\r\n")
    if _BULLET_START_RE.match(line) or re.search(r"[.!?][)\"']?\s*$", prev):
        return False
    return bool(re.match(r"[ \t]*[a-z(]", line) or _DANGLING_RE.search(prev))


# The heading word, then a separator: "Allergies: ...", "Allergies - Penicillin (rash)",
# a tabular "ALLERGIES    Penicillin (hives)" (tab or 2+ spaces), or nothing at all.
_ALLERGY_HEADER_RE = re.compile(
    r"^\s*(?:drug\s+|medication\s+|known\s+)?allerg(?:y|ies)\b"
    r"(?:(?:[^:\n]{0,20}:|[ \t]*[-–—][ \t]*|[ \t]*\t|[ \t]{2,})[ \t]*(?P<rest>.*)|[ \t]*)$", re.I)
_NO_ALLERGY_RE = re.compile(r"^(?:nkda|nka|nkma|none|no\s+known)\b", re.I)
_ALLERGIC_TO_RE = re.compile(r"^\s*(?:(?:patient|pt|she|he)\s+(?:is\s+)?)?allergic\s+to\b", re.I)


def _allergy_spans(text: str) -> list[tuple[int, int]]:
    """Spans of "Allergies: penicillin (rash)" sections; reactions listed there are not
    the patient's conditions, so they should not pull in topics like Rashes.

    Needs "Allergy"/"Allergies" followed by a colon, a dash or a column gap (or the bare
    word as a heading), so a problem-list line like "Allergic rhinitis - seasonal" is not
    one. A heading on its own line covers the list below it. A heading with the allergies
    on the same line ("Allergies: penicillin (rash)") covers only that line plus its
    wrapped or indented continuation, so a following "Asthma" line is still scanned.
    "Allergies: NKDA" and "Allergic to sulfa (rash)" cover only their own line, and a
    "Discharge Diagnosis: ..." line always ends the section."""
    spans = []
    lines = text.splitlines(keepends=True)
    pos = i = 0
    while i < len(lines):
        m = _ALLERGY_HEADER_RE.match(lines[i].rstrip("\r\n"))
        if not m:
            if _ALLERGIC_TO_RE.match(lines[i]):
                spans.append((pos, pos + len(lines[i])))
            pos += len(lines[i])
            i += 1
            continue
        start, end, j = pos, pos + len(lines[i]), i + 1
        rest = (m.group("rest") or "").strip()
        if not _NO_ALLERGY_RE.match(rest):   # NKDA: nothing follows
            while (j < len(lines) and j - i <= 10 and not _is_section_header(lines[j])
                   and not _starts_new_section(lines[j])):
                if rest and not (_continues_sentence(lines[j - 1], lines[j])
                                 or _indent(lines[j]) > _indent(lines[i])):
                    break   # inline list: the next unindented, unwrapped line is new content
                end += len(lines[j])
                j += 1
        spans.append((start, end))
        pos, i = end, j
    return spans


# "RETURN TO THE ED FOR:", "- Call 911 for ...", "Return to the ED or call if: ...",
# "Warning signs:". The symptoms listed there are ones the patient does NOT have.
_PRECAUTION_RE = re.compile(
    r"^\s*(?:[-*•·–>]\s*|\(?\d{1,2}[.)]\s*)?(?:please\s+)?"
    r"(?:(?:return\s+precautions|warning\s+signs|when\s+to\s+(?:call|seek|return)\b[^:\n]{0,40})\s*:"
    r"|(?:call|return|go\s+to|seek)\b(?![^\n]*\b(?:schedul\w*|appointment|refills?|results?)\b)"
    r"[^\n]{0,60}?\b(?:for|if|when)\b)",
    re.I,
)


def _return_precaution_spans(text: str) -> list[tuple[int, int]]:
    """Spans of return-precaution / warning-sign text ("Call 911 for chest pain"). A header
    that ends with the trigger ("... RETURN TO THE ED FOR:") covers the list below it; an
    inline one covers its own sentence, including its wrapped lines. Both stop at a new
    heading ("Discharge Diagnosis: ...")."""
    spans = []
    lines = text.splitlines(keepends=True)
    pos = i = 0
    while i < len(lines):
        line = lines[i]
        m = _PRECAUTION_RE.match(line)
        if not m:
            pos += len(line)
            i += 1
            continue
        start, end = pos, pos + len(line)
        block = not line[m.end():].strip(" \t\r\n:")   # header only -> a list follows
        j = i + 1
        while (j < len(lines) and j - i <= 12 and not _is_section_header(lines[j])
               and not _starts_new_section(lines[j])):
            if not block and not _continues_sentence(lines[j - 1], lines[j]):
                break   # inline: the sentence ended, so the next line is new content
            end += len(lines[j])
            j += 1
        spans.append((start, end))
        pos, i = end, j
    return spans


# Result-interpretation legend rows list every category, not the patient's condition:
#   "  5.7-6.4%   Prediabetes", ">=6.5%\tDiabetes"   (threshold/range + column gap + label)
#   "<150 normal | 150-199 borderline high | ..."  (threshold/range + "|"-separated labels)
# A bare leading number ("4 days of cough", "10-14 days of amoxicillin") is not enough.
_LEGEND_THRESH = (r"(?:[<>≤≥]=?[ \t]*\d{1,6}(?:\.\d{1,4})?|\d{1,6}(?:\.\d{1,4})?[ \t]*[-–][ \t]*\d{1,6}"
                  r"(?:\.\d{1,4})?)[ \t]*(?:%|[A-Za-zµ]{1,10}/[\w.]{1,10})?")
_LEGEND_ROW_RE = re.compile(
    rf"^[ \t]*{_LEGEND_THRESH}(?:[ \t]{{2,}}|\t)[A-Za-z][^\n]{{0,60}}$"
    rf"|^[ \t]*{_LEGEND_THRESH}[ \t]+[A-Za-z][^|\n]*\|")


def _legend_spans(text: str) -> list[tuple[int, int]]:
    """Spans of result-interpretation legend rows ('5.7-6.4%  Prediabetes')."""
    spans, pos = [], 0
    for line in text.splitlines(keepends=True):
        if _LEGEND_ROW_RE.match(line.rstrip("\r\n")):
            spans.append((pos, pos + len(line)))
        pos += len(line)
    return spans


# Section headings used to rank topics: a diagnosis named under DIAGNOSES / IMPRESSION /
# ASSESSMENT comes first; mentions under instructions and follow-up count for less.
_DX_HEADER_RE = re.compile(
    r"^\s*(?:(?:principal|primary|secondary|final|discharge|admitting|admission|working|clinical|other)\s+)?"
    r"(?:diagnos[ie]s|dx|impressions?|assessment(?:\s*(?:and|&|/)\s*plan)?|problem list|active problems)"
    r"\b[^:\n]{0,25}(?::|\s*$)", re.I)
_INSTR_HEADER_RE = re.compile(
    r"^\s*(?:(?:patient|discharge|home)\s+)?"
    r"(?:instructions?|sick[- ]day rules|return precautions|when to (?:call|seek)\b[^:\n]{0,40}|warning signs"
    r"|follow[- ]?up|wound care|activity|diet)\b[^:\n]{0,25}(?::|\s*$)", re.I)


def _medication_candidates(text: str) -> list[list[str]]:
    """Groups of alternative names, one group per medication mention, in text order."""
    return _medication_mentions(text)[0]


def _medication_mentions(text: str, prefer_lists: bool = False,
                         ) -> tuple[list[list[str]], list[tuple[int, int]], list[frozenset[str]]]:
    """Medication candidates (from "<name> <dose>" and lines under a MEDICATIONS header).

    Returns (groups of alternative names, character spans of the mentions, a route hint
    per group). Each group is tried in order until one name is confirmed. Groups are in
    text order, or with prefer_lists the discharge/current list first, then dose mentions
    in the narrative, then in-hospital lists, so that a one-off pre-op antibiotic does not
    take a drug slot from the medicines the patient goes home with.
    """
    found: list[tuple[int, list[str], str]] = []    # (offset, variants, text around it)
    spans: list[tuple[int, int]] = []
    raw_lines = text.splitlines(keepends=True)
    lines = [line.rstrip("\r\n") for line in raw_lines]
    offsets, pos = [], 0
    for raw in raw_lines:
        offsets.append(pos)
        pos += len(raw)

    def line_at(offset: int) -> str:
        start = text.rfind("\n", 0, offset) + 1
        end = text.find("\n", offset)
        return text[start:end if end >= 0 else len(text)]

    for m in _DOSE_RE.finditer(text):
        words = _dose_name_words(m)
        if not words:
            continue
        w1, w2 = (words[0][0] if len(words) == 2 else None), words[-1][0]
        variants = _variants(w1, w2)
        if variants:
            # Start the span at the first word actually used, and never on the line above
            # ("history of asthma\nalbuterol 2 puffs" must keep the Asthma topic).
            uses_w1 = w1 is not None and (variants[0] == f"{w1} {w2}" or w1 in variants)
            start = words[0][1] if uses_w1 and "\n" not in text[words[0][1]:words[-1][1]] else words[-1][1]
            found.append((start, variants, line_at(start)))
            spans.append((start, m.end()))

    list_spans: list[tuple[int, int, int]] = []    # (start, end, rank) of each medication list
    i = 0
    while i < len(lines):
        m = _MED_HEADER_RE.match(lines[i])
        head = (m.group("head") if m else "") or ""
        has_colon = bool(m and m.group("rest") is not None)
        if not m or "allerg" in head.lower() or (not has_colon and not _plain_med_header(lines[i], head)):
            i += 1
            continue
        rest = m.group("rest") or ""
        rest_start = offsets[i] + len(lines[i]) - len(rest)
        items = []   # [start, end, text]; end grows when a wrapped line is joined to the item
        for part in re.finditer(r"[^,;]+", rest):
            if part.group(0).strip():
                start = rest_start + part.start()
                items.append([start, start + len(part.group(0)), part.group(0)])
        item_indent = _indent(lines[i]) if items else None
        marked = None    # numbered/bulleted list? decided by its first item
        j = i + 1
        while (j < len(lines) and j - i <= 25 and not _is_section_header(lines[j])
               and not _starts_new_section(lines[j])):
            line = lines[j]
            is_marked = _ITEM_MARKER_RE.match(line) is not None
            # A wrapped line or a sub-bullet belongs to the item above: deeper indent, any
            # unmarked line in a numbered list, or a line after "take with" / "as". It is
            # not a new drug candidate ("   needed for pain" is not a medicine called "needed").
            if items and (_indent(line) > item_indent
                          or (not is_marked and (marked or _DANGLING_RE.search(lines[j - 1])))):
                items[-1][1] = offsets[j] + len(line)
            else:
                if marked is None:
                    marked = is_marked
                if item_indent is None:
                    item_indent = _indent(line)
                items.append([offsets[j], offsets[j] + len(line), line])
            j += 1
        list_end = offsets[j] if j < len(offsets) else len(text)
        inpatient = _INPATIENT_LIST_RE.search(head) and not _TAKE_HOME_LIST_RE.search(head)
        list_spans.append((offsets[i], list_end, 2 if inpatient else 0))
        for start, end, item in items:
            lead = _LEAD_WORDS_RE.match(_BULLET_RE.sub("", item).strip())
            if lead:
                variants = _variants(lead.group(1), lead.group(2))
                if variants:
                    found.append((start, variants, text[start:end]))
                    spans.append((start, end))   # wrapped lines stay inside the item's span
        i = j

    def rank(offset: int) -> int:
        # 0 = discharge/current/home list, 1 = dose mention in the narrative, 2 = in-hospital list
        for start, end, r in list_spans:
            if start <= offset < end:
                return r
        return 1

    found.sort(key=(lambda f: (rank(f[0]), f[0])) if prefer_lists else (lambda f: f[0]))
    groups: list[list[str]] = []
    hints: list[frozenset[str]] = []
    seen: set[str] = set()
    for _, variants, context in found:
        if _norm(variants[0]) in seen:   # its most specific name was already tried
            continue
        fresh = [v for v in variants if _norm(v) not in seen]   # "Insulin lispro" after "insulin glargine"
        groups.append(fresh)
        hints.append(_route_hint(context))
        seen.update(_norm(v) for v in fresh)
    return groups, spans, hints


def _confirm_drugs(groups: list[list[str]], hints: Optional[list[frozenset[str]]] = None) -> list[Reference]:
    """Look up candidate groups concurrently (<= MAX_DRUG_LOOKUPS names in total)."""
    hints = hints or [frozenset()] * len(groups)
    budgeted: list[tuple[list[str], frozenset[str]]] = []
    remaining = MAX_DRUG_LOOKUPS
    for group, hint in zip(groups, hints):
        if remaining <= 0:
            break
        budgeted.append((group[:remaining], hint))
        remaining -= len(budgeted[-1][0])
    if not budgeted:
        return []
    deadline = time.monotonic() + DRUG_PHASE_BUDGET_S

    def work(group: list[str], hint: frozenset[str]) -> Optional[Reference]:
        for name in group:
            if time.monotonic() > deadline:
                return None
            ref = lookup_drug(name, route_hint=hint)
            if ref is not None:
                return ref
        return None

    pool = ThreadPoolExecutor(max_workers=min(6, len(budgeted)), thread_name_prefix="mplus-drug")
    try:
        futures = [pool.submit(work, g, h) for g, h in budgeted]
        wait(futures, timeout=DRUG_PHASE_BUDGET_S + 0.5)
        out = []
        for fut in futures:
            if fut.done() and not fut.cancelled() and fut.exception() is None and fut.result():
                out.append(fut.result())
        return out
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


# =============================================================================
# Public API
# =============================================================================

def _dedupe(refs: Iterable[Reference], seen: set[str]) -> list[Reference]:
    out = []
    for ref in refs:
        if ref.url and ref.url not in seen:
            seen.add(ref.url)
            out.append(ref)
    return out


def _allocate(groups: list[list[Reference]], max_refs: int) -> list[Reference]:
    """Fill max_refs slots: labs first, but keep up to 2 slots each for topics and drugs."""
    floors = [max(0, max_refs - 4), 2, 2]
    take = [0] * len(groups)
    budget = max_refs
    for i, group in enumerate(groups):
        n = min(len(group), floors[i], budget)
        take[i], budget = n, budget - n
    for i, group in enumerate(groups):
        n = min(len(group) - take[i], budget)
        take[i], budget = take[i] + n, budget - n
    return [ref for i, group in enumerate(groups) for ref in group[:take[i]]]


def find_references(document_text: str, labs: list, max_refs: int = 8) -> list[Reference]:
    """MedlinePlus passages for a document, chosen before the LLM call.

    Order: lab-test pages for abnormal labs, then health topics mentioned in the
    text, then medication pages. Deduplicated by URL and capped at max_refs.
    """
    text = document_text or ""
    labs = list(labs or [])
    if max_refs <= 0:
        return []
    tests = _load_index()["lab_tests"]

    # 1) lab-test pages for abnormal labs, most abnormal first
    lab_refs: list[Reference] = []
    included_keys: set[str] = set()
    present_keys: set[str] = set()
    try:
        abnormal: list[tuple[float, int, str, str]] = []
        for pos, lab in enumerate(labs):
            key = _lab_key(lab)
            if not key:
                continue
            present_keys.add(key)
            if _attr(lab, "flag") in ("high", "low"):
                abnormal.append((-_severity(lab), pos, key, str(_attr(lab, "name", "") or key)))
        abnormal.sort()
        by_url: dict[str, list[str]] = {}
        url_key: dict[str, str] = {}
        for _, _, key, name in abnormal:
            url = tests[key]["url"]
            by_url.setdefault(url, [])
            if name not in by_url[url]:
                by_url[url].append(name)
            url_key.setdefault(url, key)
            included_keys.add(key)
        lab_refs = [_lab_reference(url_key[url], ", ".join(names[:3])) for url, names in by_url.items()]
    except Exception:
        log.exception("lab reference selection failed")

    try:
        med_groups, med_spans, med_hints = _medication_mentions(text, prefer_lists=True)
    except Exception:
        log.exception("medication extraction failed")
        med_groups, med_spans, med_hints = [], [], []

    # 2) health topics mentioned in the document. Skipped: medication mentions (they get
    # drug pages), allergy reactions, return precautions ("call 911 for chest pain") and
    # interpretation legends ("5.7-6.4%  Prediabetes"): none of them is the patient's condition.
    topic_refs: list[Reference] = []
    topic_hits: list[_TopicHit] = []
    try:
        covered = _topic_ids(t for k in included_keys for t in LAB_ANALYTE_TOPICS.get(k, ()))
        listed = _topic_ids(t for k in present_keys for t in LAB_ANALYTE_TOPICS.get(k, ()))
        skip = med_spans + _allergy_spans(text) + _return_precaution_spans(text) + _legend_spans(text)
        dx_spans = _header_spans(text, _DX_HEADER_RE)
        instr_spans = _header_spans(text, _INSTR_HEADER_RE)
        for hit in _scan_topics(text, skip, dx_spans, instr_spans):
            if hit.topic in covered:
                continue  # the lab-test page already explains it
            matched = hit.matched
            if hit.topic in listed:
                # "Calcium 9.4 mg/dL" in the table is no reason to explain calcium, but
                # "hypercalcemia" in the assessment is.
                conditions = [f for f in hit.forms if _norm(f).startswith(("hyper", "hypo"))]
                if not conditions:
                    continue
                matched = conditions[0]
            topic_refs.append(_topic_reference(hit.topic, matched))
            topic_hits.append(hit)
    except Exception:
        log.exception("topic matching failed")

    # 3) medications (live lookups)
    drug_refs: list[Reference] = []
    try:
        drug_refs = _confirm_drugs(med_groups, med_hints)
    except Exception:
        log.exception("medication lookup failed")

    # A topic matched only by a confirmed drug's own name ("insulin" -> Diabetes Medicines
    # when the Insulin Glargine page is there) adds nothing, unless it is a stated diagnosis.
    drug_words = {t for r in drug_refs for t in _tokens(r.matched or "")}
    if drug_words:
        topic_refs = [ref for ref, hit in zip(topic_refs, topic_hits)
                      if hit.in_dx or not all(set(_tokens(f)) <= drug_words for f in hit.forms)]

    seen: set[str] = set()
    groups = [_dedupe(lab_refs, seen), _dedupe(topic_refs, seen), _dedupe(drug_refs, seen)]
    return _allocate(groups, max_refs)


# ---- learn more ----------------------------------------------------------------

_LAB_FILLER = frozenset({"test", "level", "blood", "serum", "count", "result", "value", "lab", "high", "low"})


def _exact_concept_topic(term: str) -> Optional[int]:
    """Whole-term match against titles/synonyms (modifiers like "mild" ignored)."""
    _, table, _ = _phrase_tables()
    toks = tuple(_tokens(term))
    for cand in (toks, tuple(t for t in toks if t not in _CONCEPT_MODIFIERS)):
        idx = table.get(cand) if cand else None
        if idx is not None and idx not in _stop_topic_ids():
            return idx
    return None


def _scan_concept_topic(term: str) -> Optional[int]:
    """A known phrase inside the term covering >= 1/3 of its words
    ("ACE inhibitor therapy", "atrial fibrillation with rapid ventricular response")."""
    content = _concept_tokens(term)
    best: Optional[tuple[int, int]] = None
    for hit in _scan_topics(term):
        n = len(_tokens(hit.matched))
        if hit.topic not in _stop_topic_ids() and n * 3 >= len(content) and (best is None or n > best[0]):
            best = (n, hit.topic)
    return best[1] if best else None


def _concept_lab(term: str) -> Optional[str]:
    alias = _lab_alias_map()
    toks = _tokens(term)
    for cand in (toks, [t for t in toks if t not in _CONCEPT_MODIFIERS and t not in _LAB_FILLER]):
        key = alias.get(" ".join(cand)) if cand else None
        if key:
            return key
    return None


_COVERED = "covered"


def _resolve_concept_offline(term: str, kind: str, covered_topics: set[int], exclude: set[str]):
    """Returns a Reference, _COVERED (already explained by an existing reference) or None."""
    tests = _load_index()["lab_tests"]

    def lab_ref(key: str):
        return _COVERED if tests[key]["url"] in exclude else _lab_reference(key, term)

    def topic_ref(idx: int):
        return _COVERED if idx in covered_topics else _topic_reference(idx, term)

    if kind == "test":
        key = _concept_lab(term)
        if key:
            return lab_ref(key)
    idx = _exact_concept_topic(term)
    if idx is not None:
        return topic_ref(idx)
    if kind == "medication":
        return None  # live drug lookup next, then the fuzzy fallbacks
    if kind != "test":
        key = _concept_lab(term)
        if key:
            return lab_ref(key)
    return _fuzzy_concept(term, covered_topics)


def _fuzzy_concept(term: str, covered_topics: set[int]):
    """Thresholded BM25 first (precise when it accepts), then a phrase inside the term."""
    toks = _tokens(term)
    blocked = _blocked_concepts()
    if tuple(toks) in blocked or tuple(t for t in toks if t not in _CONCEPT_MODIFIERS) in blocked:
        return None  # "blood gas" is a test, not Gangrene
    idx = _bm25_concept(term)
    if idx is None:
        idx = _scan_concept_topic(term)
    if idx is None:
        return None
    return _COVERED if idx in covered_topics else _topic_reference(idx, term)


def _drug_term_names(term: str) -> list[str]:
    """Names to try for a medication concept: the term, then the base drug without salt or
    formulation words ("docusate sodium" -> "docusate", "metoprolol succinate ER" -> "metoprolol")."""
    base = " ".join(w for w in term.split() if _norm(w) not in _SALT_WORDS | _FORM_WORDS)
    return [term] + ([base] if base and _norm(base) != _norm(term) else [])


def _lookup_first(names: list[str], route_hint: frozenset[str]) -> Optional[Reference]:
    for name in names:
        ref = lookup_drug(name, route_hint=route_hint)
        if ref is not None:
            return ref
    return None


def _document_route_hint(term: str, document_text: str) -> frozenset[str]:
    """Routes named on the document lines that mention a medication concept
    ("albuterol/ipratropium nebulizers" -> {"inhalation"})."""
    toks = set(_tokens(term)) - _SALT_WORDS - _FORM_WORDS
    if not toks or not document_text:
        return frozenset()
    lines = [line for line in document_text.splitlines() if toks & set(_tokens(line))]
    return _route_hint(" ".join(lines))


def learn_more_for_concepts(concepts: list[dict], exclude_urls: set[str], max_items: int = 6,
                            document_text: str = "") -> list[Reference]:
    """"Learn more" pages for the concepts the LLM extracted (runs after the LLM call).

    Per concept: lab-test page (tests) / exact synonym match -> live drug lookup
    (medications) -> phrase inside the term -> thresholded BM25. Concepts already
    explained by a reference in exclude_urls (including "hyperkalemia" when the
    Potassium Blood Test page is there) are skipped. document_text (optional) lets a
    medication's route in the document pick the right drug page (inhaler vs nose spray).
    """
    if max_items <= 0:
        return []
    tests = _load_index()["lab_tests"]
    exclude = set(exclude_urls or ())
    url_to_lab = {lab["url"]: key for key, lab in tests.items()}
    covered_topics = _topic_ids(
        t for url in exclude if url in url_to_lab for t in LAB_ANALYTE_TOPICS.get(url_to_lab[url], ()))

    resolved: list[Optional[Reference]] = []
    pending: dict[int, str] = {}       # index -> medication term needing a live lookup
    seen_terms: set[str] = set()
    for i, concept in enumerate(concepts or []):
        resolved.append(None)
        term = " ".join(str(_attr(concept, "term", "") or "").split())
        kind = str(_attr(concept, "kind", "") or "").lower()
        norm = _norm(term)
        if not norm or norm in seen_terms:
            continue
        seen_terms.add(norm)
        try:
            ref = _resolve_concept_offline(term, kind, covered_topics, exclude)
        except Exception:
            log.exception("learn_more resolution failed")
            continue
        if isinstance(ref, Reference):
            resolved[i] = ref
        elif ref is None and kind == "medication":
            pending[i] = term

    if pending:
        jobs = dict(list(pending.items())[:MAX_DRUG_LOOKUPS])
        pool = ThreadPoolExecutor(max_workers=min(6, len(jobs)), thread_name_prefix="mplus-learn")
        try:
            futures = {i: pool.submit(_lookup_first, _drug_term_names(term),
                                      _document_route_hint(term, document_text))
                       for i, term in jobs.items()}
            wait(list(futures.values()), timeout=DRUG_PHASE_BUDGET_S)
            for i, term in pending.items():
                fut = futures.get(i)
                ref = fut.result() if fut is not None and fut.done() and fut.exception() is None else None
                if ref is not None:
                    resolved[i] = replace(ref, matched=term)
                    continue
                try:
                    fallback = _fuzzy_concept(term, covered_topics)
                except Exception:
                    log.exception("learn_more fallback failed")
                    fallback = None
                resolved[i] = fallback if isinstance(fallback, Reference) else None
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    out: list[Reference] = []
    seen_urls = set(exclude)
    for ref in resolved:
        if ref is not None and ref.url not in seen_urls:
            seen_urls.add(ref.url)
            out.append(ref)
            if len(out) >= max_items:
                break
    return out
