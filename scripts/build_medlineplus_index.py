#!/usr/bin/env python3
"""Build the offline MedlinePlus index used by backend/medlineplus.py.

Usage:
    .venv/bin/python scripts/build_medlineplus_index.py [--xml PATH] [--skip-labs]

Without --xml the script reads https://medlineplus.gov/xml.html, finds the newest
daily health-topics file (mplus_topics_YYYY-MM-DD.xml) and downloads it.

Output: backend/data/medlineplus_index.json with

* metadata   - source XML date, generation time, counts, attribution.
* topics     - every English MedlinePlus health topic: id, title, url, description
               (meta-desc without the "Learn more."-style call to action), synonyms
               (also-called + see-reference) and ~900 chars of plain-text summary.
* lab_tests  - one MedlinePlus lab-test page per canonical lab key (the keys the lab
               parser emits), found by asking MedlinePlus Connect about a LOINC code.

All sources are free NLM services that need no API key. Runtime code never calls
this script; re-run it occasionally to refresh the data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import ssl
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "backend", "data", "medlineplus_index.json")

XML_LISTING_URL = "https://medlineplus.gov/xml.html"
CONNECT_URL = "https://connect.medlineplus.gov/service"
LOINC_OID = "2.16.840.1.113883.6.1"
USER_AGENT = "Clearchart-index-builder/1.0 (educational; contact via repo)"
ATTRIBUTION = "MedlinePlus, U.S. National Library of Medicine"

TOPIC_TEXT_CHARS = 900
LAB_INTRO_CHARS = 300
LAB_RESULTS_CHARS = 480
LAB_DESC_CHARS = 260
CONNECT_SLEEP_S = 0.3

# Canonical lab key -> LOINC codes to try (in order) and the names a report may use.
# The names are also stored in the index: the runtime uses them to map unparsed lab
# names to keys and to avoid mistaking "Potassium 4.1" for a medication.
LAB_TESTS: dict[str, dict] = {
    "glucose": {"loinc": ["2345-7", "2339-0"], "names": ["glucose", "blood glucose", "blood sugar", "fasting glucose", "glu"]},
    "sodium": {"loinc": ["2951-2"], "names": ["sodium", "na"]},
    "potassium": {"loinc": ["2823-3"], "names": ["potassium", "k"]},
    "chloride": {"loinc": ["2075-0"], "names": ["chloride", "cl"]},
    "co2": {"loinc": ["2028-9", "1963-8"], "names": ["co2", "carbon dioxide", "bicarbonate", "hco3", "total co2"]},
    "bun": {"loinc": ["3094-0", "6299-2"], "names": ["bun", "blood urea nitrogen", "urea nitrogen"]},
    "creatinine": {"loinc": ["2160-0"], "names": ["creatinine", "creat", "cr"]},
    "egfr": {"loinc": ["98979-8", "33914-3", "62238-1"], "names": ["egfr", "gfr", "estimated gfr", "glomerular filtration rate"]},
    "calcium": {"loinc": ["17861-6"], "names": ["calcium", "ca"]},
    "magnesium": {"loinc": ["19123-9", "2601-3"], "names": ["magnesium"]},
    "phosphorus": {"loinc": ["2777-1", "14879-1"], "names": ["phosphorus", "phosphate", "phos"]},
    "total_protein": {"loinc": ["2885-2"], "names": ["total protein", "protein"]},
    "albumin": {"loinc": ["1751-7"], "names": ["albumin", "alb"]},
    "total_bilirubin": {"loinc": ["1975-2"], "names": ["bilirubin", "total bilirubin", "tbili", "t bili"]},
    "alk_phos": {"loinc": ["6768-6"], "names": ["alkaline phosphatase", "alk phos", "alp"]},
    "alt": {"loinc": ["1742-6", "1743-4"], "names": ["alt", "alanine aminotransferase", "alanine transaminase", "sgpt"]},
    "ast": {"loinc": ["1920-8"], "names": ["ast", "aspartate aminotransferase", "aspartate transaminase", "sgot"]},
    "total_cholesterol": {"loinc": ["2093-3"], "names": ["total cholesterol", "cholesterol"]},
    "ldl": {"loinc": ["13457-7", "2089-1", "18262-6"], "names": ["ldl", "ldl cholesterol", "ldl c", "low density lipoprotein"]},
    "hdl": {"loinc": ["2085-9"], "names": ["hdl", "hdl cholesterol", "hdl c", "high density lipoprotein"]},
    "triglycerides": {"loinc": ["2571-8"], "names": ["triglycerides", "triglyceride", "trig", "tg"]},
    "a1c": {"loinc": ["4548-4", "17856-6"], "names": ["a1c", "hba1c", "hemoglobin a1c", "glycated hemoglobin", "glycohemoglobin"]},
    "tsh": {"loinc": ["3016-3", "11580-8"], "names": ["tsh", "thyroid stimulating hormone", "thyrotropin"]},
    "free_t4": {"loinc": ["3024-7"], "names": ["free t4", "ft4", "t4", "thyroxine", "free thyroxine"]},
    "wbc": {"loinc": ["6690-2", "26464-8"], "names": ["wbc", "white blood cell", "white blood count", "white blood cell count", "leukocytes"]},
    "rbc": {"loinc": ["789-8"], "names": ["rbc", "red blood cell", "red blood cell count", "erythrocytes"]},
    "hemoglobin": {"loinc": ["718-7"], "names": ["hemoglobin", "hgb", "hb"]},
    "hematocrit": {"loinc": ["4544-3"], "names": ["hematocrit", "hct"]},
    "mcv": {"loinc": ["787-2"], "names": ["mcv", "mean corpuscular volume"]},
    "platelets": {"loinc": ["777-3", "26515-7"], "names": ["platelets", "platelet", "platelet count", "plt"]},
    "ferritin": {"loinc": ["2276-4"], "names": ["ferritin"]},
    "inr": {"loinc": ["6301-6", "34714-6"], "names": ["inr", "pt inr", "prothrombin time", "protime"]},
    "psa": {"loinc": ["2857-1", "19197-3"], "names": ["psa", "prostate specific antigen"]},
    "vitamin_d": {"loinc": ["1989-3", "62292-8"], "names": ["vitamin d", "25 hydroxyvitamin d", "25 oh vitamin d", "25 oh d"]},
    "bnp": {"loinc": ["30934-4", "33762-6"], "names": ["bnp", "nt probnp", "b type natriuretic peptide", "natriuretic peptide"]},
    "uacr": {"loinc": ["9318-7", "14959-1", "14958-3"], "names": ["uacr", "acr", "albumin creatinine ratio", "urine albumin creatinine ratio", "microalbumin creatinine ratio", "microalbumin"]},
    "anion_gap": {"loinc": ["33037-3", "1863-0", "10466-1"], "names": ["anion gap"]},
}

# Used only if Connect returns nothing usable for a key.
LAB_FALLBACK_TOPICS = {
    "glucose": "Blood Glucose",
    "a1c": "A1C",
    "total_cholesterol": "Cholesterol Levels: What You Need to Know",
    "ldl": 'LDL: The "Bad" Cholesterol',
    "hdl": 'HDL: The "Good" Cholesterol',
    "triglycerides": "Triglycerides",
    "egfr": "Kidney Tests",
    "bun": "Kidney Tests",
    "creatinine": "Kidney Tests",
    "tsh": "Thyroid Tests",
    "free_t4": "Thyroid Tests",
    "vitamin_d": "Vitamin D",
    "ferritin": "Iron",
}


# --------------------------------------------------------------------------- HTTP

def _ssl_context() -> ssl.SSLContext:
    # python.org builds on macOS ship without a CA bundle; certifi is already
    # installed as a dependency of anthropic/httpx, so prefer it when present.
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_CTX = _ssl_context()


def http_get(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as resp:
        return resp.read()


def strip_tracking(url: str) -> str:
    """Remove utm_* query parameters."""
    parts = urllib.parse.urlsplit(url)
    query = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True) if not k.lower().startswith("utm_")]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


# --------------------------------------------------------------------------- text

class SummaryParser(HTMLParser):
    """Turn MedlinePlus summary HTML into blocks: (kind, text, section, lead).

    kind is "p" (paragraph), "li" (bullet), "tr" (table row) or "h" (heading). section is the most
    recent heading. lead is the bold text a bullet starts with, if any (lab pages
    write result bullets as "<strong>Kidney disease.</strong> Long explanation").
    """

    BLOCK = {"p", "div", "ul", "ol", "table", "section", "blockquote"}
    HEAD = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str, str, str]] = []
        self._buf: list[str] = []
        self._kind = "p"
        self._section = ""
        self._lead: list[str] | None = None
        self._in_lead = False

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK:
            self._flush()
            self._kind = "p"
        elif tag == "li":
            self._flush()
            self._kind = "li"
        elif tag == "tr":
            self._flush()
            self._kind = "tr"
        elif tag in ("td", "th"):
            self._buf.append(" | ")
        elif tag in self.HEAD:
            self._flush()
            self._kind = "h"
        elif tag == "br":
            self._buf.append(" ")
        elif tag in ("strong", "b") and self._kind == "li" and not "".join(self._buf).strip():
            self._lead = []
            self._in_lead = True

    def handle_endtag(self, tag):
        if tag in ("strong", "b"):
            self._in_lead = False
        elif tag in self.BLOCK or tag in self.HEAD or tag in ("li", "tr"):
            self._flush()
            self._kind = "p"

    def handle_data(self, data):
        self._buf.append(data)
        if self._in_lead and self._lead is not None:
            self._lead.append(data)

    def _flush(self) -> None:
        text = " ".join("".join(self._buf).split()).strip(" |")
        lead = " ".join("".join(self._lead or []).split())
        self._buf, self._lead, self._in_lead = [], None, False
        if not text:
            return
        if self._kind == "h":
            self._section = text
            self.blocks.append(("h", text, text, ""))
        else:
            self.blocks.append((self._kind, text, self._section, lead))

    def close(self):
        super().close()
        self._flush()


def html_blocks(html: str) -> list[tuple[str, str, str, str]]:
    parser = SummaryParser()
    parser.feed(html or "")
    parser.close()
    return parser.blocks


def is_attribution(text: str) -> bool:
    """Trailing "NIH: National Institute of ..." / "Centers for Disease Control" lines."""
    return len(text) < 150 and not re.search(r"[.!?:]$", text)


def blocks_to_text(blocks, use_lead: bool = False) -> str:
    """Plain text: one line per paragraph, "• " bullets.

    Headings are dropped (they are questions like "What is it used for?"), and so are
    tables, which read badly as plain text, together with the "...:" line that
    introduces them. With use_lead, long bullets are reduced to their bold lead.
    """
    lines = []
    for i, (kind, text, _section, lead) in enumerate(blocks):
        nxt = blocks[i + 1][0] if i + 1 < len(blocks) else ""
        if kind in ("h", "tr") or (kind == "p" and text.endswith(":") and nxt == "tr"):
            continue
        if kind == "li":
            short_lead = lead and len(lead) >= 8 and not lead.endswith(":")
            item = lead if (use_lead and short_lead and len(text) > 90) else text
            lines.append("• " + item)
        else:
            lines.append(text)
    return "\n".join(lines)


_SENT_END = re.compile(r"[.!?](?=\s)|\n")


def truncate(text: str, limit: int) -> str:
    """Cut text at a sentence (or bullet) boundary at or before ``limit`` chars."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    ends = [m.end() for m in _SENT_END.finditer(cut)]
    if text[limit:limit + 1].isspace() and re.search(r"[.!?]$", cut):
        ends.append(len(cut))
    ends = [e for e in ends if e >= limit * 0.45]
    if ends:
        out = cut[: max(ends)].rstrip()
    else:
        out = cut[: cut.rfind(" ")].rstrip(" ,;:") + "…"
    # Don't end on a dangling lead-in like "Symptoms include:".
    if out.endswith(":"):
        shorter = out[: max(out.rfind("\n"), out.rfind(". ") + 1)].rstrip()
        if len(shorter) >= limit * 0.4:
            out = shorter
    return out


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_CTA = re.compile(r"^(Learn|Read|Find out|Get|Discover|See|Explore|Check|Here's|Know|Go)\b")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(" ".join(text.split())) if s.strip()]


def clean_description(meta_desc: str) -> str:
    """Drop trailing calls to action ("Learn more.", "Read about treatments.")."""
    sents = sentences(meta_desc or "")
    while sents and _CTA.match(sents[-1]):
        sents.pop()
    return " ".join(sents)


def first_sentences(text: str, max_sentences: int = 2, max_chars: int = LAB_DESC_CHARS) -> str:
    out = ""
    for sent in sentences(text.replace("\n• ", " ").replace("• ", ""))[:max_sentences]:
        candidate = (out + " " + sent).strip()
        if out and len(candidate) > max_chars:
            break
        out = candidate
    return truncate(out, max_chars) if len(out) > max_chars else out


def dedupe(items):
    seen, out = set(), []
    for item in items:
        item = " ".join((item or "").split())
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


# --------------------------------------------------------------------------- topics

def find_latest_xml_url() -> tuple[str, str]:
    html = http_get(XML_LISTING_URL).decode("utf-8", "replace")
    found = set(re.findall(r"mplus_topics_(\d{4}-\d{2}-\d{2})\.xml", html))
    if not found:
        raise SystemExit("Could not find an mplus_topics_YYYY-MM-DD.xml link on " + XML_LISTING_URL)
    date = max(found)
    return f"https://medlineplus.gov/xml/mplus_topics_{date}.xml", date


def download_xml() -> tuple[str, str]:
    url, date = find_latest_xml_url()
    print(f"Downloading {url} ...")
    data = http_get(url, timeout=300)
    # Untrusted download -> its own fresh directory, never executed.
    path = os.path.join(tempfile.mkdtemp(prefix="mplus_xml_"), os.path.basename(url))
    with open(path, "wb") as fh:
        fh.write(data)
    print(f"  saved {len(data) / 1e6:.1f} MB to {path}")
    return path, date


def parse_topics(xml_path: str) -> tuple[list[dict], str]:
    root = ET.parse(xml_path).getroot()
    source_date = root.get("date-generated", "")
    m = re.match(r"(\d{2})/(\d{2})/(\d{4})", source_date)
    if m:
        source_date = f"{m.group(3)}-{m.group(1)}-{m.group(2)}"
    topics = []
    for node in root.findall("health-topic"):
        if node.get("language") != "English":
            continue
        blocks = html_blocks(node.findtext("full-summary") or "")
        if blocks and blocks[-1][0] == "p" and is_attribution(blocks[-1][1]):
            blocks = blocks[:-1]
        text = truncate(blocks_to_text(blocks), TOPIC_TEXT_CHARS)
        description = clean_description(node.get("meta-desc", ""))
        # Some meta descriptions are teasers ("Are you tired?"); the summary's opening
        # sentences make a better card snippet.
        if text and (len(description) < 50 or "?" in description):
            description = first_sentences(text)
        text = text or description
        synonyms = dedupe([e.text for e in node.findall("also-called")] + [e.text for e in node.findall("see-reference")])
        title = node.get("title", "").strip()
        synonyms = [s for s in synonyms if s.lower() != title.lower()]
        topics.append({
            "id": node.get("id"),
            "title": title,
            "url": strip_tracking(node.get("url", "")),
            "description": description,
            "synonyms": synonyms,
            "text": text,
        })
    topics.sort(key=lambda t: t["title"].lower())
    return topics, source_date


# --------------------------------------------------------------------------- lab tests

def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def contains_phrase(haystack: str, needle: str) -> bool:
    return f" {norm(needle)} " in f" {norm(haystack)} "


def connect_loinc(code: str) -> list[dict]:
    params = {
        "mainSearchCriteria.v.cs": LOINC_OID,
        "mainSearchCriteria.v.c": code,
        "knowledgeResponseType": "application/json",
        "informationRecipient.languageCode.c": "en",
    }
    url = CONNECT_URL + "?" + urllib.parse.urlencode(params)
    data = json.loads(http_get(url, timeout=30).decode("utf-8"))
    return data.get("feed", {}).get("entry", []) or []


def parse_lab_entry(entry: dict) -> dict:
    title = (entry.get("title") or {}).get("_value", "").strip()
    links = entry.get("link") or [{}]
    url = strip_tracking(links[0].get("href", ""))
    blocks = html_blocks((entry.get("summary") or {}).get("_value", ""))
    other_names: list[str] = []
    kept = []
    for block in blocks:
        m = re.match(r"^Other names[^:]*:\s*(.+)$", block[1], re.I)
        if m and block[0] == "p":
            other_names += [n.strip(" .") for n in re.split(r",|;", m.group(1))]
            continue
        if block[2].lower() == "references":
            continue
        kept.append(block)
    sections: dict[str, list] = {}
    order: list[str] = []
    for block in kept:
        if block[0] == "h":
            continue
        if block[2] not in sections:
            order.append(block[2])
            sections[block[2]] = []
        sections[block[2]].append(block)
    intro_blocks = sections.get(order[0], []) if order else []
    results_key = next((s for s in order if "results mean" in s.lower()), None)
    intro = blocks_to_text(intro_blocks)
    description = first_sentences(intro)
    text = truncate(intro, LAB_INTRO_CHARS)
    if results_key:
        results = truncate(blocks_to_text(sections[results_key], use_lead=True), LAB_RESULTS_CHARS)
        text = f"{text}\nWhat the results mean:\n{results}"
    return {
        "title": title,
        "url": url,
        "description": description,
        "text": text,
        "synonyms": dedupe(other_names),
    }


def score_lab_entry(names: list[str], parsed: dict) -> float:
    score = 0.0
    if "/lab-tests/" in parsed["url"]:
        score += 2
    if any(contains_phrase(parsed["title"], n) for n in names):
        score += 3
    if any(contains_phrase(o, n) for o in parsed["synonyms"] for n in names if len(n) > 3):
        score += 1.5
    if re.search(r"\b(panel|function tests|blood count tests|indices)\b", parsed["title"].lower()):
        score -= 1
    return score


def build_lab_tests(topics: list[dict]) -> dict:
    by_title = {t["title"]: t for t in topics}
    out: dict[str, dict] = {}
    print("\nLab test pages (key -> MedlinePlus page):")
    for key, spec in LAB_TESTS.items():
        best, best_score, best_code = None, -99.0, None
        for code in spec["loinc"]:
            try:
                entries = connect_loinc(code)
            except Exception as exc:  # network hiccup: try the next code
                print(f"  ! {key}: LOINC {code} failed: {exc}")
                entries = []
            time.sleep(CONNECT_SLEEP_S)
            for entry in entries:
                parsed = parse_lab_entry(entry)
                s = score_lab_entry(spec["names"], parsed)
                if s > best_score:
                    best, best_score, best_code = parsed, s, code
            if best is not None and best_score >= 4:  # title names the test: good enough
                break
        if best is None or best_score < 2:
            topic = by_title.get(LAB_FALLBACK_TOPICS.get(key, ""))
            if topic:
                best = {"title": topic["title"], "url": topic["url"], "description": topic["description"],
                        "text": truncate(topic["text"], LAB_INTRO_CHARS + LAB_RESULTS_CHARS), "synonyms": []}
                best_code, source = None, "topic"
            else:
                print(f"  {key:18s} -> (no page found)")
                continue
        else:
            source = "connect"
        out[key] = {"key": key, "loinc": best_code, "source": source, "names": spec["names"], **best}
        print(f"  {key:18s} -> {best['title']}  [{best['url']}]  (LOINC {best_code}, {source})")
    return out


# --------------------------------------------------------------------------- main

def write_index(index: dict, path: str) -> None:
    """Compact JSON with one topic / lab test per line (small file, readable diffs)."""
    dump = lambda obj: json.dumps(obj, ensure_ascii=False, separators=(",", ":"))  # noqa: E731
    lines = ["{", '"metadata":' + json.dumps(index["metadata"], ensure_ascii=False, indent=1) + ",", '"topics":[']
    lines.append(",\n".join(dump(t) for t in index["topics"]))
    lines.append("],")
    lines.append('"lab_tests":{')
    lines.append(",\n".join(f"{dump(k)}:{dump(v)}" for k, v in index["lab_tests"].items()))
    lines.append("}}")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(tmp, encoding="utf-8") as fh:
        json.load(fh)  # sanity check before replacing the old index
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xml", help="path to an already-downloaded mplus_topics_*.xml")
    ap.add_argument("--out", default=OUT_PATH, help="output JSON path")
    ap.add_argument("--skip-labs", action="store_true", help="don't query MedlinePlus Connect (keeps existing lab_tests)")
    args = ap.parse_args(argv)
    # Resolve and create the output folder up front: a bare "--out index.json" has no
    # directory part (os.makedirs("") fails), and finding that out only after the
    # download and the Connect queries would throw all that work away.
    args.out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.xml:
        xml_path = args.xml
        m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(xml_path))
        file_date = m.group(1) if m else ""
    else:
        xml_path, file_date = download_xml()

    topics, source_date = parse_topics(xml_path)
    source_date = source_date or file_date
    print(f"Parsed {len(topics)} English health topics (source date {source_date}).")

    if args.skip_labs and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            lab_tests = json.load(fh).get("lab_tests", {})
    else:
        lab_tests = build_lab_tests(topics)
    missing = [k for k in LAB_TESTS if k not in lab_tests]

    index = {
        "metadata": {
            "source": "MedlinePlus health topics XML + MedlinePlus Connect (lab tests)",
            "source_xml_date": source_date,
            "source_xml_url": f"https://medlineplus.gov/xml/mplus_topics_{file_date or source_date}.xml",
            "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            "attribution": ATTRIBUTION,
            "counts": {"topics": len(topics), "lab_tests": len(lab_tests)},
            "lab_tests_missing": missing,
        },
        "topics": topics,
        "lab_tests": lab_tests,
    }
    write_index(index, args.out)
    size = os.path.getsize(args.out)
    print(f"\nWrote {args.out}: {len(topics)} topics, {len(lab_tests)} lab tests, {size / 1e6:.2f} MB")
    if missing:
        print("Lab keys without a page:", ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
