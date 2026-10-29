"""
API tests with a fake Claude client, so they run offline and cost nothing.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import llm
import main
import pipeline
from medlineplus import Reference

SAMPLE = (Path(__file__).parent.parent / "samples" / "sample_lab_report.txt").read_text()

TRANSLATION = {
    "explanation": (
        "## Lab results\n\nYour potassium was 5.8, which is high [1]. "
        "Your BUN was 22.\n\nThis is an educational explanation, not medical advice."
    ),
    "concepts": [{"term": "potassium", "kind": "test"}, {"term": "lisinopril", "kind": "medication"}],
}
JUDGE = {"total_claims_checked": 2, "supported_count": 2, "unsupported_claims": []}


class FakeMessages:
    def __init__(self, calls):
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        system = kwargs["system"][0]["text"]
        if "fact-checker" in system:
            payload = json.dumps(JUDGE)
        elif "transcribe" in system:
            payload = "PATIENT: TEST\nPotassium 5.8 mmol/L (3.5-5.0)"
        else:
            payload = json.dumps(TRANSLATION)
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=payload)],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )


class FakeClient:
    def __init__(self):
        self.calls = []
        self.messages = FakeMessages(self.calls)
        self.beta = SimpleNamespace(messages=FakeMessages(self.calls))


@pytest.fixture
def client(monkeypatch):
    fake = FakeClient()
    llm.set_client(fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(main, "RATE_LIMIT_PER_HOUR", 0)
    # Keep the tests offline: no live drug lookups.
    potassium_ref = Reference(
        title="Potassium Blood Test",
        url="https://medlineplus.gov/lab-tests/potassium-blood-test/",
        snippet="A potassium blood test measures the amount of potassium in your blood.",
        text="A potassium blood test measures the amount of potassium in your blood.",
        kind="lab_test",
        matched="Potassium",
    )
    monkeypatch.setattr(pipeline, "find_references", lambda text, labs: [potassium_ref])
    monkeypatch.setattr(pipeline, "learn_more_for_concepts", lambda concepts, exclude_urls, **kw: [])
    test_client = TestClient(main.app)
    test_client.fake = fake
    yield test_client
    llm.set_client(None)


def test_translate_returns_the_full_contract(client):
    res = client.post("/api/translate", data={"document_text": SAMPLE, "reading_level": "5th"})
    assert res.status_code == 200, res.text
    body = res.json()
    for key in ["explanation", "concepts", "lab_values", "references", "learn_more", "faithfulness", "readability"]:
        assert key in body
    assert body["reading_level"] == "5th"
    assert body["references"][0]["id"] == 1 and body["references"][0]["cited"] is True
    flags = {lab["canonical"]: lab["flag"] for lab in body["lab_values"]}
    assert flags["potassium"] == "high"
    assert body["faithfulness"]["numeric"]["unsupported_numbers"] == []
    assert body["faithfulness"]["judge"]["error"] is None


def test_translation_prompt_contains_flags_references_and_level(client):
    client.post("/api/translate", data={"document_text": SAMPLE, "reading_level": "5th"})
    translation_call = client.fake.calls[0]
    user_message = translation_call["messages"][0]["content"]
    assert "FLAG: HIGH" in user_message
    assert "[1] Potassium Blood Test" in user_message
    assert "5th-grade" in user_message
    assert translation_call["output_config"]["format"]["type"] == "json_schema"


def test_empty_document_is_rejected(client):
    res = client.post("/api/translate", data={"document_text": "   "})
    assert res.status_code == 400


def test_too_long_document_is_rejected(client):
    res = client.post("/api/translate", data={"document_text": "x" * (main.MAX_DOCUMENT_CHARS + 1)})
    assert res.status_code == 413


def test_missing_api_key_gives_a_clear_error(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    res = client.post("/api/translate", data={"document_text": SAMPLE})
    assert res.status_code == 503
    assert "ANTHROPIC_API_KEY" in res.json()["detail"]


def test_refusal_becomes_a_readable_error(client, monkeypatch):
    def refuse(**kwargs):
        return SimpleNamespace(stop_reason="refusal", content=[], usage=SimpleNamespace(input_tokens=1, output_tokens=0))

    monkeypatch.setattr(client.fake.beta.messages, "create", refuse)
    monkeypatch.setattr(client.fake.messages, "create", refuse)
    res = client.post("/api/translate", data={"document_text": SAMPLE})
    assert res.status_code == 502
    assert "declined" in res.json()["detail"]


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT_PER_HOUR", 2)
    main._recent_requests.clear()
    codes = [client.post("/api/translate", data={"document_text": SAMPLE}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    main._recent_requests.clear()


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["api_key_configured"] is True
    assert body["medlineplus_topics"] > 0


def test_samples_list_and_fetch(client):
    samples = client.get("/api/samples").json()
    ids = [s["id"] for s in samples]
    assert "sample_lab_report" in ids
    one = client.get("/api/samples/sample_lab_report").json()
    assert "Potassium" in one["text"]
    assert client.get("/api/samples/does_not_exist").status_code == 404
    # The handler itself rejects anything that isn't a plain sample name.
    with pytest.raises(main.HTTPException) as err:
        main.get_sample("../backend/.env")
    assert err.value.status_code == 404


def test_ocr_reads_text_pdf(client):
    from io import BytesIO

    pytest.importorskip("pypdf")
    # A minimal one-page PDF with a text layer.
    pdf = _make_text_pdf("Potassium 5.8 mmol/L (3.5-5.0) Sodium 138 mmol/L")
    res = client.post("/api/ocr", files={"file": ("report.pdf", BytesIO(pdf), "application/pdf")})
    assert res.status_code == 200, res.text
    assert res.json()["engine"] == "pdf-text"
    assert "Potassium" in res.json()["text"]


def test_ocr_image_uses_claude_when_tesseract_missing(client, monkeypatch):
    from io import BytesIO

    from PIL import Image

    monkeypatch.setattr("ocr.shutil.which", lambda name: None)
    buffer = BytesIO()
    Image.new("RGB", (400, 200), "white").save(buffer, format="PNG")
    res = client.post("/api/ocr", files={"file": ("photo.png", BytesIO(buffer.getvalue()), "image/png")})
    assert res.status_code == 200, res.text
    assert res.json()["engine"] == "claude"
    image_call = client.fake.calls[-1]
    assert image_call["messages"][0]["content"][0]["type"] == "image"


def test_ocr_rejects_non_images(client):
    from io import BytesIO

    res = client.post("/api/ocr", files={"file": ("notes.txt", BytesIO(b"hello world, not an image"), "text/plain")})
    assert res.status_code == 400


def test_oversized_request_is_rejected_before_parsing(client):
    big = b"x" * (main.MAX_REQUEST_BYTES + 1)
    res = client.post("/api/ocr", files={"file": ("big.png", big, "image/png")})
    assert res.status_code == 413


def test_decompression_bomb_image_is_rejected(client, monkeypatch):
    from io import BytesIO

    from PIL import Image

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)  # anything over 2,000 px is a "bomb" now
    buffer = BytesIO()
    Image.new("L", (100, 100), 255).save(buffer, format="PNG")
    res = client.post("/api/ocr", files={"file": ("bomb.png", BytesIO(buffer.getvalue()), "image/png")})
    assert res.status_code == 400
    assert "too large" in res.json()["detail"]


def test_busy_server_says_so_instead_of_queueing(client, monkeypatch):
    import threading

    monkeypatch.setattr(main, "_job_slots", threading.BoundedSemaphore(1))
    main._job_slots.acquire()  # someone else is using the only slot
    try:
        res = client.post("/api/translate", data={"document_text": SAMPLE})
    finally:
        main._job_slots.release()
    assert res.status_code == 503
    assert "busy" in res.json()["detail"]


def test_rate_limiter_forgets_idle_clients(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(main, "RATE_LIMIT_PER_HOUR", 5)
    main._recent_requests.clear()
    clock = [1000.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: clock[0])
    for i in range(50):
        main.check_rate_limit(SimpleNamespace(client=SimpleNamespace(host=f"10.0.0.{i}")))
    assert len(main._recent_requests) == 50
    clock[0] += main.RATE_WINDOW_SECONDS + 1
    main.check_rate_limit(SimpleNamespace(client=SimpleNamespace(host="10.0.1.1")))
    assert list(main._recent_requests) == ["10.0.1.1"]
    main._recent_requests.clear()


def test_readability_treats_headers_and_bullets_as_sentences():
    formatted = "## Lab results\n\n- Potassium: high\n- Sugar: a bit high\n- Salt: normal"
    prose = "Lab results. Potassium: high. Sugar: a bit high. Salt: normal."
    assert pipeline.grade_level(formatted) == pipeline.grade_level(prose)


def test_cited_reference_ranges():
    assert pipeline.cited_reference_ids("See [1-3] and [5, 7].", 6) == {1, 2, 3, 5}


def test_frontend_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Clearchart" in res.text


def _make_text_pdf(text: str) -> bytes:
    """Build a tiny valid PDF by hand (avoids a PDF-writing dependency)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)
