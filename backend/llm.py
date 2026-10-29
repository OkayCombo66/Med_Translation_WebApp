"""
One place for every Claude API call the app makes.

Keeping this in a single module means the model name, effort level, refusal
handling and error types are decided once, and tests can swap in a fake
client without touching the rest of the code.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import anthropic
from anthropic import Anthropic

# The current Sonnet. Override with ANTHROPIC_MODEL in backend/.env if you want
# a different model (e.g. claude-opus-5-5 for higher quality at higher cost).
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

# Thinking tokens count toward this cap too, so don't set it too low.
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "6000"))

# The SDK default waits up to 10 minutes per call and retries twice; a web
# request shouldn't hang that long, so fail sooner and retry once.
API_TIMEOUT_SECONDS = float(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", "150"))
API_MAX_RETRIES = int(os.getenv("ANTHROPIC_MAX_RETRIES", "1"))

# Server-side refusal fallback: if a safety classifier declines a request, the
# API re-runs it on Anthropic's recommended fallback model inside the same call.
# Only some models accept the parameter, so it is enabled just for those.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1"}


class LLMError(Exception):
    """Raised when Claude can't produce a usable answer (refusal, cut-off, bad JSON)."""


@dataclass
class LLMResult:
    text: str
    data: Optional[dict] = None          # parsed JSON when a schema was requested
    usage: dict = field(default_factory=dict)
    model: str = MODEL             # the model that actually wrote the reply
    fallback_used: bool = False    # True when a server-side fallback model served it


_client: Optional[Anthropic] = None


def get_client() -> Anthropic:
    """Create the API client lazily so the app can start (and show a helpful
    message) even before an API key is configured."""
    global _client
    if _client is None:
        # Reads ANTHROPIC_API_KEY from the environment.
        _client = Anthropic(
            timeout=anthropic.Timeout(API_TIMEOUT_SECONDS, connect=5.0),
            max_retries=API_MAX_RETRIES,
        )
    return _client


def set_client(client) -> None:
    """Used by tests to inject a fake client."""
    global _client
    _client = client


def api_key_configured() -> bool:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    return bool(key) and key != "your_key_here"


def ask_claude(
    system: str,
    content,
    *,
    schema: Optional[dict] = None,
    effort: str = "low",
    max_tokens: Optional[int] = None,
) -> LLMResult:
    """
    Send one request and return the text (and parsed JSON if `schema` is given).

    `content` is either a string or a list of content blocks (e.g. an image
    block followed by a text block). `schema` turns on structured outputs, which
    guarantees the reply is valid JSON matching the schema.
    """
    output_config = {"effort": effort}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}

    request = dict(
        model=MODEL,
        max_tokens=max_tokens or MAX_OUTPUT_TOKENS,
        # The system prompts never change between requests, so cache them.
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
        output_config=output_config,
    )

    client = get_client()
    if MODEL in FALLBACK_MODELS:
        response = client.beta.messages.create(
            betas=[FALLBACK_BETA], fallbacks="default", **request
        )
    else:
        response = client.messages.create(**request)

    # Check why the model stopped before trusting the content.
    if response.stop_reason == "refusal":
        raise LLMError("The model declined to process this document.")
    if response.stop_reason == "max_tokens":
        raise LLMError(
            "The response was cut off. Raise MAX_OUTPUT_TOKENS in backend/.env "
            "or try a shorter document."
        )

    # Responses can include thinking blocks, so only read the text blocks.
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    usage = {
        "input_tokens": getattr(response.usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(response.usage, "output_tokens", 0) or 0,
    }

    # If a fallback model answered, say so instead of claiming MODEL wrote it.
    iterations = getattr(response.usage, "iterations", None) or []
    fallback_used = any(getattr(step, "type", None) == "fallback_message" for step in iterations)
    served_by = getattr(response, "model", None) or MODEL

    data = None
    if schema is not None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"The model returned malformed JSON: {e}") from e

    return LLMResult(text=text, data=data, usage=usage, model=served_by, fallback_used=fallback_used)
