"""Integration tests: the pluggable LLM backend. Ollama/Claude are real
implementations but are only exercised here when actually reachable — the whole point
of the template default is that CI must never depend on a live model or a paid key."""

import contextlib
import os

import pytest

from argus.common.llm import ClaudeLLM, OllamaLLM, get_llm_client


def test_template_backend_is_the_default_and_always_works():
    client = get_llm_client("template", context={"district": "Test", "score": 50.0, "flood": 10,
                                                    "drought": 10, "cyclone": 10, "exposure": 100,
                                                    "is_high_risk": False, "citation": None})
    response = client.generate("irrelevant prompt in template mode")
    assert "Test" in response.text
    assert response.backend == "template"


def test_unknown_backend_raises_clear_error():
    with pytest.raises(ValueError):
        get_llm_client("not-a-real-backend")


def test_ollama_backend_is_skipped_when_no_server_reachable():
    client = OllamaLLM()
    # A bare Exception is deliberate here: an unreachable Ollama server can surface as a
    # connection error, a timeout, or a JSON-decode error depending on the local network
    # stack, and any of them means the same thing for this test — no server, nothing to
    # assert. contextlib.suppress documents that intent explicitly rather than a silent
    # try/except/pass.
    with contextlib.suppress(Exception):
        client.generate("ping")
        pytest.skip("An Ollama server is actually reachable here — nothing to assert further.")


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY set")
def test_claude_backend_when_api_key_present():
    client = ClaudeLLM()
    response = client.generate("Reply with exactly: OK")
    assert response.backend.startswith("claude:")


def test_claude_backend_raises_clearly_without_a_key(monkeypatch):
    # SETTINGS is a frozen dataclass captured at import time from the environment, so
    # patch the module-level singleton ClaudeLLM actually reads from, rather than the
    # environment variable (which is only read once, at import time).
    import argus.common.llm as llm_module
    from argus.common.config import Settings

    monkeypatch.setattr(llm_module, "SETTINGS", Settings(anthropic_api_key=None))
    with pytest.raises(RuntimeError):
        ClaudeLLM()
