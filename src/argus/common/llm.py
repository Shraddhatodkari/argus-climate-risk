"""Pluggable LLM backend.

Three real implementations behind one interface:

- TemplateLLM   — deterministic, dependency-free, zero network. The default. Every
                   number and citation in its output is assembled directly from the
                   structured inputs it's given, so it cannot hallucinate a figure —
                   this is the baseline the reviewer gate compares generative output
                   against, and what CI runs so tests never depend on a live model.
- OllamaLLM     — real HTTP call to a local Ollama server. Zero-cost, offline, the
                   documented default for day-to-day use once a model is pulled.
- ClaudeLLM     — real Anthropic SDK call. Optional, documented enterprise-grade path;
                   applies the "Building with the Claude API" certification directly.

Certification applied: Anthropic — Building with the Claude API;
NVIDIA — Building LLM Applications With Prompt Engineering (prompt construction below).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from argus.common.config import SETTINGS

SYSTEM_PROMPT = (
    "You are a financial risk disclosure drafting assistant. You must ONLY use the "
    "numeric figures and regulatory citations provided to you in the prompt. Never "
    "invent a figure, a citation, or a regulatory clause that was not given to you. "
    "If asked something the provided context does not cover, say so explicitly instead "
    "of guessing."
)


@dataclass
class LLMResponse:
    text: str
    backend: str


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse: ...


class TemplateLLM(LLMClient):
    """Deterministic fallback: fills a fixed narrative template from structured facts
    passed in the prompt's ``context`` dict. Used whenever no live model is configured,
    and always used in tests, so correctness never depends on model availability."""

    def __init__(self, context: dict | None = None) -> None:
        self.context = context or {}

    def generate(self, prompt: str) -> LLMResponse:
        c = self.context
        citation = c.get("citation")
        citation_line = (
            f" Per [{citation['doc_id']} §{citation['section']}], {citation['text']}"
            if citation
            else " No specific regulatory citation was retrieved for this district; "
            "this paragraph should be reviewed before any regulatory claim is added."
        )
        text = (
            f"{c.get('district', 'The district')} shows a composite climate physical-hazard "
            f"score of {c.get('score', 0):.1f}/100 "
            f"(flood: {c.get('flood', 0):.1f}, drought: {c.get('drought', 0):.1f}, "
            f"cyclone: {c.get('cyclone', 0):.1f}, heat: {c.get('heat', 0):.1f}); "
            f"the primary hazard is {c.get('primary_hazard', 'n/a')} and the most climate-exposed "
            f"sector is {c.get('most_affected_sector', 'n/a')}. Estimated climate-exposed public-data "
            f"exposure: ₹{c.get('exposure', 0):,.0f} crore (a public-data estimate, not a specific "
            f"institution's portfolio) "
            f"({'HIGH RISK — flagged for review' if c.get('is_high_risk') else 'within tolerance'})."
            f"{citation_line}"
        )
        return LLMResponse(text=text, backend="template")


class OllamaLLM(LLMClient):
    """Real HTTP call to a local Ollama server. Not exercised by default tests — skipped
    automatically when no server is reachable (see tests/integration/test_llm_backends.py)."""

    def __init__(self) -> None:
        self.host = SETTINGS.ollama_host
        self.model = SETTINGS.ollama_model

    def generate(self, prompt: str) -> LLMResponse:
        import requests

        resp = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "system": SYSTEM_PROMPT, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return LLMResponse(text=resp.json().get("response", ""), backend=f"ollama:{self.model}")


class ClaudeLLM(LLMClient):
    """Real Anthropic API call. Requires ANTHROPIC_API_KEY. Optional — never required
    to run the pipeline end to end."""

    def __init__(self) -> None:
        if not SETTINGS.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set — cannot use ClaudeLLM backend.")
        import anthropic

        self._client = anthropic.Anthropic(api_key=SETTINGS.anthropic_api_key)
        self.model = SETTINGS.anthropic_model

    def generate(self, prompt: str) -> LLMResponse:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in msg.content if hasattr(block, "text"))
        return LLMResponse(text=text, backend=f"claude:{self.model}")


def get_llm_client(mode: str | None = None, *, context: dict | None = None) -> LLMClient:
    """Factory. Defaults to ARGUS_LLM_MODE (env), which defaults to 'template'."""
    mode = (mode or SETTINGS.llm_mode).lower()
    if mode == "template":
        return TemplateLLM(context=context)
    if mode == "ollama":
        return OllamaLLM()
    if mode == "claude":
        return ClaudeLLM()
    raise ValueError(f"Unknown ARGUS_LLM_MODE: {mode!r} (expected template|ollama|claude)")
