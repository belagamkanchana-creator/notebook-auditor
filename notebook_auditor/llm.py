"""Thin provider-agnostic LLM client (Anthropic, OpenAI, LiteLLM) returning parsed JSON."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, List, Optional

log = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4o-mini",
    "litellm": "anthropic/claude-sonnet-5-5",
}


def parse_json_response(text: str) -> Any:
    """Parse the first JSON object/array in a model response (tolerates code fences and prose)."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


class LLMClient:
    provider = "base"

    # After this many consecutive failed calls the API is treated as unavailable for the rest of the run.
    max_consecutive_failures = 2

    def __init__(self, model: str) -> None:
        self.model = model
        self.errors: List[str] = []
        self.calls = 0
        self.successes = 0
        self.consecutive_failures = 0
        self.disabled = False

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def complete_json(self, system: str, user: str, max_tokens: int = 2000) -> Any:
        if self.disabled:
            return None
        self.calls += 1
        try:
            text = self.complete(system + "\nRespond with a single JSON object and nothing else.", user, max_tokens)
        except Exception as exc:  # network / auth / rate limit: degrade to heuristics
            msg = f"{self.provider} call failed: {type(exc).__name__}: {exc}"
            log.warning(msg)
            self.errors.append(msg)
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.disabled = True
                self.errors.append(
                    f"{self.provider} disabled after {self.consecutive_failures} consecutive failures; "
                    "remaining steps used heuristics only."
                )
            return None
        self.consecutive_failures = 0
        self.successes += 1
        data = parse_json_response(text)
        if data is None:
            self.errors.append(f"{self.provider} returned non-JSON output")
        return data

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def mode_label(self) -> str:
        """How the run actually went, for the report header."""
        if self.calls and not self.successes:
            return f"heuristic (LLM {self.label} unavailable: all {self.calls} calls failed)"
        if self.errors:
            return f"llm:{self.label} (partial: {self.successes}/{self.calls} calls succeeded)"
        return f"llm:{self.label}"


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: Optional[str] = None) -> None:
        super().__init__(model or DEFAULT_MODELS["anthropic"])
        import anthropic

        self._client = anthropic.Anthropic()

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")


class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, model: Optional[str] = None) -> None:
        super().__init__(model or DEFAULT_MODELS["openai"])
        import openai

        self._client = openai.OpenAI()

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


class LiteLLMClient(LLMClient):
    provider = "litellm"

    def __init__(self, model: Optional[str] = None) -> None:
        super().__init__(model or DEFAULT_MODELS["litellm"])
        import litellm

        self._litellm = litellm

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        resp = self._litellm.completion(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


def make_client(provider: str = "auto", model: Optional[str] = None) -> Optional[LLMClient]:
    """Return an LLM client, or None for heuristic-only mode.

    ``auto`` picks Anthropic if ANTHROPIC_API_KEY is set, else OpenAI if OPENAI_API_KEY is set, else none.
    """
    provider = (provider or "auto").lower()
    if provider in ("none", "off", "heuristic"):
        return None
    if provider == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            return None
    classes = {"anthropic": AnthropicClient, "openai": OpenAIClient, "litellm": LiteLLMClient}
    if provider not in classes:
        raise ValueError(f"Unknown provider {provider!r}; choose from auto, anthropic, openai, litellm, none")
    try:
        return classes[provider](model)
    except ImportError as exc:
        raise RuntimeError(f"Provider {provider!r} needs its SDK installed: pip install 'notebook-auditor[{provider}]'") from exc
