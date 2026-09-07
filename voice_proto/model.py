"""Pluggable intake-model backends (feature Decision 4).

Contract: turn(conversation) -> {"text", "latency_ms", "ttfb_ms"} where
conversation is [{"role": "patient"|"intake", "content": str}]. One-shot,
non-streaming, plain-text reply (a single concise question or summary).
The mock is deterministic; the Baseten backend is OpenAI-compatible and used
only when BASETEN_API_KEY and BASETEN_MODEL_URL are set. Research only —
non-diagnostic.
"""

from __future__ import annotations

import abc
import os
import time

import requests

SYSTEM_PROMPT = (
    "You are a research-only clinical intake interviewer prototype. Ask one "
    "concise question at a time, or produce a faithful structured summary. "
    "Never diagnose, never recommend treatment, and never invent patient "
    "facts that were not stated."
)


class ModelBackend(abc.ABC):
    name = "abstract"

    @abc.abstractmethod
    def turn(self, conversation: list[dict]) -> dict:
        """Return {"text", "latency_ms", "ttfb_ms", "status"}."""


class MockModel(ModelBackend):
    """Deterministic scripted intake behavior; one concise question."""

    name = "mock"

    def turn(self, conversation: list[dict]) -> dict:
        start = time.perf_counter()
        text = "How long does each headache typically last?"
        return {
            "text": text,
            "status": "ok",
            "ttfb_ms": 0.0,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }


class BasetenModel(ModelBackend):
    """OpenAI-compatible chat request to a Baseten-deployed Qwen endpoint."""

    name = "baseten"
    TIMEOUT_S = 60

    def __init__(self, api_key: str, model_url: str, model_id: str | None = None):
        self.api_key = api_key
        self.model_url = model_url.rstrip("/")
        if not self.model_url.endswith("/v1/chat/completions"):
            self.model_url += "/v1/chat/completions"
        self.model_id = model_id or os.environ.get("BASETEN_MODEL_ID")

    def turn(self, conversation: list[dict]) -> dict:
        role_map = {"patient": "user", "intake": "assistant"}
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            {"role": role_map.get(m["role"], "user"), "content": m["content"]}
            for m in conversation
        ]
        body = {
            "messages": messages,
            "max_tokens": 128,
            "temperature": 0.2,
        }
        if self.model_id:
            body["model"] = self.model_id
        start = time.perf_counter()
        try:
            resp = requests.post(
                self.model_url,
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.TIMEOUT_S,
                stream=True,
            )
            ttfb = round((time.perf_counter() - start) * 1000, 1)
            chunks = []
            for chunk in resp.iter_content(chunk_size=8192):
                chunks.append(chunk)
            payload = b"".join(chunks).decode("utf-8", "replace")
            latency = round((time.perf_counter() - start) * 1000, 1)
            if resp.status_code != 200:
                return {
                    "text": None, "status": "error",
                    "error": f"HTTP {resp.status_code}",
                    "ttfb_ms": ttfb, "latency_ms": latency,
                }
            data = resp.json() if payload.startswith("{") else None
            text = None
            if data:
                try:
                    text = data["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError, TypeError):
                    pass
            if not text:
                return {
                    "text": None, "status": "unexpected-format",
                    "error": "no chat-completion text in response",
                    "ttfb_ms": ttfb, "latency_ms": latency,
                }
            return {"text": text, "status": "ok",
                    "ttfb_ms": ttfb, "latency_ms": latency}
        except requests.RequestException as exc:
            return {
                "text": None, "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "ttfb_ms": None, "latency_ms": None,
            }


def select_backend(preferred: str = "auto") -> ModelBackend | None:
    """Return the configured backend, or None if the requested one is not
    credentialed (caller records a missing-credential marker)."""
    api_key = os.environ.get("BASETEN_API_KEY")
    model_url = os.environ.get("BASETEN_MODEL_URL")
    baseten_ready = bool(api_key and model_url)
    if preferred == "baseten" or (preferred == "auto" and baseten_ready):
        if baseten_ready:
            return BasetenModel(api_key, model_url)
        return None
    return MockModel()
