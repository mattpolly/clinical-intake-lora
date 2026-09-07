"""Server-side Baseten client: wake, readiness, inference.

The Baseten API key is read from the environment/``.env`` by the caller and
held only in this process. It is sent as an ``Authorization: Bearer`` header
and never returned, logged, or written to artifacts:

- ``BasetenClient.__repr__`` redacts the key,
- no client method returns the key or includes it in a result,
- error messages are sanitized before they reach the instrumentation layer.

The documented interfaces implemented here (with their exact verbs, paths and
success codes) are cited in ``BASETEN_DOCS.md``:

- wake:        ``POST /deployment/{deployment_id}/wake``  -> 202 Accepted
- readiness:   ``GET  /v1/models/{model_id}/deployments/{deployment_id}``
               -> ``status == "ACTIVE"`` and ``active_replica_count >= 1``
- inference:   OpenAI-compatible chat completions over
               ``/environments/{environment}/sync/v1/chat/completions``

The transport is injectable so tests run against mocks of these interfaces
with no live Baseten calls.
"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from .deployment import (
    chat_completions_url,
    status_url,
    wake_url,
)

# Documented deployment status values (SDK DeploymentStatus / lifecycle docs).
STATUS_ACTIVE = "ACTIVE"
STATUS_SCALED_TO_ZERO = "SCALED_TO_ZERO"
STATUS_WAKING_UP = "WAKING_UP"
STATUS_LOADING_MODEL = "LOADING_MODEL"

# HTTP statuses that are transient and safe to retry exactly once (bounded).
TRANSIENT_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504, 529})


class BasetenError(Exception):
    """Base error for the Baseten client."""


class AuthError(BasetenError):
    """401/403: bad or missing credentials. Never retried."""


class RequestError(BasetenError):
    """Malformed request (400/404). Never retried."""


class TransientError(BasetenError):
    """Transient platform/capacity error. Retryable, bounded at most once."""


class DeploymentTimeoutError(BasetenError):
    """Readiness deadline expired."""


def _safe_detail(status_code: int, body: str, limit: int = 300) -> str:
    """Bounded, secret-free description of an HTTP failure."""
    text = (body or "").strip().replace("\r", " ").replace("\n", " ")
    return f"HTTP {status_code}: {text[:limit]}"


class HttpResponse(Protocol):
    status_code: int

    def json(self) -> Any: ...

    @property
    def text(self) -> str: ...

    def iter_lines(self) -> Any: ...


class HttpTransport(Protocol):
    def post(
        self, url: str, *, headers: dict[str, str], json: Any, stream: bool = False
    ) -> HttpResponse: ...

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse: ...


class RequestsTransport:
    """Thin adapter over ``requests`` (the only live HTTP path)."""

    def __init__(self) -> None:
        import requests  # local import; tests never need a live client

        self._session = requests.Session()

    def post(
        self, url: str, *, headers: dict[str, str], json: Any, stream: bool = False
    ) -> HttpResponse:
        return self._session.post(url, headers=headers, json=json, stream=stream, timeout=1200)

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
        return self._session.get(url, headers=headers, timeout=60)


class BasetenClient:
    """Synchronous, server-side Baseten client."""

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str,
        deployment_id: str,
        environment: str = "production",
        served_model_name: str | None = None,
        transport: HttpTransport | None = None,
    ):
        if not api_key:
            raise AuthError("BASETEN_API_KEY is not set")
        if not model_id or not deployment_id:
            raise RequestError(
                "both model_id and deployment_id are required for the "
                "Baseten lifetime endpoints"
            )
        self._api_key = api_key
        self.model_id = model_id
        self.deployment_id = deployment_id
        self.environment = environment
        self.served_model_name = served_model_name
        self._transport = transport if transport is not None else RequestsTransport()

    def __repr__(self) -> str:  # never leak the key
        return (
            f"BasetenClient(model_id={self.model_id!r}, "
            f"deployment_id={self.deployment_id!r}, api_key=<redacted>)"
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _raise_for_status(self, method: str, url: str, response: HttpResponse) -> None:
        code = response.status_code
        if code in (401, 403):
            raise AuthError(_safe_detail(code, response.text))
        if code in TRANSIENT_HTTP_STATUSES:
            raise TransientError(_safe_detail(code, response.text))
        if code >= 400:
            raise RequestError(_safe_detail(code, response.text))

    def wake(self) -> None:
        """Explicit wake (scale-from-zero). Returns on 202 Accepted."""
        url = wake_url(self.model_id, self.deployment_id)
        response = self._transport.post(url, headers=self._headers(), json=None)
        if response.status_code != 202:
            self._raise_for_status("wake", url, response)

    def deployment_status(self) -> dict[str, Any]:
        """Readiness/health signal: deployment status + active replica count."""
        url = status_url(self.model_id, self.deployment_id)
        response = self._transport.get(url, headers=self._headers())
        if response.status_code != 200:
            self._raise_for_status("status", url, response)
        try:
            data = response.json()
        except (ValueError, AttributeError) as exc:
            raise TransientError(f"non-JSON status response: {exc}") from exc
        if not isinstance(data, dict):
            raise TransientError("status response is not a JSON object")
        return {
            "status": data.get("status"),
            "active_replica_count": data.get("active_replica_count", 0),
        }

    def is_ready(self) -> bool:
        """Documented readiness/health signal (status == ACTIVE, replica >= 1)."""
        status = self.deployment_status()
        return (
            status.get("status") == STATUS_ACTIVE
            and int(status.get("active_replica_count") or 0) >= 1
        )

    @staticmethod
    def _parse_chat_completion_line(
        line: str, text: str, done: bool
    ) -> tuple[str, bool]:
        chunk_text = ""
        if not line or not line.startswith("data:"):
            return text, done
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            return text, True
        try:
            data = json.loads(payload)
        except ValueError:
            return text, done
        choice = (data.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content:
            chunk_text = content
        return text + chunk_text, done

    def infer(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 128,
        temperature: float = 0.2,
        stream: bool = True,
    ) -> dict[str, Any]:
        """OpenAI-compatible chat completion, returning text + timing.

        Time-to-first-token is the time from sending the request until the
        first non-empty content delta (streaming), and duration is the full
        request time. Callers receive ``status`` one of ``ok``/``error`` plus
        ``time_to_first_token_ms`` and ``duration_ms``.
        """
        url = chat_completions_url(self.model_id, self.environment)
        body: dict[str, Any] = {
            "model": self.served_model_name or self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        start = time.perf_counter()
        ttfb: float | None = None
        text = ""
        done = False
        if stream:
            response = self._transport.post(
                url, headers=self._headers(), json=body, stream=True
            )
            if response.status_code != 200:
                self._raise_for_status("infer", url, response)
            for line in response.iter_lines():
                decoded = line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else line
                before = text
                text, done = self._parse_chat_completion_line(str(decoded), text, done)
                if text != before and ttfb is None:
                    ttfb = (time.perf_counter() - start) * 1000
                if done:
                    break
            duration_ms = (time.perf_counter() - start) * 1000
        else:
            response = self._transport.post(
                url, headers=self._headers(), json=body, stream=False
            )
            duration_ms = (time.perf_counter() - start) * 1000
            if response.status_code != 200:
                self._raise_for_status("infer", url, response)
            try:
                data = response.json()
            except (ValueError, AttributeError) as exc:
                raise TransientError(f"non-JSON inference response: {exc}") from exc
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            text = message.get("content") or ""
            ttfb = duration_ms  # non-streaming: no earlier first-token sample

        return {
            "status": "ok",
            "text": text,
            "time_to_first_token_ms": round(ttfb, 1) if ttfb is not None else None,
            "duration_ms": round(duration_ms, 1),
        }
