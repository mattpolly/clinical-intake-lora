"""Mock Baseten client for dry-run and CI (no live Baseten calls).

Simulates the documented lifecycle state machine
(``SCALED_TO_ZERO -> WAKING_UP -> ACTIVE``) with scripted inference, so the
full wake -> readiness -> infer -> idle path is testable without credentials
or network access. It reproduces the *shape* of the documented interfaces
(wake returns 202-equivalent; readiness is ``status == ACTIVE`` with an active
replica; inference is an OpenAI-compatible chat completion with a
time-to-first-token and duration).
"""

from __future__ import annotations

import time
from typing import Any

from .client import (
    STATUS_ACTIVE,
    STATUS_SCALED_TO_ZERO,
    STATUS_WAKING_UP,
)

MOCK_REPLY = "How long does each headache typically last?"


class MockBasetenClient:
    """Deterministic simulation of the Baseten lifetime endpoints."""

    def __init__(
        self,
        model_id: str = "mock-model-id",
        deployment_id: str = "mock-deployment-id",
        *,
        ready_after_seconds: float = 0.0,
        probe_reply: str = "ok",
        infer_reply: str = MOCK_REPLY,
    ):
        self.model_id = model_id
        self.deployment_id = deployment_id
        self.served_model_name = "Qwen/Qwen3-8B"
        self.ready_after_seconds = ready_after_seconds
        self.probe_reply = probe_reply
        self.infer_reply = infer_reply

        self._status = STATUS_SCALED_TO_ZERO
        self._active_replica_count = 0
        self._wake_called = False
        self._wake_at: float | None = None
        self._scaled_back_to_zero = False

    # -- lifecycle simulation ------------------------------------------------

    def _advance(self) -> None:
        if not self._wake_called:
            self._status = STATUS_SCALED_TO_ZERO
            self._active_replica_count = 0
            return
        elapsed = time.monotonic() - (self._wake_at or time.monotonic())
        if self._scaled_back_to_zero:
            self._status = STATUS_SCALED_TO_ZERO
            self._active_replica_count = 0
        elif elapsed < self.ready_after_seconds:
            self._status = STATUS_WAKING_UP
            self._active_replica_count = 0
        else:
            self._status = STATUS_ACTIVE
            self._active_replica_count = 1

    def wake(self) -> None:
        self._wake_called = True
        self._scaled_back_to_zero = False
        self._wake_at = time.monotonic()
        self._advance()

    def deployment_status(self) -> dict[str, Any]:
        self._advance()
        return {
            "status": self._status,
            "active_replica_count": self._active_replica_count,
        }

    def is_ready(self) -> bool:
        status = self.deployment_status()
        return (
            status.get("status") == STATUS_ACTIVE
            and int(status.get("active_replica_count") or 0) >= 1
        )

    def mark_idle_and_scale_to_zero(self) -> None:
        """Simulate the idle interval elapsing; the next status poll reports
        ``SCALED_TO_ZERO``."""
        self._scaled_back_to_zero = True
        self._advance()

    def infer(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 128,
        temperature: float = 0.2,
        stream: bool = True,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        # Small deterministic sleep so ttfb < duration in the simulated trace.
        time.sleep(0.001)
        ttfb = (time.perf_counter() - start) * 1000
        time.sleep(0.001)
        duration = (time.perf_counter() - start) * 1000
        is_probe = max_tokens == 1
        text = self.probe_reply if is_probe else self.infer_reply
        return {
            "status": "ok",
            "text": text,
            "time_to_first_token_ms": round(ttfb, 1),
            "duration_ms": round(duration, 1),
        }
