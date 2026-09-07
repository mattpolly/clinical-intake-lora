"""Wake -> readiness -> inference orchestration with bounded failure behavior.

- Explicit wake call, timestamped as ``wake_requested`` before any polling.
- Readiness = documented health signal (``status == ACTIVE``) **plus** one
  trivial inference probe (the probe doubles as the first inference).
- Readiness polling is bounded: it stops in success or an explicit timeout
  error once a configurable deadline (default 600s) elapses; it never polls
  indefinitely.
- At most one bounded retry on transient wake/inference failures.
- Every failure surfaces as an explicit error state and an ``error`` event.

The same code path serves both live and mock (dry-run) transports: the client
is injected, so CI runs entirely against mocks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .client import BasetenClient, BasetenError, DeploymentTimeoutError, TransientError
from .instrumentation import EventWriter

PROBE_MESSAGES = [
    {"role": "system", "content": "You are a research-only prototype."},
    {"role": "user", "content": "Reply with exactly one word."},
]

DEFAULT_READINESS_DEADLINE_SECONDS = 600
DEFAULT_POLL_INTERVAL_SECONDS = 5.0


@dataclass
class WakeReadyInferResult:
    ok: bool
    state: str  # "ready" | "wake_error" | "readiness_timeout" | "probe_error"
    detail: str
    wake_requested_at: str
    first_inference: dict[str, Any] | None = None
    replica_ready: dict[str, Any] | None = None


@dataclass
class HarnessConfig:
    readiness_deadline_seconds: float = DEFAULT_READINESS_DEADLINE_SECONDS
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    probe_messages: list[dict[str, str]] = field(
        default_factory=lambda: list(PROBE_MESSAGES)
    )
    probe_max_tokens: int = 1
    inference_max_tokens: int = 128
    inference_messages: list[dict[str, str]] = field(
        default_factory=lambda: [
            {"role": "system", "content": "You are a research-only prototype."},
            {"role": "user", "content": "Ask one concise intake question."},
        ]
    )
    sleep: Callable[[float], None] = time.sleep


def _retry_once(client_call: Callable[[], Any]) -> Any:
    """Run *client_call*, retrying at most once on a transient failure."""
    try:
        return client_call()
    except TransientError:
        return client_call()


class Harness:
    """Runs the demo lifecycle and writes every required event."""

    def __init__(
        self,
        client: BasetenClient,
        writer: EventWriter,
        *,
        config: HarnessConfig | None = None,
        profile_name: str | None = None,
    ):
        self.client = client
        self.writer = writer
        self.config = config or HarnessConfig()
        self.profile_name = profile_name

    def _emit_error(self, stage: str, error_type: str, message: str) -> dict:
        return self.writer.emit(
            "error",
            stage=stage,
            error_type=error_type,
            message=message,
            profile=self.profile_name,
        )

    def wake(self) -> dict[str, Any]:
        """Explicit wake; records ``wake_requested`` first. Bounded retry."""
        requested = self.writer.emit(
            "wake_requested",
            profile=self.profile_name,
            model_id=self.client.model_id,
            deployment_id=self.client.deployment_id,
        )
        try:
            _retry_once(self.client.wake)
        except BasetenError as exc:
            self._emit_error("wake", type(exc).__name__, str(exc))
            raise
        return requested

    def _poll_ready(self, deadline: float) -> bool:
        """Poll the documented health signal until ready or deadline."""
        while True:
            if self.client.is_ready():
                return True
            if time.monotonic() >= deadline:
                return False
            self.config.sleep(self.config.poll_interval_seconds)

    def _infer(self, messages, max_tokens) -> dict[str, Any]:
        try:
            return _retry_once(
                lambda: self.client.infer(messages, max_tokens=max_tokens, stream=True)
            )
        except BasetenError as exc:
            self._emit_error("inference", type(exc).__name__, str(exc))
            raise

    def _record_inference(self, result: dict[str, Any], *, first: bool) -> None:
        if result.get("status") != "ok":
            message = f"inference returned status {result.get('status')!r}"
            self._emit_error("inference", "InferenceError", message)
            raise DeploymentTimeoutError(message)
        if first:
            self.writer.emit(
                "first_inference",
                profile=self.profile_name,
                time_to_first_token_ms=result.get("time_to_first_token_ms"),
                duration_ms=result.get("duration_ms"),
            )
        self.writer.emit(
            "inference_result",
            profile=self.profile_name,
            duration_ms=result.get("duration_ms"),
            time_to_first_token_ms=result.get("time_to_first_token_ms"),
            status=result.get("status"),
        )
        self.writer.emit("last_activity", profile=self.profile_name)

    def probe(self, *, first: bool) -> dict[str, Any]:
        """Trivial inference probe (doubles as first inference)."""
        result = self._infer(self.config.probe_messages, self.config.probe_max_tokens)
        self._record_inference(result, first=first)
        return result

    def infer_turn(self) -> dict[str, Any]:
        """One full demo inference turn (emits inference_result + last_activity)."""
        result = self._infer(
            self.config.inference_messages, self.config.inference_max_tokens
        )
        self._record_inference(result, first=False)
        return result

    def wake_ready_probe(self) -> WakeReadyInferResult:
        """Full wake -> readiness (health + probe) -> first-token flow."""
        wake_event = self.wake()

        deadline = time.monotonic() + self.config.readiness_deadline_seconds
        if not self._poll_ready(deadline):
            message = (
                f"replica not ready within {self.config.readiness_deadline_seconds:g}s "
                "deadline"
            )
            self._emit_error("readiness", "ReadinessTimeout", message)
            return WakeReadyInferResult(
                ok=False,
                state="readiness_timeout",
                detail=message,
                wake_requested_at=wake_event["timestamp"],
            )

        # Health signal satisfied; the trivial probe proves readiness and is
        # recorded as the first inference request.
        try:
            probe_result = self.probe(first=True)
        except BasetenError as exc:
            return WakeReadyInferResult(
                ok=False,
                state="probe_error",
                detail=str(exc),
                wake_requested_at=wake_event["timestamp"],
            )

        ready_event = self.writer.emit(
            "replica_ready",
            profile=self.profile_name,
            status=self.client.deployment_status().get("status"),
        )
        return WakeReadyInferResult(
            ok=True,
            state="ready",
            detail=probe_result.get("text", ""),
            wake_requested_at=wake_event["timestamp"],
            first_inference={
                "time_to_first_token_ms": probe_result.get("time_to_first_token_ms"),
                "duration_ms": probe_result.get("duration_ms"),
            },
            replica_ready=ready_event,
        )

    def observe_scale_to_zero(self, *, deadline_seconds: float) -> dict[str, Any] | None:
        """Best-effort observation of scale-down (``scaled_to_zero``).

        Polls the documented status until ``status == SCALED_TO_ZERO`` (or
        ``active_replica_count == 0``). Returns the event, or ``None`` when the
        bounded observation window elapses (cost guardrail: nothing polls
        forever).
        """
        from .client import STATUS_SCALED_TO_ZERO

        deadline = time.monotonic() + deadline_seconds
        while time.monotonic() < deadline:
            status = self.client.deployment_status()
            if (
                status.get("status") == STATUS_SCALED_TO_ZERO
                or int(status.get("active_replica_count") or 0) == 0
            ):
                return self.writer.emit(
                    "scaled_to_zero",
                    profile=self.profile_name,
                    status=status.get("status"),
                    active_replica_count=status.get("active_replica_count"),
                )
            self.config.sleep(self.config.poll_interval_seconds)
        return None
