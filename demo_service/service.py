"""Session lifecycle for the research-only text demo API.

This module owns only bounded in-memory session state. Baseten lifecycle,
readiness, inference, and JSONL serialization remain in ``baseten_demo``.
"""

from __future__ import annotations

import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from baseten_demo.client import BasetenClient
from baseten_demo.harness import Harness, HarnessConfig, WakeReadyInferResult
from baseten_demo.instrumentation import EventWriter
from voice_proto.model import SYSTEM_PROMPT

from . import RESEARCH_DISCLAIMER

MAX_TURNS = 40
MAX_HISTORY_TOKENS = 6_000
MAX_PATIENT_TEXT_CHARS = 4_000
DEFAULT_SESSION_TIMEOUT_SECONDS = 1_800


class DemoServiceError(Exception):
    """A client-visible service error with a stable state."""

    def __init__(self, message: str, *, state: str, status_code: int) -> None:
        super().__init__(message)
        self.state = state
        self.status_code = status_code


class DemoClient(Protocol):
    model_id: str
    deployment_id: str

    def wake(self) -> None: ...
    def is_ready(self) -> bool: ...
    def deployment_status(self) -> dict: ...
    def infer(self, messages: list[dict[str, str]], *, max_tokens: int = 128,
              temperature: float = 0.2, stream: bool = True) -> dict: ...


@dataclass(frozen=True)
class DemoSettings:
    api_key: str
    model_id: str
    deployment_id: str
    environment: str = "production"
    served_model_name: str | None = None
    output_dir: Path = Path("outputs/baseten_demo")
    readiness_deadline_seconds: float = 600.0
    poll_interval_seconds: float = 5.0
    session_timeout_seconds: float = DEFAULT_SESSION_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> "DemoSettings":
        required = {
            "BASETEN_API_KEY": os.environ.get("BASETEN_API_KEY", ""),
            "BASETEN_MODEL_ID": os.environ.get("BASETEN_MODEL_ID", ""),
            "BASETEN_DEPLOYMENT_ID": os.environ.get("BASETEN_DEPLOYMENT_ID", ""),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("demo service requires server-side env vars: " + ", ".join(missing))
        return cls(
            api_key=required["BASETEN_API_KEY"],
            model_id=required["BASETEN_MODEL_ID"],
            deployment_id=required["BASETEN_DEPLOYMENT_ID"],
            environment=os.environ.get("BASETEN_ENVIRONMENT", "production"),
            served_model_name=os.environ.get("BASETEN_SERVED_MODEL_NAME") or None,
            output_dir=Path(os.environ.get("DEMO_OUTPUT_DIR", "outputs/baseten_demo")),
        )


@dataclass
class DemoSession:
    session_id: str
    started_monotonic: float
    started_at: str
    state: str = "waking"
    history: list[dict[str, str]] = field(default_factory=list)
    turns: int = 0
    error: str | None = None
    ended_at: str | None = None
    last_activity_monotonic: float = field(default_factory=time.monotonic)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def elapsed_seconds(self) -> float:
        return round(max(0.0, time.monotonic() - self.started_monotonic), 1)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _estimated_tokens(messages: list[dict[str, str]]) -> int:
    return sum(math.ceil(len(message["content"]) / 4) for message in messages)


def _session_writer(writer: EventWriter, session_id: str):
    """Inject a session id into the shared, bounded JSONL writer."""

    class SessionWriter:
        def emit(self, event: str, **fields):
            return writer.emit(event, session_id=session_id, **fields)

    return SessionWriter()


class DemoService:
    """Shares one deployment-readiness worker while retaining bounded sessions."""

    def __init__(
        self,
        settings: DemoSettings,
        *,
        client_factory: Callable[[], DemoClient] | None = None,
        writer: EventWriter | None = None,
    ) -> None:
        self.settings = settings
        self._client_factory = client_factory or self._build_client
        self.writer = writer or EventWriter(settings.output_dir / "demo_service_events.jsonl")
        self._sessions: dict[str, DemoSession] = {}
        self._lock = threading.RLock()
        self._readiness_thread: threading.Thread | None = None
        self._deployment_state = "cold"
        self._deployment_error: str | None = None

    def _build_client(self) -> BasetenClient:
        return BasetenClient(
            api_key=self.settings.api_key,
            model_id=self.settings.model_id,
            deployment_id=self.settings.deployment_id,
            environment=self.settings.environment,
            served_model_name=self.settings.served_model_name,
        )

    def _new_session(self) -> DemoSession:
        now = time.monotonic()
        return DemoSession(session_id=uuid.uuid4().hex, started_monotonic=now, started_at=_utc_now())

    def _expire_if_needed(self, session: DemoSession) -> None:
        if session.state in {"ended", "error", "expired"}:
            return
        if time.monotonic() - session.last_activity_monotonic > self.settings.session_timeout_seconds:
            session.state = "expired"
            session.ended_at = _utc_now()
            self.writer.emit("session_ended", session_id=session.session_id, state="expired")

    def _status_payload(self, session: DemoSession) -> dict:
        self._expire_if_needed(session)
        return {
            "session_id": session.session_id,
            "state": session.state,
            "elapsed_seconds": session.elapsed_seconds(),
            "turns": session.turns,
            "research_only": True,
            "disclaimer": RESEARCH_DISCLAIMER,
            **({"error": session.error} if session.error else {}),
        }

    def start(self) -> dict:
        with self._lock:
            session = self._new_session()
            self._sessions[session.session_id] = session
            self.writer.emit("session_started", session_id=session.session_id, state="waking")
            if self._deployment_state == "ready":
                session.state = "ready"
                self.writer.emit("session_ready", session_id=session.session_id, state="ready")
            elif self._deployment_state == "error":
                session.state, session.error = "error", self._deployment_error
            elif self._deployment_state != "waking":
                self._deployment_state = "waking"
                self._readiness_thread = threading.Thread(
                    target=self._wake_and_probe, args=(session.session_id,), daemon=True,
                    name="demo-baseten-readiness",
                )
                self._readiness_thread.start()
            return self._status_payload(session)

    def _wake_and_probe(self, initiating_session_id: str) -> None:
        session = self._sessions[initiating_session_id]
        harness = Harness(
            self._client_factory(), _session_writer(self.writer, initiating_session_id),
            config=HarnessConfig(
                readiness_deadline_seconds=self.settings.readiness_deadline_seconds,
                poll_interval_seconds=self.settings.poll_interval_seconds,
            ),
            profile_name="demo-service",
        )
        try:
            wake_event = harness.wake()
            result = harness.wait_ready_probe(wake_event)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {str(exc)[:200]}"
            self.writer.emit("error", session_id=initiating_session_id, stage="readiness", error_type=type(exc).__name__, message=detail)
            result = WakeReadyInferResult(False, "wake_error", detail, _utc_now())
        with self._lock:
            if result.ok:
                self._deployment_state, self._deployment_error = "ready", None
                for item in self._sessions.values():
                    if item.state == "waking":
                        item.state = "ready"
                        item.last_activity_monotonic = time.monotonic()
                        self.writer.emit("session_ready", session_id=item.session_id, state="ready")
            else:
                self._deployment_state, self._deployment_error = "error", result.detail
                for item in self._sessions.values():
                    if item.state == "waking":
                        item.state, item.error = "error", result.detail

    def get_status(self, session_id: str) -> dict:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise DemoServiceError("unknown session", state="missing", status_code=404)
            return self._status_payload(session)

    def turn(self, session_id: str, patient_text: str) -> dict:
        text = patient_text.strip() if isinstance(patient_text, str) else ""
        if not text:
            raise DemoServiceError("patient_text is required", state="invalid", status_code=422)
        if len(text) > MAX_PATIENT_TEXT_CHARS:
            raise DemoServiceError(f"patient_text exceeds {MAX_PATIENT_TEXT_CHARS} characters", state="invalid", status_code=422)
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise DemoServiceError("unknown session", state="missing", status_code=404)
            self._expire_if_needed(session)
            if session.state != "ready":
                raise DemoServiceError("model is not ready", state=session.state, status_code=503)
            if session.turns >= MAX_TURNS:
                raise DemoServiceError("session turn limit reached", state="ended", status_code=409)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + session.history + [{"role": "user", "content": text}]
            if _estimated_tokens(messages) > MAX_HISTORY_TOKENS:
                raise DemoServiceError("session history token limit reached", state="ended", status_code=409)
            self.writer.emit("turn_requested", session_id=session_id, input_characters=len(text), estimated_tokens=_estimated_tokens(messages))
        try:
            result = self._client_factory().infer(messages, max_tokens=128, temperature=0.2, stream=False)
        except Exception as exc:
            # Client failures are reported as a bounded, non-secret diagnostic;
            # do not turn an unavailable model into an indefinitely queued turn.
            detail = f"{type(exc).__name__}: {str(exc)[:200]}"
            self.writer.emit("error", session_id=session_id, stage="turn", error_type=type(exc).__name__, message=detail)
            raise DemoServiceError("model inference failed", state="error", status_code=502) from exc
        answer = str(result.get("text") or "").strip()
        if result.get("status") != "ok" or not answer:
            self.writer.emit("error", session_id=session_id, stage="turn", error_type="InferenceError", message="model returned no text")
            raise DemoServiceError("model returned no intake response", state="error", status_code=502)
        with self._lock:
            session.history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": answer}])
            session.turns += 1
            session.last_activity_monotonic = time.monotonic()
            self.writer.emit("inference_result", session_id=session_id, duration_ms=result.get("duration_ms"), time_to_first_token_ms=result.get("time_to_first_token_ms"), status="ok")
            self.writer.emit("last_activity", session_id=session_id)
            return {
                "session_id": session_id,
                "state": "ready",
                "turn": {"text": answer, "time_to_first_token_ms": result.get("time_to_first_token_ms"), "duration_ms": result.get("duration_ms")},
                "research_only": True,
                "disclaimer": RESEARCH_DISCLAIMER,
            }

    def end(self, session_id: str) -> dict:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise DemoServiceError("unknown session", state="missing", status_code=404)
            if session.state not in {"ended", "expired"}:
                session.state, session.ended_at = "ended", _utc_now()
                session.last_activity_monotonic = time.monotonic()
                self.writer.emit("last_activity", session_id=session_id)
                self.writer.emit("session_ended", session_id=session_id, state="ended")
            return self._status_payload(session)
