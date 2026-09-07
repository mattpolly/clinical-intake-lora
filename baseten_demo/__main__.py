"""CLI entry: ``python -m baseten_demo ...``.

Covers wake -> readiness -> inference with a ``--dry-run`` mock mode that
emits a complete simulated event trace (no live Baseten calls, no API key).
Live mode reads server-side environment variables (``BASETEN_API_KEY``,
``BASETEN_MODEL_ID``, ``BASETEN_DEPLOYMENT_ID``) and never in this slice
creates or touches a live deployment beyond wake/status/inference.

Research only: no PHI, no diagnosis/treatment, no clinical-use claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .deployment import build_deployment_definition
from .harness import Harness, HarnessConfig
from .instrumentation import EventWriter
from .mock import MockBasetenClient
from .profiles import ProfileValidationError, load_profiles

_DEFAULT_OUTPUT_DIR = "outputs/baseten_demo"


def _env_file_loader(path: str) -> None:
    """Load KEY=VALUE entries without overwriting real environment variables."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _new_output_path(output_dir: str) -> Path:
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}Z"
    return Path(output_dir) / f"events_{run_id}.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m baseten_demo",
        description="Baseten scale-to-zero demo backend (server-side).",
    )
    parser.add_argument("--profile", default=None, help="Profile name (default: config default).")
    parser.add_argument(
        "--profiles-file", default=None, help="Path to profiles.yaml (default: package file)."
    )
    parser.add_argument("--env-file", default=None, help="Optional server-side KEY=VALUE file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run against the mock client (no live Baseten calls, no API key).",
    )
    parser.add_argument(
        "--deployment",
        action="store_true",
        help="Print the deployment definition for the profile and exit.",
    )
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--readiness-deadline-seconds",
        type=float,
        default=600,
        help="Bounded readiness deadline (default 600).",
    )
    parser.add_argument(
        "--poll-interval-seconds", type=float, default=5.0,
        help="Readiness poll spacing (default 5).",
    )
    parser.add_argument(
        "--demo-turn",
        action="store_true",
        help="After readiness, run one scripted demo inference turn.",
    )
    parser.add_argument(
        "--observe-scale-to-zero",
        action="store_true",
        help="After inference, observe scale-down (best-effort, bounded).",
    )
    parser.add_argument(
        "--scale-to-zero-window-seconds",
        type=float,
        default=120,
        help="Bound for scale-to-zero observation (default 120).",
    )
    return parser.parse_args()


def _resolve_live_ids() -> tuple[str, str, str, str]:
    api_key = os.environ.get("BASETEN_API_KEY", "")
    model_id = os.environ.get("BASETEN_MODEL_ID", "")
    deployment_id = os.environ.get("BASETEN_DEPLOYMENT_ID", "")
    environment = os.environ.get("BASETEN_ENVIRONMENT", "production")
    missing = [
        name
        for name, value in (
            ("BASETEN_API_KEY", api_key),
            ("BASETEN_MODEL_ID", model_id),
            ("BASETEN_DEPLOYMENT_ID", deployment_id),
        )
        if not value
    ]
    if missing:
        print(
            "error: live mode requires server-side env vars: " + ", ".join(missing),
            file=sys.stderr,
        )
        raise SystemExit(2)
    return api_key, model_id, deployment_id, environment


def _print_definition(args: argparse.Namespace, profile) -> None:
    model_id = os.environ.get("BASETEN_MODEL_ID", "<BASETEN_MODEL_ID>")
    deployment_id = os.environ.get("BASETEN_DEPLOYMENT_ID", "<BASETEN_DEPLOYMENT_ID>")
    environment = os.environ.get("BASETEN_ENVIRONMENT", "production")
    definition = build_deployment_definition(profile, model_id, deployment_id, environment)
    print(json.dumps(definition, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = parse_args()
    if args.env_file:
        _env_file_loader(args.env_file)

    try:
        profiles = load_profiles(args.profiles_file) if args.profiles_file else load_profiles()
        profile = profiles.get(args.profile)
    except ProfileValidationError as exc:
        print(f"profile error: {exc}", file=sys.stderr)
        return 2

    if args.deployment:
        _print_definition(args, profile)
        return 0

    from .client import BasetenClient

    if args.dry_run:
        client = MockBasetenClient(
            model_id="mock-model-id", deployment_id="mock-deployment-id"
        )
    else:
        api_key, model_id, deployment_id, environment = _resolve_live_ids()
        client = BasetenClient(
            api_key=api_key,
            model_id=model_id,
            deployment_id=deployment_id,
            environment=environment,
            served_model_name=profile.served_model_name,
        )

    writer = EventWriter(_new_output_path(args.output_dir))
    harness = Harness(
        client,
        writer,
        config=HarnessConfig(
            readiness_deadline_seconds=args.readiness_deadline_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        ),
        profile_name=profile.name,
    )

    result = harness.wake_ready_probe()
    if not result.ok:
        print(f"state: {result.state} — {result.detail}", file=sys.stderr)
    else:
        print(f"ready: wake->ready via health+probe (ttfb "
              f"{result.first_inference['time_to_first_token_ms']}ms)")

    if result.ok and args.demo_turn:
        turn = harness.infer_turn()
        if args.dry_run and hasattr(client, "mark_idle_and_scale_to_zero"):
            client.mark_idle_and_scale_to_zero()
        print(f"demo turn: {turn.get('text')!r} (ttfb "
              f"{turn.get('time_to_first_token_ms')}ms, "
              f"duration {turn.get('duration_ms')}ms)")

    if args.observe_scale_to_zero:
        event = harness.observe_scale_to_zero(
            deadline_seconds=args.scale_to_zero_window_seconds
        )
        print("scaled_to_zero: " + ("observed" if event else "not observed in window"))

    print(f"events: {writer.path}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
