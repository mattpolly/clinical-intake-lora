"""One-shot voice-intake prototype loop (research only, synthetic scenarios).

Pipeline: Polly fixture -> streaming Transcribe -> intake model (mock/Baseten)
-> Rime TTS, saving a complete JSON artifact with per-stage latencies and
explicit per-stage markers when a credential is missing or denied. The run
never silently aborts; unexpected errors still produce an artifact.

Usage:
  .venv/bin/python -m voice_proto.run_loop [--env-file ../textlife-specs/.env]
      [--backend auto|mock|baseten] [--skip-aws] [--fixture-text "..."]
      [--output-dir outputs/voice_proto]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from . import fixture as fixture_mod
from . import model as model_mod
from . import stt as stt_mod
from . import tts as tts_mod
from .envfile import load_env_file

RESEARCH_NOTE = (
    "Research-only prototype. Synthetic scenarios, no PHI, no diagnosis or "
    "treatment behavior, not validated for clinical use."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-file", default=None,
                   help="Optional KEY=VALUE file; existing env always wins.")
    p.add_argument("--backend", choices=["auto", "mock", "baseten"], default="auto",
                   help="Intake model backend (auto: baseten if credentialed).")
    p.add_argument("--skip-aws", action="store_true",
                   help="Skip Polly/Transcribe stages with explicit markers.")
    p.add_argument("--fixture-text", default=fixture_mod.DEFAULT_FIXTURE_TEXT)
    p.add_argument("--output-dir", default="outputs/voice_proto")
    return p.parse_args()


def run(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir)
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}Z"

    artifact: dict = {
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "voice_proto_version": __version__,
        "region": region,
        "backend_requested": args.backend,
        "research_note": RESEARCH_NOTE,
        "stages": {},
    }

    # Stage 1: synthetic patient-speech fixture (Polly; timed separately).
    if args.skip_aws:
        fx = {"status": "skipped", "text": args.fixture_text,
              "wav_path": None, "duration_s": None, "latency_ms": None}
    else:
        fx = fixture_mod.synthesize_fixture(
            text=args.fixture_text, region=region, out_dir=out_dir)
    artifact["stages"]["fixture"] = fx

    # Stage 2: streaming STT (measured loop latency starts here).
    loop_start = time.perf_counter()
    pcm = None
    if fx["status"] == "ok" and fx["wav_path"]:
        pcm = fixture_mod.read_wav_pcm(fx["wav_path"])
    if pcm is not None and not args.skip_aws:
        stt = stt_mod.transcribe_pcm(pcm, region=region)
    else:
        reason = "skipped" if args.skip_aws else fx["status"]
        stt = {"status": reason, "transcript": None, "partial_count": 0,
               "ttfp_ms": None, "latency_ms": None,
               "error": "no fixture audio; stage marked, not run"}
    artifact["stages"]["stt"] = stt

    # Stage 3: intake-model turn (uses the transcript, or the scripted fixture
    # text when STT could not run, so the artifact stays complete).
    patient_utterance = stt.get("transcript") or fx["text"]
    backend = model_mod.select_backend(args.backend)
    if backend is None:
        mdl = {"status": "missing-credential", "backend": "baseten",
               "text": None, "ttfb_ms": None, "latency_ms": None,
               "error": "BASETEN_API_KEY/BASETEN_MODEL_URL not set"}
    else:
        mdl = backend.turn([{"role": "patient", "content": patient_utterance}])
        mdl["backend"] = backend.name
    artifact["stages"]["model"] = mdl

    # Stage 4: TTS of the model reply.
    if mdl.get("text"):
        tts = tts_mod.synthesize(mdl["text"], out_dir=out_dir)
    else:
        tts = {"status": "skipped", "text": None, "audio_path": None,
               "bytes": None, "ttfb_ms": None, "latency_ms": None,
               "error": "no model text to synthesize"}
    artifact["stages"]["tts"] = tts

    artifact["total_loop_latency_ms"] = round(
        (time.perf_counter() - loop_start) * 1000, 1)
    artifact["patient_utterance_used"] = patient_utterance

    path = out_dir / f"run_{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    artifact["artifact_path"] = str(path)
    return artifact


def main() -> int:
    args = parse_args()
    if args.env_file:
        load_env_file(args.env_file)
    try:
        artifact = run(args)
    except Exception as exc:  # unexpected: still write what we can, then fail
        print(f"unexpected failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for name, stage in artifact["stages"].items():
        extra = stage.get("transcript") or stage.get("text") or stage.get("error") or ""
        print(f"  {name:8s} {stage.get('status'):20s} "
              f"{stage.get('latency_ms') or '-'}ms  {str(extra)[:70]}")
    print(f"  loop     {artifact['total_loop_latency_ms']}ms")
    print(f"artifact: {artifact['artifact_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
