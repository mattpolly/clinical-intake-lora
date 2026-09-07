"""voice_proto — headless voice-intake prototype (research only).

Prototype of the docs/DEPLOYMENT.md call path: AWS Transcribe streaming STT,
a pluggable intake-model backend (deterministic mock now, Baseten later), and
Rime Mist TTS, with per-stage latency evidence.

Research only. Synthetic/standardized-patient scenarios, no PHI, no diagnosis
or treatment behavior, not validated for clinical use.
"""

__version__ = "0.1.0"
