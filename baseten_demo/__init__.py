"""Baseten on-demand demo deployment backend (research only).

This package is the hosting layer for a scale-to-zero Qwen3 demo deployment:
declarative model/GPU profiles, a server-side wake -> readiness -> inference
path (Baseten API key never leaves the backend), cold-start instrumentation,
and a dry-run/mock mode so the full path is testable without live Baseten
calls.

Research only: synthetic/standardized-patient scenarios, no PHI, no diagnosis
or treatment behavior, no clinical-use claims.
"""

__version__ = "0.1.0"
