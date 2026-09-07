"""Resolve Hugging Face models from this project's local cache only."""

from __future__ import annotations

from pathlib import Path


def cached_model_path(model_id: str, cache_dir: Path) -> Path:
    """Return the sole local snapshot for ``model_id`` without Hub metadata calls."""
    snapshots = cache_dir / f"models--{model_id.replace('/', '--')}" / "snapshots"
    candidates = sorted(path for path in snapshots.iterdir() if path.is_dir()) if snapshots.is_dir() else []
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one local snapshot for {model_id} under {snapshots}; found {len(candidates)}")
    return candidates[0]
