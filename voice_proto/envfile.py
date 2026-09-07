"""Load KEY=VALUE env files without overwriting real environment variables.

Secret values are never logged, returned, or written to artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path) -> int:
    """Set env vars from a .env-style file. Existing env wins. Returns count."""
    loaded = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded
