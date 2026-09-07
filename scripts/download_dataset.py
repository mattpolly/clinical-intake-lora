#!/usr/bin/env python3
"""Download a reproducible local snapshot of the MedSP1000 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "byrLLCC/MedSP1000"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "medsp1000",
        help="Destination for the Hugging Face dataset snapshot.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help=(
            "Case path to download, repeatable (e.g. mededportal_9045/scenario1). "
            "Without this flag, download the full dataset."
        ),
    )
    args = parser.parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = [f"{case.rstrip('/')}/**" for case in args.case] or None
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=args.local_dir,
        allow_patterns=allow_patterns,
    )
    print(path)


if __name__ == "__main__":
    main()
