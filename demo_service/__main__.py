"""Run the research-only demo API with server-side environment configuration."""

from __future__ import annotations

import argparse
import os

from voice_proto.envfile import load_env_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only clinical intake demo API.")
    parser.add_argument("--env-file", default=None, help="Optional server-side .env path.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.env_file:
        load_env_file(args.env_file)
    from uvicorn import run
    from .app import create_app
    run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
