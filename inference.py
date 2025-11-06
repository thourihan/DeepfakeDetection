"""Thin inference entrypoint that delegates to the orchestrator."""

from __future__ import annotations

import argparse
from pathlib import Path

from orchestration.orchestrator import orchestrate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deepfake inference")
    parser.add_argument("--config", type=Path, default=Path("config/inference.yaml"))
    args = parser.parse_args()

    orchestrate(args.config.resolve(), mode="inference")


if __name__ == "__main__":
    main()
