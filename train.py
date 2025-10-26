"""Thin training entrypoint that delegates to the orchestrator."""

from __future__ import annotations

import argparse
from pathlib import Path

from orchestrator import orchestrate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deepfake detectors")
    parser.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    args = parser.parse_args()

    orchestrate(args.config.resolve(), mode="training")


if __name__ == "__main__":
    main()
