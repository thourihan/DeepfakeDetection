from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


@dataclass
class RunContext:
    """Filesystem + runtime info for a single model run."""

    model_name: str
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    plots_dir: Path
    device: torch.device
    seed: int | None


class CheckpointManager:
    """Save/load checkpoints with fingerprint validation."""

    def __init__(self, run_context: RunContext, fingerprint: dict[str, Any]):
        self.run_context = run_context
        self.fingerprint = fingerprint
        self.latest_path = run_context.checkpoints_dir / "latest.ckpt"
        self.best_path = run_context.checkpoints_dir / "best.ckpt"

    def save(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        best_metric: float,
        best_epoch: int,
        extra: dict[str, Any] | None = None,
        is_best: bool = False,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "fingerprint": self.fingerprint,
            "saved_at": datetime.now().isoformat(),
        }
        if extra:
            state.update(extra)
        torch.save(state, self.latest_path)
        if is_best:
            torch.save(state, self.best_path)
        return state

    def load(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        prefer_best: bool = False,
        strict: bool = True,
    ) -> dict[str, Any] | None:
        path = self.best_path if prefer_best and self.best_path.exists() else self.latest_path
        if not path.exists():
            return None
        state = torch.load(path, map_location="cpu")
        stored_fp = state.get("fingerprint")
        if stored_fp and stored_fp != self.fingerprint:
            if strict:
                msg = (
                    "Checkpoint fingerprint mismatch. Refusing to resume. "
                    f"expected={self.fingerprint} stored={stored_fp}"
                )
                raise ValueError(msg)
            return None
        model.load_state_dict(state["model"])
        if optimizer and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        return state


__all__ = ["RunContext", "CheckpointManager"]
