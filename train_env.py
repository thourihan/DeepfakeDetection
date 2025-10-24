"""Shared helpers for orchestrated training scripts.

The orchestrator drives training through existing model-specific scripts. To
avoid duplicating boilerplate in each trainer we provide a compact utility
module that handles environment overrides (output directories, random seeds,
auto-resume checkpoints, etc.). The intent is to keep every trainer focused on
deepfake frame classification while still cooperating with the orchestration
layer.
"""

from __future__ import annotations

import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


@dataclass(frozen=True)
class TrainingEnvironment:
    """Resolved runtime settings derived from environment variables."""

    output_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    best_weights_path: Path
    best_checkpoint_path: Path
    latest_checkpoint_path: Path
    resume_checkpoint: Path | None
    seed: int | None
    device_override: str | None


def prepare_training_environment(
    *,
    weights_name: str,
    default_output_dir: Path | None = None,
    best_checkpoint_name: str = "best.ckpt",
    latest_checkpoint_name: str = "latest.ckpt",
) -> TrainingEnvironment:
    """Resolve directories and resume behaviour for a trainer.

    Parameters
    ----------
    weights_name:
        Filename used when saving the best model weights.
    default_output_dir:
        Where to place outputs when the orchestrator does not set
        ``DD_OUTPUT_DIR``. Defaults to the current working directory.
    best_checkpoint_name / latest_checkpoint_name:
        Filenames for checkpoint files within the checkpoints directory.
    """

    base_dir = Path(
        os.environ.get("DD_OUTPUT_DIR", default_output_dir or Path.cwd())
    ).expanduser().resolve()
    checkpoints_dir = base_dir / "checkpoints"
    logs_dir = base_dir / "logs"
    for path in (base_dir, checkpoints_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    best_weights_path = base_dir / weights_name
    best_checkpoint_path = checkpoints_dir / best_checkpoint_name
    latest_checkpoint_path = checkpoints_dir / latest_checkpoint_name

    resume_flag = os.environ.get("DD_RESUME_AUTO", "").strip()
    resume_checkpoint: Path | None = None
    if resume_flag == "1" and latest_checkpoint_path.exists():
        resume_checkpoint = latest_checkpoint_path

    seed = int(os.environ["DD_SEED"]) if "DD_SEED" in os.environ else None
    device_override = os.environ.get("DD_DEVICE")

    return TrainingEnvironment(
        output_dir=base_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        best_weights_path=best_weights_path,
        best_checkpoint_path=best_checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
        resume_checkpoint=resume_checkpoint,
        seed=seed,
        device_override=device_override,
    )


def apply_seed(seed: int | None) -> None:
    """Seed Python, NumPy, and PyTorch if requested."""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def env_path(name: str, default: Path) -> Path:
    """Return a :class:`Path` override supplied via environment variables."""

    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


def env_str(name: str, default: str) -> str:
    """Return a string override supplied via environment variables."""

    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    """Return an integer override supplied via environment variables."""

    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def save_latest_checkpoint(
    env: TrainingEnvironment,
    *,
    model: torch.nn.Module,
    optimizer: Optimizer | None,
    scheduler: LRScheduler | None,
    epoch: int,
    best_val_acc: float,
    best_epoch: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the most recent training state to ``latest.ckpt``."""

    state: dict[str, Any] = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
    }
    if extra:
        state.update(extra)
    torch.save(state, env.latest_checkpoint_path)
    return state


def save_best_checkpoint(env: TrainingEnvironment, state: dict[str, Any]) -> None:
    """Persist the best-performing weights and checkpoint."""

    torch.save(state, env.best_checkpoint_path)
    torch.save(state["model"], env.best_weights_path)


def maybe_load_checkpoint(
    env: TrainingEnvironment,
    *,
    model: torch.nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> dict[str, Any] | None:
    """Load ``latest.ckpt`` when auto-resume is requested."""

    if env.resume_checkpoint is None:
        return None

    state = torch.load(env.resume_checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state


def require_num_classes(dataset: Any, expected: int, *, split: str) -> None:
    """Ensure an ImageFolder-style dataset exposes the configured classes."""

    if expected <= 0:
        raise ValueError("expected number of classes must be positive")

    classes: Sequence[Any] | None = getattr(dataset, "classes", None)
    if classes is None:
        return

    actual = len(classes)
    if actual == expected:
        return

    preview = ", ".join(str(name) for name in classes[: min(5, actual)])
    if actual > 5:
        preview += ", …"
    raise ValueError(
        "Class count mismatch for split "
        f"'{split}': dataset exposes {actual} classes ({preview}) "
        f"but configuration sets DD_NUM_CLASSES={expected}. "
        "Update config.data.num_classes (e.g., match it to the true number of "
        "categories in your ImageFolder)."
    )


__all__ = [
    "TrainingEnvironment",
    "apply_seed",
    "env_int",
    "env_path",
    "env_str",
    "maybe_load_checkpoint",
    "prepare_training_environment",
    "require_num_classes",
    "save_best_checkpoint",
    "save_latest_checkpoint",
]
