"""Shared helpers for orchestrated training scripts.

The orchestrator drives training through existing model-specific scripts. To
avoid duplicating boilerplate in each trainer we provide a compact utility
module that handles environment overrides (output directories, random seeds,
auto-resume checkpoints, etc.). The intent is to keep every trainer focused on
deepfake frame classification while still cooperating with the orchestration
layer.
"""

from __future__ import annotations

import atexit
import io
import json
import os
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
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


_LOG_HANDLES: list[io.TextIOBase] = []


class _TeeStream(io.TextIOBase):
    """Duplicate writes to both the terminal and a log file."""

    def __init__(self, primary: io.TextIOBase, secondary: io.TextIOBase) -> None:
        super().__init__()
        self._primary = primary
        self._secondary = secondary
        self._encoding = getattr(primary, "encoding", "utf-8")

    def write(self, data: str) -> int:  # noqa: D401 - TextIOBase contract
        self._primary.write(data)
        self._secondary.write(data)
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def isatty(self) -> bool:  # noqa: D401 - TextIOBase contract
        return bool(getattr(self._primary, "isatty", lambda: False)())

    @property
    def encoding(self) -> str:  # noqa: D401 - TextIOBase contract
        return self._encoding

    def close(self) -> None:
        try:
            self._secondary.close()
        finally:
            super().close()


def create_console(*, width: int | None = None) -> Console:
    """Build a Rich console that mirrors output to ``LOG_PATH`` if provided."""

    log_path_value = os.environ.get("LOG_PATH")
    stream: io.TextIOBase = sys.stdout
    if log_path_value:
        log_path = Path(log_path_value).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")
        _LOG_HANDLES.append(log_file)
        atexit.register(log_file.close)
        stream = _TeeStream(stream, log_file)

    force_terminal = bool(getattr(sys.stdout, "isatty", lambda: False)())
    return Console(file=stream, force_terminal=force_terminal, width=width)


def _as_bool(value: Any) -> bool:
    """Best-effort boolean coercion used for transform toggles."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def load_transform_toggles(
    defaults: dict[str, bool],
    *,
    env_var: str = "TRANSFORMS",
    required: Sequence[str] | None = None,
) -> dict[str, bool]:
    """Return per-transform enable flags supplied via environment variables.

    Parameters
    ----------
    defaults:
        Baseline toggle values for each transform key.
    env_var:
        Environment variable that may contain a JSON mapping of
        ``{"transform_name": bool}`` overrides. Missing keys default to
        ``defaults``.
    required:
        Keys that must remain enabled. If an override disables them the value
        is forced back to ``True`` to avoid breaking the data pipeline.
    """

    toggles = dict(defaults)
    overrides_raw = os.environ.get(env_var)
    if overrides_raw:
        try:
            parsed = json.loads(overrides_raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                toggles[key] = _as_bool(value)

    if required:
        for key in required:
            if not toggles.get(key, False):
                toggles[key] = True

    return toggles


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
        ``OUTPUT_DIR``. Defaults to the current working directory.
    best_checkpoint_name / latest_checkpoint_name:
        Filenames for checkpoint files within the checkpoints directory.
    """

    base_dir = Path(
        os.environ.get("OUTPUT_DIR", default_output_dir or Path.cwd())
    ).expanduser().resolve()
    checkpoints_dir = base_dir / "checkpoints"
    logs_dir = base_dir / "logs"
    for path in (base_dir, checkpoints_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    best_weights_path = base_dir / weights_name
    best_checkpoint_path = checkpoints_dir / best_checkpoint_name
    latest_checkpoint_path = checkpoints_dir / latest_checkpoint_name

    resume_flag = os.environ.get("RESUME_AUTO", "").strip()
    resume_checkpoint: Path | None = None
    if resume_flag == "1" and latest_checkpoint_path.exists():
        resume_checkpoint = latest_checkpoint_path

    seed = int(os.environ["SEED"]) if "SEED" in os.environ else None
    device_override = os.environ.get("DEVICE")

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

    value = os.environ.get(name)
    return value if value is not None else default


def env_int(name: str, default: int) -> int:
    """Return an integer override supplied via environment variables."""

    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    """Return a float override supplied via environment variables."""

    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
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


def require_num_classes(
    dataset: Any,
    expected: int,
    *,
    split: str,
    dataset_root: Path | str | None = None,
) -> None:
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
    root_hint = ""
    if dataset_root is not None:
        root_hint = f" at {Path(dataset_root)}"
    raise ValueError(
        "Class count mismatch for split "
        f"'{split}'{root_hint}: dataset exposes {actual} classes ({preview}) "
        f"but configuration sets NUM_CLASSES={expected}. "
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
