"""Unified orchestration layer for DeepfakeDetection training and inference."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_registry import get_model_spec
from train_env import apply_seed

console = Console()


@dataclass(frozen=True)
class RunPaths:
    """Filesystem layout for a single model run."""

    run_dir: Path
    checkpoints: Path
    logs: Path
    plots: Path


@contextlib.contextmanager
def patched_environ(overrides: dict[str, str]) -> Iterator[None]:
    """Temporarily set environment variables for a trainer."""

    original: dict[str, str | None] = {}
    for key, value in overrides.items():
        original[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def tee_output(log_path: Path) -> Iterator[None]:
    """Mirror stdout/stderr to both the console and a log file."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        stdout, stderr = sys.stdout, sys.stderr

        class Tee(io.TextIOBase):
            def write(self, data: str) -> int:  # noqa: D401 - standard file contract
                stdout.write(data)
                log_file.write(data)
                return len(data)

            def flush(self) -> None:
                stdout.flush()
                log_file.flush()

            def isatty(self) -> bool:  # noqa: D401 - standard file contract
                return bool(getattr(stdout, "isatty", lambda: False)())

            @property
            def encoding(self) -> str:  # noqa: D401 - standard file contract
                return getattr(stdout, "encoding", "utf-8")

        tee = Tee()
        sys.stdout = tee  # type: ignore[assignment]
        sys.stderr = tee  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = stdout  # type: ignore[assignment]
            sys.stderr = stderr  # type: ignore[assignment]
            log_file.flush()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def ensure_run_dirs(base: Path, timestamp: str) -> RunPaths:
    run_dir = base / timestamp
    checkpoints = run_dir / "checkpoints"
    logs = run_dir / "logs"
    plots = run_dir / "plots"
    for path in (run_dir, checkpoints, logs, plots):
        path.mkdir(parents=True, exist_ok=True)
    return RunPaths(run_dir=run_dir, checkpoints=checkpoints, logs=logs, plots=plots)


def snapshot_config(run_paths: RunPaths, *, config: dict[str, Any], model_cfg: dict[str, Any]) -> None:
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "global": {
            k: v
            for k, v in config.items()
            if k not in {"models", "selection"}
        },
        "model": model_cfg,
    }
    with (run_paths.run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle)


def resolve_transform_mapping(model_cfg: dict[str, Any], *, phase: str) -> dict[str, Any] | None:
    """Return a mapping of transform toggles for ``phase`` if configured."""

    transforms_cfg = model_cfg.get("transforms")
    if isinstance(transforms_cfg, dict):
        phase_cfg = transforms_cfg.get(phase)
        if isinstance(phase_cfg, dict):
            return phase_cfg
        if all(isinstance(v, bool | int | float | str) for v in transforms_cfg.values()):
            return transforms_cfg

    if phase == "train":
        scoped = model_cfg.get("training", {}).get("transforms")
    else:
        scoped = model_cfg.get("inference", {}).get("transforms")
    if isinstance(scoped, dict):
        return scoped

    return None


def build_env_overrides(
    *,
    config: dict[str, Any],
    model_cfg: dict[str, Any],
    run_paths: RunPaths,
    training: bool,
) -> dict[str, str]:
    data_cfg = config.get("data", {})
    overrides: dict[str, str] = {
        "DD_OUTPUT_DIR": str(run_paths.run_dir),
    }

    seed = config.get("seed")
    if seed is not None:
        overrides["DD_SEED"] = str(seed)

    device = config.get("device")
    if device:
        overrides["DD_DEVICE"] = str(device)

    data_root = data_cfg.get("root")
    if data_root:
        overrides["DD_DATA_ROOT"] = str(Path(data_root).expanduser().resolve())
    for key, env_key in (
        ("train_split", "DD_TRAIN_SPLIT"),
        ("val_split", "DD_VAL_SPLIT"),
        ("test_split", "DD_TEST_SPLIT"),
        ("img_size", "DD_IMG_SIZE"),
        ("num_classes", "DD_NUM_CLASSES"),
    ):
        value = data_cfg.get(key)
        if value is not None:
            overrides[env_key] = str(value)

    num_classes = model_cfg.get("num_classes")
    if num_classes is not None:
        overrides["DD_NUM_CLASSES"] = str(num_classes)

    if training:
        train_cfg = model_cfg.get("training", {})
        if "batch_size" in train_cfg:
            overrides["DD_BATCH_SIZE"] = str(train_cfg["batch_size"])
        if "epochs" in train_cfg:
            overrides["DD_EPOCHS"] = str(train_cfg["epochs"])
        if "num_workers" in train_cfg:
            overrides["DD_NUM_WORKERS"] = str(train_cfg["num_workers"])
        resume_flag = str(train_cfg.get("resume", "")).lower()
        overrides["DD_RESUME_AUTO"] = "1" if resume_flag in {"1", "true", "auto"} else "0"
    else:
        infer_cfg = model_cfg.get("inference", {})
        if "batch_size" in infer_cfg:
            overrides["DD_BATCH_SIZE"] = str(infer_cfg["batch_size"])
        if "num_workers" in infer_cfg:
            overrides["DD_NUM_WORKERS"] = str(infer_cfg["num_workers"])

    phase_key = "train" if training else "eval"
    transform_overrides = resolve_transform_mapping(model_cfg, phase=phase_key)
    if transform_overrides:
        overrides["DD_TRANSFORMS"] = json.dumps(transform_overrides)

    return overrides


def import_trainer(module_name: str) -> Any:
    module = importlib.import_module(module_name)
    if hasattr(module, "main"):
        return module.main
    msg = f"Trainer module '{module_name}' must expose a main() function."
    raise AttributeError(msg)


def run_training_job(config: dict[str, Any], model_cfg: dict[str, Any], run_paths: RunPaths) -> None:
    spec = get_model_spec(model_cfg["name"])
    overrides = build_env_overrides(config=config, model_cfg=model_cfg, run_paths=run_paths, training=True)
    log_path = run_paths.logs / "train.log"
    log_path.unlink(missing_ok=True)
    overrides["DD_LOG_PATH"] = str(log_path)

    console.print(f"[bold]→ training {model_cfg['name']}[/]")
    with patched_environ(overrides):
        trainer_main = import_trainer(spec.train_module)
        trainer_main()


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def build_eval_transforms(
    image_size: int,
    *,
    toggles: dict[str, Any] | None = None,
) -> transforms.Compose:
    defaults = {
        "ensure_rgb": True,
        "val_resize": True,
        "val_center_crop": True,
        "val_to_tensor": True,
        "val_normalize": True,
    }
    resolved = dict(defaults)
    if toggles:
        for key, value in toggles.items():
            resolved[key] = _coerce_bool(value)

    ops: list[object] = []
    if resolved.get("ensure_rgb", True):
        ops.append(transforms.Lambda(_ensure_rgb))
    if resolved.get("val_resize", True):
        ops.append(transforms.Resize(image_size))
    if resolved.get("val_center_crop", True):
        ops.append(transforms.CenterCrop(image_size))
    if resolved.get("val_to_tensor", True):
        ops.append(transforms.ToTensor())
    if resolved.get("val_normalize", True):
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms.Compose(ops)


def load_model(
    model_name: str,
    num_classes: int,
    weights_path: Path | None,
    device: torch.device,
) -> nn.Module:
    spec = get_model_spec(model_name)
    model = spec.builder(model_name, num_classes)
    model.to(device)
    model.eval()

    if weights_path is not None and weights_path.exists():
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)

    return model


def build_inference_loader(
    *,
    dataset: datasets.ImageFolder,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def save_confusion_matrix(cm: np.ndarray, labels: list[str], path: Path) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, path: Path) -> None:
    from sklearn.metrics import RocCurveDisplay

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_scores, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_inference_job(
    *,
    config_path: Path,
    config: dict[str, Any],
    model_cfg: dict[str, Any],
    run_paths: RunPaths,
) -> None:
    console.print(f"[bold]→ inference {model_cfg['name']}[/]")
    log_path = run_paths.logs / "inference.log"
    log_path.unlink(missing_ok=True)

    with tee_output(log_path):
        _run_inference_job(config_path=config_path, config=config, model_cfg=model_cfg, run_paths=run_paths)


def _run_inference_job(
    *,
    config_path: Path,
    config: dict[str, Any],
    model_cfg: dict[str, Any],
    run_paths: RunPaths,
) -> None:
    local_console = Console(file=sys.stdout, force_terminal=bool(getattr(sys.stdout, "isatty", lambda: False)()))
    spec = get_model_spec(model_cfg["name"])
    data_cfg = config.get("data", {})
    infer_cfg = model_cfg.get("inference", {})
    split_name = infer_cfg.get("split", data_cfg.get("test_split", "test"))
    local_console.print(
        f"[bold]Model[/]: {model_cfg['name']} | split={split_name} | batch={infer_cfg.get('batch_size', 64)}",
    )

    num_classes = int(model_cfg.get("num_classes", data_cfg.get("num_classes", 2)))
    image_size = int(infer_cfg.get("img_size", data_cfg.get("img_size", spec.default_image_size)))
    batch_size = int(infer_cfg.get("batch_size", 64))
    num_workers = int(infer_cfg.get("num_workers", 4))

    device_str = config.get("device", "cuda")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        local_console.print("[bold yellow]⚠️  CUDA requested but unavailable[/]: using CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    weights_path_value = infer_cfg.get("weights")
    weights_path: Path | None = None
    if weights_path_value:
        weights_path = Path(weights_path_value)
        if not weights_path.is_absolute():
            weights_path = (config_path.parent / weights_path).resolve()

    model = load_model(model_cfg["name"], num_classes, weights_path, device)
    eval_toggles = resolve_transform_mapping(model_cfg, phase="eval")
    transforms_eval = build_eval_transforms(image_size, toggles=eval_toggles)

    data_root_value = data_cfg.get("root", "data/DeepFakeFrames")
    data_root = Path(data_root_value).expanduser()
    if not data_root.is_absolute():
        data_root = (config_path.parent / data_root).resolve()

    split = split_name
    dataset_path = data_root / split
    if not dataset_path.exists():
        local_console.print(f"[bold red]Split not found:[/] {dataset_path}")
        raise SystemExit(1)

    dataset = datasets.ImageFolder(dataset_path, transform=transforms_eval)
    if len(dataset) == 0:
        local_console.print(f"[bold yellow]No images found in[/] {dataset_path}")
        return

    dataloader = build_inference_loader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[speed]}"),
        console=local_console,
    )

    all_probs: list[torch.Tensor] = []
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    start = perf_counter()
    images_seen = 0
    with progress:
        task = progress.add_task("inference", total=len(dataloader), speed="")
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            with torch.inference_mode():
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            images_seen += targets.size(0)
            elapsed = perf_counter() - start
            speed = images_seen / max(elapsed, 1e-6)
            progress.update(task, advance=1, speed=f"{speed:.1f} img/s")

    probs_tensor = torch.cat(all_probs, dim=0)
    preds_tensor = torch.cat(all_preds, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    accuracy = (preds_tensor == targets_tensor).float().mean().item()
    metrics: dict[str, Any] = {
        "model": model_cfg["name"],
        "split": split,
        "accuracy": accuracy,
        "timestamp": datetime.now().isoformat(),
    }

    unique_targets = torch.unique(targets_tensor)
    if unique_targets.numel() > 1:
        try:
            if num_classes == 2:
                roc_auc = roc_auc_score(targets_tensor.numpy(), probs_tensor[:, 1].numpy())
            else:
                roc_auc = roc_auc_score(targets_tensor.numpy(), probs_tensor.numpy(), multi_class="ovr")
            metrics["roc_auc"] = float(roc_auc)
        except ValueError:
            pass

    cm = confusion_matrix(targets_tensor.numpy(), preds_tensor.numpy())
    metrics["confusion_matrix"] = cm.tolist()
    save_confusion_matrix(cm, dataset.classes, run_paths.plots / "confusion_matrix.png")
    if num_classes == 2 and unique_targets.numel() > 1:
        save_roc_curve(targets_tensor.numpy(), probs_tensor[:, 1].numpy(), run_paths.plots / "roc_curve.png")

    metrics_path = run_paths.logs / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics) + "\n")

    local_console.print(
        "[bold]Accuracy[/]: "
        f"{accuracy:.4f}"
        + " "
        + " ".join(
            f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float) and k != "accuracy"
        )
    )


def orchestrate(config_path: Path, *, mode: str) -> None:
    config = load_config(config_path)
    seed = config.get("seed")
    apply_seed(seed)

    models_cfg = config.get("models", {})
    if not isinstance(models_cfg, dict):
        raise TypeError("models section must be a mapping of name -> config")

    selection = config.get("selection")
    if selection is None:
        selected_models = list(models_cfg.keys())
    else:
        selected_models = [str(name) for name in selection]

    for model_name in selected_models:
        base_cfg = models_cfg.get(model_name)
        if base_cfg is None:
            console.print(f"[bold yellow]Skipping unknown model[/]: {model_name}")
            continue
        model_cfg = {"name": model_name, **base_cfg}
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_output = Path(model_cfg.get("output_dir", f"runs/{model_name}"))
        run_paths = ensure_run_dirs(base_output, timestamp)
        snapshot_config(run_paths, config=config, model_cfg=model_cfg)

        if mode == "training":
            run_training_job(config, model_cfg, run_paths)
        elif mode == "inference":
            run_inference_job(config_path=config_path, config=config, model_cfg=model_cfg, run_paths=run_paths)
        else:
            raise ValueError(f"Unknown mode '{mode}'")


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="DeepfakeDetection orchestrator")
    parser.add_argument("--mode", choices=["training", "inference"], default="training")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        default = "config/train.yaml" if args.mode == "training" else "config/inference.yaml"
        config_path = Path(default)

    orchestrate(config_path.resolve(), mode=args.mode)


if __name__ == "__main__":
    run_cli()
