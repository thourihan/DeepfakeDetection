"""Headless inference utility for DeepfakeDetection."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml
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
from torch.utils.data import DataLoader
from torchvision import datasets

from pipeline import DeepfakePipeline

console = Console()

def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataloader(
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


def save_confusion_matrix(cm: torch.Tensor, labels: list[str], path: Path) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm.numpy(), display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_roc_curve(y_true: torch.Tensor, y_scores: torch.Tensor, path: Path) -> None:
    from sklearn.metrics import RocCurveDisplay

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true.numpy(), y_scores.numpy(), ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batched inference on an ImageFolder split.")
    parser.add_argument("--config", default=Path("config/inference.yaml"), type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    models_cfg = config.get("models", {})
    if not isinstance(models_cfg, dict):
        console.print("[bold red]Config 'models' section must be a mapping[/]")
        raise SystemExit(1)

    model_cfg = models_cfg.get(args.model_name)
    if model_cfg is None:
        console.print(f"[bold red]Model {args.model_name} not found in config[/]")
        raise SystemExit(1)

    data_cfg = config.get("data", {})
    inference_cfg = model_cfg.get("inference", {})

    base_output = Path(model_cfg.get("output_dir", f"runs/{args.model_name}"))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.run_dir or (base_output / timestamp)
    run_dir = run_dir.resolve()
    logs_dir = run_dir / "logs"
    plots_dir = run_dir / "plots"
    for path in (run_dir, logs_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    weights_path = inference_cfg.get("weights")
    if weights_path is not None:
        weights_path = Path(weights_path)
        if not weights_path.is_absolute():
            weights_path = (args.config.parent / weights_path).resolve()
    else:
        weights_path = None

    device_str = os.environ.get("DD_DEVICE", config.get("device", "cuda"))
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        console.print("[bold yellow]⚠️  CUDA requested but unavailable[/]: using CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    pipeline = DeepfakePipeline(device=device)
    num_classes = int(model_cfg.get("num_classes", data_cfg.get("num_classes", 2)))
    pipeline.load_model(args.model_name, num_classes, weights_path)
    transforms = pipeline.build_transforms(
        image_size=inference_cfg.get("img_size", data_cfg.get("img_size")),
    )

    data_root_value = data_cfg.get("root", "data/DeepFakeFrames")
    data_root = Path(data_root_value).expanduser()
    if not data_root.is_absolute():
        data_root = (args.config.parent / data_root).resolve()

    split = inference_cfg.get("split", data_cfg.get("test_split", "test"))
    dataset_path = data_root / split
    if not dataset_path.exists():
        console.print(f"[bold red]Split not found:[/] {dataset_path}")
        raise SystemExit(1)

    dataset = datasets.ImageFolder(dataset_path, transform=transforms)
    if len(dataset) == 0:
        console.print(f"[bold yellow]No images found in[/] {dataset_path}")
        return

    batch_size = int(inference_cfg.get("batch_size", 64))
    num_workers = int(inference_cfg.get("num_workers", 4))
    dataloader = build_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[speed]}"),
        console=console,
    )

    all_probs: list[torch.Tensor] = []
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    start = perf_counter()
    images_seen = 0
    with progress:
        task = progress.add_task("inference", total=len(dataloader), speed="")
        for images, targets in dataloader:
            logits = pipeline.predict_batch(images)
            probs, preds = pipeline.postprocess(logits)
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
        "model": args.model_name,
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
                roc_auc = roc_auc_score(
                    targets_tensor.numpy(),
                    probs_tensor.numpy(),
                    multi_class="ovr",
                )
            metrics["roc_auc"] = float(roc_auc)
        except ValueError:
            pass

        cm = confusion_matrix(targets_tensor.numpy(), preds_tensor.numpy())
        cm_tensor = torch.from_numpy(cm)
        metrics["confusion_matrix"] = cm_tensor.tolist()
        cm_path = plots_dir / "confusion_matrix.png"
        save_confusion_matrix(cm_tensor, dataset.classes, cm_path)

        if num_classes == 2:
            roc_path = plots_dir / "roc_curve.png"
            save_roc_curve(targets_tensor, probs_tensor[:, 1], roc_path)
    else:
        cm = confusion_matrix(targets_tensor.numpy(), preds_tensor.numpy())
        cm_tensor = torch.from_numpy(cm)
        metrics["confusion_matrix"] = cm_tensor.tolist()
        cm_path = plots_dir / "confusion_matrix.png"
        save_confusion_matrix(cm_tensor, dataset.classes, cm_path)

    metrics_path = logs_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics) + "\n")

    console.print(
        f"[bold]Accuracy[/]: {accuracy:.4f} | "
        + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float) and k != "accuracy")
    )


if __name__ == "__main__":
    main()
