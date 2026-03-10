from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from orchestration.model_factory import ModelBuildConfig, build_model, infer_default_image_size
from orchestration.runtime import CheckpointManager, RunContext

console = Console()


@dataclass
class TrainerState:
    epoch: int = 0
    best_metric: float = float("-inf")
    best_epoch: int = 0


@dataclass
class ModelRunConfig:
    name: str
    model: ModelBuildConfig
    training: Any
    inference: Any
    transforms: dict[str, Any]
    num_classes: int
    model_fingerprint: dict[str, Any]


# ---------------------------------------------------------------------------
# Data pipeline helpers
# ---------------------------------------------------------------------------

def build_transforms(image_size: int, config: dict[str, Any] | None, *, train: bool) -> transforms.Compose:
    defaults = {
        "ensure_rgb": True,
        "resize": True,
        "center_crop": not train,
        "random_resized_crop": train,
        "random_flip": train,
        "to_tensor": True,
        "normalize": True,
    }
    resolved = dict(defaults)
    if config:
        for key, value in config.items():
            resolved[key] = bool(value)

    ops: list[Any] = []
    if resolved.get("ensure_rgb"):
        ops.append(transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img))
    if resolved.get("random_resized_crop"):
        ops.append(transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)))
    elif resolved.get("resize"):
        ops.append(transforms.Resize(image_size + 32))
    if resolved.get("center_crop"):
        ops.append(transforms.CenterCrop(image_size))
    if resolved.get("random_flip"):
        ops.append(transforms.RandomHorizontalFlip())
    if resolved.get("color_jitter"):
        ops.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.05))
    if resolved.get("to_tensor"):
        ops.append(transforms.ToTensor())
    if resolved.get("normalize"):
        ops.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    return transforms.Compose(ops)


def build_dataloaders(
    *,
    data_root: Path,
    train_split: str,
    val_split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    transforms_cfg: dict[str, Any] | None,
    num_classes: int,
) -> tuple[DataLoader, DataLoader]:
    train_tf = build_transforms(image_size, (transforms_cfg or {}).get("train"), train=True)
    eval_tf = build_transforms(image_size, (transforms_cfg or {}).get("eval"), train=False)

    train_ds = datasets.ImageFolder(data_root / train_split, transform=train_tf)
    val_ds = datasets.ImageFolder(data_root / val_split, transform=eval_tf)

    for split_name, dataset in ((train_split, train_ds), (val_split, val_ds)):
        classes = getattr(dataset, "classes", None)
        if classes is not None and len(classes) != num_classes:
            raise ValueError(
                f"Split '{split_name}' exposes {len(classes)} classes but config expects {num_classes}. "
                "Update config.data.num_classes to match the dataset."
            )

    common_loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader_kwargs)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    *,
    run_context: RunContext,
    data_cfg: Any,
    model_cfg: ModelRunConfig,
) -> dict[str, Any]:
    if run_context.seed is not None:
        torch.manual_seed(run_context.seed)
    device = run_context.device
    use_cuda = device.type == "cuda"

    training_cfg = model_cfg.training
    image_size = training_cfg.img_size or infer_default_image_size(model_cfg.model)
    early_stop_patience: int | None = getattr(training_cfg, "early_stop_patience", None)

    train_loader, val_loader = build_dataloaders(
        data_root=Path(data_cfg.root).expanduser(),
        train_split=data_cfg.train_split,
        val_split=data_cfg.val_split,
        image_size=image_size,
        batch_size=training_cfg.batch_size,
        num_workers=training_cfg.num_workers,
        transforms_cfg=model_cfg.transforms,
        num_classes=model_cfg.num_classes,
    )

    model = build_model(model_cfg.model, model_cfg.num_classes)
    if use_cuda:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=training_cfg.lr, weight_decay=training_cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg.epochs)
    scaler = torch.amp.GradScaler(enabled=use_cuda)

    ckpt = CheckpointManager(run_context, fingerprint=model_cfg.model_fingerprint)
    state = TrainerState()
    start_epoch = 0
    resume_flag = str(training_cfg.resume).lower() if training_cfg.resume is not None else ""
    if resume_flag in {"1", "true", "auto", "yes"}:
        loaded = ckpt.load(model=model, optimizer=optimizer, scheduler=scheduler, strict=True)
        if loaded:
            start_epoch = int(loaded.get("epoch", 0)) + 1
            state.best_metric = float(loaded.get("best_metric", state.best_metric))
            state.best_epoch = int(loaded.get("best_epoch", state.best_epoch))
            console.print(f"[green]✓ Resumed from[/] {ckpt.latest_path}")

    epochs_no_improve = 0
    for epoch in range(start_epoch, training_cfg.epochs):
        state.epoch = epoch
        train_loss = _run_epoch(
            model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            accum_steps=training_cfg.accum_steps or 1,
        )
        val_metrics = _evaluate(model, loader=val_loader, device=device)
        scheduler.step()

        is_best = val_metrics["accuracy"] > state.best_metric
        if is_best:
            state.best_metric = val_metrics["accuracy"]
            state.best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        ckpt.save(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=state.best_metric,
            best_epoch=state.best_epoch,
            extra={"val_loss": val_metrics["loss"], "train_loss": train_loss},
            is_best=is_best,
        )

        console.print(
            f"[bold]{model_cfg.name}[/] epoch {epoch+1}/{training_cfg.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
            + (f" [yellow](no improvement for {epochs_no_improve})[/]" if epochs_no_improve else "")
        )

        if early_stop_patience is not None and epochs_no_improve >= early_stop_patience:
            console.print(
                f"[yellow]Early stopping[/]: no improvement for {early_stop_patience} epoch(s). "
                f"Best val_acc={state.best_metric:.4f} at epoch {state.best_epoch + 1}."
            )
            break

    return {
        "best_accuracy": state.best_metric,
        "best_epoch": state.best_epoch,
    }


def _run_epoch(
    model: nn.Module,
    *,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    accum_steps: int,
) -> float:
    model.train()
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    use_cuda = device.type == "cuda"

    optimizer.zero_grad(set_to_none=True)
    pending = 0
    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        if use_cuda:
            images = images.to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            logits = model(images)
            loss = criterion(logits, targets) / accum_steps

        scaler.scale(loss).backward()
        pending += 1

        if pending == accum_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pending = 0

        running_loss += loss.item() * accum_steps

    # Flush any remaining gradients if loader length is not divisible by accum_steps.
    if pending > 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return running_loss / len(loader)


def _evaluate(model: nn.Module, *, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return {"loss": total_loss / len(loader), "accuracy": correct / max(total, 1)}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    *,
    run_context: RunContext,
    data_cfg: Any,
    model_cfg: ModelRunConfig,
    weights_path: Path | None,
) -> dict[str, Any]:
    device = run_context.device
    image_size = (
        model_cfg.inference.img_size
        or model_cfg.training.img_size
        or infer_default_image_size(model_cfg.model)
    )
    transforms_cfg = model_cfg.transforms
    eval_tf = build_transforms(image_size, (transforms_cfg or {}).get("eval"), train=False)

    split = model_cfg.inference.split or data_cfg.test_split
    dataset = datasets.ImageFolder(Path(data_cfg.root) / split, transform=eval_tf)
    if len(dataset) == 0:
        raise FileNotFoundError(f"No images found under {dataset.root}")

    num_workers = model_cfg.inference.num_workers or model_cfg.training.num_workers
    loader = DataLoader(
        dataset,
        batch_size=model_cfg.inference.batch_size or model_cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    model = build_model(model_cfg.model, model_cfg.num_classes).to(device)
    ckpt = CheckpointManager(run_context, fingerprint=model_cfg.model_fingerprint)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
    else:
        ckpt.load(model=model, prefer_best=True, strict=True)

    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / max(total, 1)
    console.print(
        f"[bold]{model_cfg.name}[/] split={split} samples={total} accuracy={accuracy:.4f}"
    )
    return {"accuracy": accuracy, "split": split}


__all__ = ["ModelRunConfig", "build_transforms", "train_model", "run_inference"]
