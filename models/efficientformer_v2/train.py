from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Tuple

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CONSTANTS
DATA_ROOT: Path = Path.home() / "code" / "DeepfakeDetection" / "data" / "Dataset"
MODEL_NAME: str = "efficientformerv2_s1"
EPOCHS: int = 5
BATCH_SIZE: int = 128
IMG_SIZE: int = 224
NUM_WORKERS: int = 8 
OUTPUT_WEIGHTS: Path = Path("EfficientFormerV2_S1.pth")

# Layers to unfreeze after warmup
UNFREEZE_KEYS: tuple[str, ...] = (
    "stages.3",
    "blocks.3",
    "layer4",
    "bneck",
    "features.6",
    "classifier",
    "head",
)

console = Console()


def get_loaders(data_root: Path, img_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_t = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
        ],
    )
    val_t = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ],
    )
    train_ds = datasets.ImageFolder(data_root / "Train", transform=train_t)
    val_ds = datasets.ImageFolder(data_root / "Validation", transform=val_t)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    return train_dl, val_dl


def evaluate(model: nn.Module, dl: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


def train_one_epoch(
    model: nn.Module,
    dl: DataLoader,
    opt: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: str,
    use_cuda_amp: bool,
    progress: Progress,
    task: TaskID,
) -> None:
    model.train()
    start = perf_counter()
    for i, (x, y) in enumerate(dl, 1):
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_cuda_amp):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        # live imgs/sec
        elapsed = perf_counter() - start
        seen = min(i * dl.batch_size, len(dl.dataset))
        ips = seen / max(1e-6, elapsed)
        progress.update(task, advance=1, description=f"train | loss={loss.item():.4f} | {ips:.0f} img/s")


def main() -> None:
    # Device info
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    if use_cuda:
        console.print("[bold green]✅ CUDA available[/]: using GPU")
        console.print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        console.print("[bold yellow]⚠️  CUDA not available[/]: falling back to CPU")

    # Data presence
    if not (DATA_ROOT / "Train").exists() or not (DATA_ROOT / "Validation").exists():
        console.print(f"[bold red]Dataset not found under[/] {DATA_ROOT}")
        console.print("Expected: Dataset/Train/{Real,Fake} and Dataset/Validation/{Real,Fake}")
        raise SystemExit(1)

    train_dl, val_dl = get_loaders(DATA_ROOT, IMG_SIZE, BATCH_SIZE)
    console.print(
        f"[bold]Data[/]: train={len(train_dl.dataset)} | val={len(val_dl.dataset)} | "
        f"bs={BATCH_SIZE} | steps/epoch={len(train_dl)}",
    )

    # Model
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=2)
    model.to(memory_format=torch.channels_last)
    model = model.to(device)

    # Freeze everything and train only the head first
    for name, param in model.named_parameters():
        param.requires_grad = ("classifier" in name) or ("head" in name)

    criterion: nn.Module = nn.CrossEntropyLoss(label_smoothing=0.1)
    head_params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(head_params, lr=3e-4, weight_decay=5e-2)
    scaler = torch.amp.GradScaler(enabled=use_cuda)

    # Rich progress setup
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[extra]}"),
        console=console,
        transient=False,
    )

    with progress:
        # Warmup epoch (head only)
        warmup_task = progress.add_task(
            "warmup",
            total=len(train_dl),
            extra="",
        )
        console.print("[bold]Warmup (head only)[/]")
        start = perf_counter()

        model.train()
        for i, (x, y) in enumerate(train_dl, 1):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            elapsed = perf_counter() - start
            seen = min(i * train_dl.batch_size, len(train_dl.dataset))
            ips = seen / max(1e-6, elapsed)
            progress.update(warmup_task, advance=1, description=f"warmup | loss={loss.item():.4f}", extra=f"{ips:.0f} img/s")

        # Unfreeze last stage and head, train remaining epochs
        for p in model.parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            if any(key in name for key in UNFREEZE_KEYS):
                p.requires_grad = True

        opt = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4, weight_decay=5e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, EPOCHS - 1))

        for epoch in range(1, EPOCHS):
            task = progress.add_task(f"epoch {epoch}", total=len(train_dl), extra="")
            train_one_epoch(model, train_dl, opt, scaler, criterion, device, use_cuda, progress, task)
            scheduler.step()
            acc = evaluate(model, val_dl, device)
            console.print(f"[bold cyan]epoch {epoch}[/] | val_acc={acc:.4f}")

    torch.save(model.state_dict(), OUTPUT_WEIGHTS)
    console.print(f"[bold green]Saved weights →[/] {OUTPUT_WEIGHTS.resolve()}")


if __name__ == "__main__":
    main()
