# ruff: noqa: INP001
"""Supervised training script for EfficientFormerV2-S1 on a Real/Fake dataset.

Expected layout (ImageFolder-compatible):
    DATA_ROOT/
        Train/{Real,Fake}/...
        Validation/{Real,Fake}/...

Training regime:
- Warm up by training only the classification head.
- Then unfreeze selected late-stage layers and fine-tune.
- Uses AMP on CUDA (if available) and channels_last for potential throughput gains.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import timm
import torch
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
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Configuration constants (could be moved to a config file).
DEFAULT_DATA_ROOT: Path = Path.home() / "code" / "DeepfakeDetection" / "data" / "Dataset"
MODEL_NAME: str = "efficientformerv2_s1"
EPOCHS: int = 5
BATCH_SIZE: int = 128
IMG_SIZE: int = 224
NUM_WORKERS: int = 8
DEFAULT_WEIGHTS_NAME: str = "efficientformer_v2_s1_weights.pth"
DEFAULT_LATEST_CKPT: str = "latest.ckpt"
DEFAULT_BEST_CKPT: str = "best.ckpt"

# Parameter name substrings to unfreeze after the head-only warmup.
UNFREEZE_KEYS: tuple[str, ...] = (
    "stages.3",
    "blocks.3",
    "layer4",
    "bneck",
    "features.6",
    "classifier",
    "head",
)
# ---------------------------------------------------------------------- #

console = Console()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: optim.Optimizer | None,
    scheduler: LRScheduler | None,
    epoch: int,
    phase: str,
    best_val_acc: float,
    best_epoch: int,
) -> None:
    """Persist the training state for warm starts orchestrated externally."""

    state: dict[str, Any] = {
        "epoch": epoch,
        "phase": phase,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
    }
    torch.save(state, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any] | None:
    """Load the checkpoint if present."""

    if not path.exists():
        return None
    return torch.load(path, map_location=device)


def get_loaders(
    data_root: Path,
    img_size: int,
    batch_size: int,
    *,
    train_split: str,
    val_split: str,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    """Build train/validation loaders with light augmentations on train."""
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

    train_ds = datasets.ImageFolder(data_root / train_split, transform=train_t)
    val_ds = datasets.ImageFolder(data_root / val_split, transform=val_t)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader_kwargs = loader_kwargs.copy()
    val_loader_kwargs.pop("prefetch_factor", None)
    if num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = 2
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **val_loader_kwargs,
    )
    return train_dl, val_dl


def evaluate(model: nn.Module, dl: DataLoader, device: str) -> float:
    """Compute top-1 accuracy on the given dataloader."""
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch_x, batch_y in dl:
            inputs = batch_x.to(device, non_blocking=True)
            targets = batch_y.to(device, non_blocking=True)
            logits = model(inputs)
            pred = logits.argmax(1)
            correct += (pred == targets).sum().item()
            total += targets.numel()
    return correct / max(1, total)


def train_one_epoch(  # noqa: PLR0913
    model: nn.Module,
    dl: DataLoader,
    opt: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: str,
    *,
    use_cuda_amp: bool,
    progress: Progress,
    task: TaskID,
) -> None:
    """Single-epoch training loop with AMP and live throughput reporting."""
    model.train()
    start = perf_counter()
    for i, (batch_x, batch_y) in enumerate(dl, 1):
        inputs = batch_x.to(device, non_blocking=True).to(
            memory_format=torch.channels_last,
        )
        targets = batch_y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_cuda_amp):
            loss = criterion(model(inputs), targets)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        # Report instantaneous images/sec and loss on the progress bar.
        elapsed = perf_counter() - start
        seen = min(i * dl.batch_size, len(dl.dataset))
        ips = seen / max(1e-6, elapsed)
        progress.update(
            task,
            advance=1,
            description=f"train | loss={loss.item():.4f} | {ips:.0f} img/s",
        )


def main() -> None:  # noqa: PLR0915
    """Entrypoint: device setup, data, warmup, fine-tune, save weights."""

    output_dir = Path(os.environ.get("DD_OUTPUT_DIR", ".")).expanduser().resolve()
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    for path in (output_dir, checkpoints_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    best_weights_path = checkpoints_dir / DEFAULT_WEIGHTS_NAME
    latest_ckpt_path = checkpoints_dir / DEFAULT_LATEST_CKPT
    best_ckpt_path = checkpoints_dir / DEFAULT_BEST_CKPT

    data_root = Path(os.environ.get("DD_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser()
    train_split = os.environ.get("DD_TRAIN_SPLIT", "Train")
    val_split = os.environ.get("DD_VAL_SPLIT", "Validation")

    img_size = int(os.environ.get("DD_IMG_SIZE", IMG_SIZE))
    batch_size = int(os.environ.get("DD_BATCH_SIZE", BATCH_SIZE))
    epochs = int(os.environ.get("DD_EPOCHS", EPOCHS))
    num_workers = int(os.environ.get("DD_NUM_WORKERS", NUM_WORKERS))
    num_classes = int(os.environ.get("DD_NUM_CLASSES", 2))

    seed_env = os.environ.get("DD_SEED")
    seeded = False
    if seed_env is not None:
        try:
            seed_value = int(seed_env)
        except ValueError:
            console.print(f"[bold yellow]Invalid DD_SEED[/]: {seed_env}")
        else:
            set_seed(seed_value)
            seeded = True
            console.print(f"[bold]Seeded[/]: {seed_value}")

    device_env = os.environ.get("DD_DEVICE")
    if device_env:
        device = device_env
        use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        if device.startswith("cuda") and not torch.cuda.is_available():
            console.print(
                "[bold yellow]⚠️  Requested CUDA device unavailable[/]: falling back to CPU",
            )
            device = "cpu"
            use_cuda = False
    else:
        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"

    if use_cuda:
        if not seeded:
            torch.backends.cudnn.benchmark = True
        console.print("[bold green]✅ CUDA available[/]: using GPU")
        console.print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        console.print("[bold yellow]⚠️  CUDA not available[/]: falling back to CPU")

    device_obj = torch.device(device)
    resume_requested = os.environ.get("DD_RESUME_AUTO") == "1"
    resume_state = load_checkpoint(latest_ckpt_path, device_obj) if resume_requested else None

    if not (data_root / train_split).exists() or not (data_root / val_split).exists():
        console.print(f"[bold red]Dataset not found under[/] {data_root}")
        console.print(
            f"Expected: {train_split}/ and {val_split}/ folders with class subdirectories",
        )
        raise SystemExit(1)

    train_dl, val_dl = get_loaders(
        data_root,
        img_size,
        batch_size,
        train_split=train_split,
        val_split=val_split,
        num_workers=num_workers,
    )
    console.print(
        f"[bold]Data[/]: train={len(train_dl.dataset)} | val={len(val_dl.dataset)} | "
        f"bs={batch_size} | steps/epoch={len(train_dl)}",
    )

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    model.to(memory_format=torch.channels_last)
    model = model.to(device_obj)

    if resume_state is not None and "model" in resume_state:
        model.load_state_dict(resume_state["model"])
        console.print("[bold]Loaded checkpoint weights from latest.ckpt[/]")

    criterion: nn.Module = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=use_cuda)

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

    best_val_acc = float(resume_state.get("best_val_acc", -1.0)) if resume_state else -1.0
    best_epoch = int(resume_state.get("best_epoch", -1)) if resume_state else -1

    warmup_completed = False
    if resume_state is not None and resume_state.get("phase") in {"warmup", "finetune"}:
        warmup_completed = True

    with progress:
        if not warmup_completed:
            for name, param in model.named_parameters():
                param.requires_grad = ("classifier" in name) or ("head" in name)

            head_params = [p for p in model.parameters() if p.requires_grad]
            opt = optim.AdamW(head_params, lr=3e-4, weight_decay=5e-2)

            warm_task = progress.add_task("warmup", total=len(train_dl), extra="")
            console.print("[bold]Warmup (head only)[/]")
            train_one_epoch(
                model=model,
                dl=train_dl,
                opt=opt,
                scaler=scaler,
                criterion=criterion,
                device=device,
                use_cuda_amp=use_cuda,
                progress=progress,
                task=warm_task,
            )

            warmup_acc = evaluate(model, val_dl, device)
            console.print(f"[bold cyan]warmup[/] | val_acc={warmup_acc:.4f}")
            best_val_acc = warmup_acc
            best_epoch = 0
            save_checkpoint(
                latest_ckpt_path,
                model=model,
                optimizer=opt,
                scheduler=None,
                epoch=0,
                phase="warmup",
                best_val_acc=best_val_acc,
                best_epoch=best_epoch,
            )
            save_checkpoint(
                best_ckpt_path,
                model=model,
                optimizer=opt,
                scheduler=None,
                epoch=0,
                phase="warmup",
                best_val_acc=best_val_acc,
                best_epoch=best_epoch,
            )
            torch.save(model.state_dict(), best_weights_path)
        else:
            console.print("[bold]Skipping warmup[/]: restored from checkpoint")

        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(key in name for key in UNFREEZE_KEYS):
                param.requires_grad = True

        opt = optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=1e-4,
            weight_decay=5e-2,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - 1))

        start_epoch = 1
        if resume_state is not None and resume_state.get("phase") == "finetune":
            start_epoch = int(resume_state.get("epoch", 0)) + 1
            opt_state = resume_state.get("optimizer")
            if opt_state:
                opt.load_state_dict(opt_state)
            sched_state = resume_state.get("scheduler")
            if sched_state:
                scheduler.load_state_dict(sched_state)
            console.print(f"[bold]Resuming fine-tune from epoch[/] {start_epoch}")

        if start_epoch > EPOCHS:
            console.print("[bold yellow]Nothing to do[/]: already finished training")
            return

        for epoch in range(start_epoch, EPOCHS + 1):
            task = progress.add_task(f"epoch {epoch}", total=len(train_dl), extra="")
            train_one_epoch(
                model=model,
                dl=train_dl,
                opt=opt,
                scaler=scaler,
                criterion=criterion,
                device=device,
                use_cuda_amp=use_cuda,
                progress=progress,
                task=task,
            )
            scheduler.step()

            acc = evaluate(model, val_dl, device)
            console.print(f"[bold cyan]epoch {epoch}[/] | val_acc={acc:.4f}")

            if acc > best_val_acc + 1e-4:
                best_val_acc = acc
                best_epoch = epoch
                torch.save(model.state_dict(), best_weights_path)
                save_checkpoint(
                    best_ckpt_path,
                    model=model,
                    optimizer=opt,
                    scheduler=scheduler,
                    epoch=epoch,
                    phase="finetune",
                    best_val_acc=best_val_acc,
                    best_epoch=best_epoch,
                )
                console.print(
                    f"[bold green]new best[/] val_acc={best_val_acc:.4f} → saved {best_weights_path.name}",
                )

            save_checkpoint(
                latest_ckpt_path,
                model=model,
                optimizer=opt,
                scheduler=scheduler,
                epoch=epoch,
                phase="finetune",
                best_val_acc=best_val_acc,
                best_epoch=best_epoch,
            )

    console.print(f"[bold green]Best weights saved →[/] {best_weights_path.resolve()}")
    console.print(f"[bold green]Best checkpoint saved →[/] {best_ckpt_path.resolve()}")



if __name__ == "__main__":
    main()
