# ruff: noqa: INP001
"""Supervised training script for EfficientNet-B3 on a Real/Fake dataset.

Expected layout (ImageFolder-compatible):
    DATA_ROOT/
        Train/{Real,Fake}/...
        Validation/{Real,Fake}/...

Training regime:
- Warm up by training only the classification head.
- Then fine-tune the whole network with a lower LR.
- Uses AMP on CUDA (if available) and channels_last for throughput.

This script saves:
- Best weights (by Validation accuracy): EfficientNetModel.pth
- A full training checkpoint (model+optimizer+scheduler+epoch) for resume:
  checkpoints/efficientnet_best.ckpt
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
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

# ---------------------------- Config --------------------------------- #

# Adjust for your environment if needed.
DEFAULT_DATA_ROOT: Path = Path.home() / "code" / "DeepfakeDetection" / "data" / "Dataset"

# Training hyperparameters.
EPOCHS: int = 25
BATCH_SIZE: int = 64
IMG_SIZE: int = 224
NUM_WORKERS: int = 8

# LRs/WDs for the two phases.
HEAD_LR: float = 3e-4
HEAD_WD: float = 5e-2
FT_LR: float = 1e-4
FT_WD: float = 5e-2

# Early stopping by Validation accuracy.
PATIENCE: int = 4  # epochs without improvement

# Default output paths (overridden by DD_OUTPUT_DIR when orchestrated).
DEFAULT_WEIGHTS_NAME: str = "efficientnet_weights.pth"
DEFAULT_LATEST_CKPT: str = "latest.ckpt"
DEFAULT_BEST_CKPT: str = "best.ckpt"

# Optional: gradient accumulation (helps laptops with limited VRAM)
FT_BATCH_SIZE: int = 32  # micro-batch size during fine-tune
EFFECTIVE_BATCH: int = 128  # desired effective batch
ACCUM_STEPS: int = max(1, EFFECTIVE_BATCH // FT_BATCH_SIZE)

# --------------------------------------------------------------------- #

console = Console()


@dataclass(frozen=True)
class EvalResult:
    """Container for evaluation metrics."""

    acc: float
    loss: float
    total: int
    correct: int


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
    epochs_no_improve: int,
) -> None:
    """Persist the latest training state so orchestration can resume runs."""

    state: dict[str, Any] = {
        "epoch": epoch,
        "phase": phase,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "epochs_no_improve": epochs_no_improve,
    }
    torch.save(state, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any] | None:
    """Load a checkpoint if it exists."""

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
    """Build train/validation loaders. EfficientNet expects ImageNet norm."""
    train_t = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value=0,
            ),
        ],
    )
    val_t = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
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


def evaluate(
    model: nn.Module,
    dl: DataLoader,
    device: str,
    criterion: nn.Module,
) -> EvalResult:
    """Compute top-1 accuracy and mean loss."""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for batch_x, batch_y in dl:
            inputs = batch_x.to(device, non_blocking=True).to(
                memory_format=torch.channels_last,
            )
            targets = batch_y.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            pred = logits.argmax(1)
            correct += (pred == targets).sum().item()
            total += targets.numel()
            loss_sum += float(loss.item()) * targets.size(0)
    acc = correct / max(1, total)
    mean_loss = loss_sum / max(1, total)
    return EvalResult(acc=acc, loss=mean_loss, total=total, correct=correct)


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
    accum_steps: int = 1,
) -> float:
    """Single-epoch training loop with AMP, accumulation, and live throughput reporting.

    Returns mean train loss.
    """
    model.train()
    start = perf_counter()
    opt.zero_grad(set_to_none=True)

    loss_sum = 0.0
    seen_total = 0
    pending_steps = 0

    for i, (batch_x, batch_y) in enumerate(dl, 1):
        inputs = batch_x.to(device, non_blocking=True).to(
            memory_format=torch.channels_last,
        )
        targets = batch_y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_cuda_amp):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if accum_steps > 1:
                loss = loss / accum_steps

        scaler.scale(loss).backward()
        pending_steps += 1

        if pending_steps == accum_steps:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            pending_steps = 0

        # Stats
        bsz = targets.size(0)
        seen_total += bsz
        loss_sum += float(loss.item()) * bsz * (max(1, accum_steps))

        # Progress bar: show unscaled loss and images/sec
        elapsed = perf_counter() - start
        seen = min(i * dl.batch_size, len(dl.dataset))
        ips = seen / max(1e-6, elapsed)
        shown_loss = float(loss.item() * (max(1, accum_steps)))
        progress.update(
            task,
            advance=1,
            description=f"train | loss={shown_loss:.4f} | {ips:.0f} img/s",
        )

    # Flush any leftover grads
    if pending_steps > 0:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    return loss_sum / max(1, seen_total)


def save_best(
    *,
    model: nn.Module,
    opt: optim.Optimizer,
    sched: LRScheduler | None,
    epoch: int,
    best_weights_path: Path,
    best_ckpt_path: Path,
    best_val_acc: float,
    best_epoch: int,
    epochs_no_improve: int,
) -> None:
    """Persist best weights and a full resume checkpoint."""
    torch.save(model.state_dict(), best_weights_path)
    ckpt = {
        "epoch": epoch,
        "phase": "finetune",
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched is not None else None,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "epochs_no_improve": epochs_no_improve,
    }
    torch.save(ckpt, best_ckpt_path)


def main() -> None:  # noqa: PLR0915
    """Entrypoint: data, model, warmup, fine-tune, early stop, save best."""

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
        console.print("[bold yellow]⚠️  CUDA not available[/]: using CPU")

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

    model = EfficientNet.from_pretrained("efficientnet-b3")
    in_features = model._fc.in_features  # noqa: SLF001
    model._fc = nn.Linear(in_features, num_classes)  # noqa: SLF001

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
    epochs_no_improve = int(resume_state.get("epochs_no_improve", 0)) if resume_state else 0

    warmup_completed = False
    if resume_state is not None and resume_state.get("phase") in {"warmup", "finetune"}:
        warmup_completed = True

    with progress:
        if not warmup_completed:
            for param in model.parameters():
                param.requires_grad = False
            for name, param in model.named_parameters():
                if "_fc" in name:
                    param.requires_grad = True

            head_params = [p for p in model.parameters() if p.requires_grad]
            opt = optim.AdamW(head_params, lr=HEAD_LR, weight_decay=HEAD_WD)

            warm_task = progress.add_task(
                "warmup (head only)",
                total=len(train_dl),
                extra="",
            )
            console.print("[bold]Warmup (head only)[/]")
            _ = train_one_epoch(
                model=model,
                dl=train_dl,
                opt=opt,
                scaler=scaler,
                criterion=criterion,
                device=device,
                use_cuda_amp=use_cuda,
                progress=progress,
                task=warm_task,
                accum_steps=1,
            )

            res = evaluate(model, val_dl, device, criterion)
            console.print(
                f"[bold cyan]warmup[/] | val_acc={res.acc:.4f} | val_loss={res.loss:.4f} "
                f"({res.correct}/{res.total})",
            )
            best_val_acc = res.acc
            best_epoch = 0
            epochs_no_improve = 0
            save_checkpoint(
                latest_ckpt_path,
                model=model,
                optimizer=opt,
                scheduler=None,
                epoch=0,
                phase="warmup",
                best_val_acc=best_val_acc,
                best_epoch=best_epoch,
                epochs_no_improve=epochs_no_improve,
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
                epochs_no_improve=epochs_no_improve,
            )
            torch.save(model.state_dict(), best_weights_path)
        else:
            console.print("[bold]Skipping warmup[/]: restored from checkpoint")

        for param in model.parameters():
            param.requires_grad = True

        console.print(
            f"[bold]Fine-tune[/]: bs={FT_BATCH_SIZE}, accum_steps={ACCUM_STEPS} "
            f"(effective ≈ {FT_BATCH_SIZE * ACCUM_STEPS})",
        )
        train_dataset = train_dl.dataset
        ft_loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            ft_loader_kwargs["prefetch_factor"] = 2
        train_dl_ft = DataLoader(
            train_dataset,
            batch_size=FT_BATCH_SIZE,
            shuffle=True,
            **ft_loader_kwargs,
        )

        opt = optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=FT_LR,
            weight_decay=FT_WD,
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

        if start_epoch > epochs:
            console.print("[bold yellow]Nothing to do[/]: already finished training")
            return

        for epoch in range(start_epoch, epochs + 1):
            task = progress.add_task(f"epoch {epoch}", total=len(train_dl_ft), extra="")
            train_loss = train_one_epoch(
                model=model,
                dl=train_dl_ft,
                opt=opt,
                scaler=scaler,
                criterion=criterion,
                device=device,
                use_cuda_amp=use_cuda,
                progress=progress,
                task=task,
                accum_steps=ACCUM_STEPS,
            )
            scheduler.step()

            res = evaluate(model, val_dl, device, criterion)
            console.print(
                f"[bold cyan]epoch {epoch}[/] | train_loss={train_loss:.4f} "
                f"| val_loss={res.loss:.4f} | val_acc={res.acc:.4f} "
                f"({res.correct}/{res.total}) | lr={scheduler.get_last_lr()[0]:.2e}",
            )

            improved = res.acc > best_val_acc + 1e-4
            if improved:
                best_val_acc = res.acc
                best_epoch = epoch
                epochs_no_improve = 0
                save_best(
                    model=model,
                    opt=opt,
                    sched=scheduler,
                    epoch=epoch,
                    best_weights_path=best_weights_path,
                    best_ckpt_path=best_ckpt_path,
                    best_val_acc=best_val_acc,
                    best_epoch=best_epoch,
                    epochs_no_improve=epochs_no_improve,
                )
                console.print(
                    f"[bold green]new best[/] val_acc={best_val_acc:.4f} "
                    f"(epoch {best_epoch}) → saved {best_weights_path.name}",
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    console.print(
                        f"[bold yellow]Early stopping[/]: no improvement for "
                        f"{PATIENCE} epoch(s). Best at epoch {best_epoch} "
                        f"with val_acc={best_val_acc:.4f}.",
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
                        epochs_no_improve=epochs_no_improve,
                    )
                    break

            save_checkpoint(
                latest_ckpt_path,
                model=model,
                optimizer=opt,
                scheduler=scheduler,
                epoch=epoch,
                phase="finetune",
                best_val_acc=best_val_acc,
                best_epoch=best_epoch,
                epochs_no_improve=epochs_no_improve,
            )

    console.print(f"[bold green]Best weights saved →[/] {best_weights_path.resolve()}")
    console.print(f"[bold green]Best checkpoint saved →[/] {best_ckpt_path.resolve()}")



if __name__ == "__main__":
    main()
