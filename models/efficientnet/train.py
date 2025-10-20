from __future__ import annotations
"""
Supervised training script for EfficientNet-B3 on a Real/Fake dataset.

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

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------- Config --------------------------------- #

# Adjust for your environment if needed.
DATA_ROOT: Path = Path.home() / "code" / "DeepfakeDetection" / "data" / "Dataset"

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

# Output paths.
BEST_WEIGHTS: Path = Path("EfficientNetModel.pth")
BEST_CKPT: Path = Path("checkpoints") / "efficientnet_best.ckpt"
BEST_CKPT.parent.mkdir(parents=True, exist_ok=True)

# Optional: gradient accumulation (helps laptops with limited VRAM)
FT_BATCH_SIZE: int = 32          # micro-batch size during fine-tune
EFFECTIVE_BATCH: int = 128       # desired effective batch
ACCUM_STEPS: int = max(1, EFFECTIVE_BATCH // FT_BATCH_SIZE)

# --------------------------------------------------------------------- #

console = Console()


@dataclass(frozen=True)
class EvalResult:
    acc: float
    loss: float
    total: int
    correct: int


def get_loaders(
    data_root: Path, img_size: int, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """Build train/validation loaders. EfficientNet expects ImageNet norm."""
    train_t = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ]
    )
    val_t = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
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
        prefetch_factor=2,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2,
    )
    return train_dl, val_dl


def evaluate(model: nn.Module, dl: DataLoader, device: str, criterion: nn.Module) -> EvalResult:
    """Compute top-1 accuracy and mean loss."""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for x, y in dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            loss_sum += float(loss.item()) * y.size(0)
    acc = correct / max(1, total)
    mean_loss = loss_sum / max(1, total)
    return EvalResult(acc=acc, loss=mean_loss, total=total, correct=correct)


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
    accum_steps: int = 1,
) -> float:
    """Single-epoch training loop with AMP, accumulation, live throughput reporting. Returns mean train loss."""
    model.train()
    start = perf_counter()
    opt.zero_grad(set_to_none=True)

    loss_sum = 0.0
    seen_total = 0
    pending_steps = 0

    for i, (x, y) in enumerate(dl, 1):
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_cuda_amp):
            logits = model(x)
            loss = criterion(logits, y)
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
        bsz = y.size(0)
        seen_total += bsz
        loss_sum += float(loss.item()) * bsz * (accum_steps if accum_steps > 1 else 1)

        # Progress bar: show unscaled loss and images/sec
        elapsed = perf_counter() - start
        seen = min(i * dl.batch_size, len(dl.dataset))
        ips = seen / max(1e-6, elapsed)
        shown_loss = float(loss.item() * (accum_steps if accum_steps > 1 else 1))
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

    mean_train_loss = loss_sum / max(1, seen_total)
    return mean_train_loss


def save_best(
    model: nn.Module,
    opt: optim.Optimizer,
    sched: optim.lr_scheduler._LRScheduler | None,
    epoch: int,
) -> None:
    """Persist best weights and a full resume checkpoint."""
    torch.save(model.state_dict(), BEST_WEIGHTS)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched is not None else None,
    }
    torch.save(ckpt, BEST_CKPT)


def main() -> None:
    """Entrypoint: data, model, warmup, fine-tune, early stop, save best."""
    # Device
    torch.backends.cudnn.benchmark = True  # faster kernels on stable shapes
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    if use_cuda:
        console.print("[bold green]✅ CUDA available[/]: using GPU")
        console.print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        console.print("[bold yellow]⚠️  CUDA not available[/]: using CPU")

    # Dataset presence check.
    if not (DATA_ROOT / "Train").exists() or not (DATA_ROOT / "Validation").exists():
        console.print(f"[bold red]Dataset not found under[/] {DATA_ROOT}")
        console.print("Expected: Dataset/Train/{Real,Fake} and Dataset/Validation/{Real,Fake}")
        raise SystemExit(1)

    # Data loaders
    train_dl, val_dl = get_loaders(DATA_ROOT, IMG_SIZE, BATCH_SIZE)
    console.print(
        f"[bold]Data[/]: train={len(train_dl.dataset)} | val={len(val_dl.dataset)} | "
        f"bs={BATCH_SIZE} | steps/epoch={len(train_dl)}"
    )

    # Model: EfficientNet-B3 (ImageNet-pretrained), 2-class head.
    model = EfficientNet.from_pretrained("efficientnet-b3")
    in_features = model._fc.in_features  # type: ignore[attr-defined]
    model._fc = nn.Linear(in_features, 2)  # type: ignore[attr-defined]
    # Optional: bump dropout a bit; EfficientNet already uses dropout internally.
    # model._dropout = nn.Dropout(p=0.5)  # uncomment if you want extra regularization

    model.to(memory_format=torch.channels_last)
    model = model.to(device)

    # Loss, scaler
    criterion: nn.Module = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=use_cuda)

    # Progress UI.
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

    best_val_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    with progress:
        # ---------------------- Phase 1: head warmup ---------------------- #
        for p in model.parameters():
            p.requires_grad = False
        for n, p in model.named_parameters():
            if "_fc" in n:
                p.requires_grad = True

        head_params = [p for p in model.parameters() if p.requires_grad]
        opt = optim.AdamW(head_params, lr=HEAD_LR, weight_decay=HEAD_WD)

        warm_task = progress.add_task("warmup (head only)", total=len(train_dl), extra="")
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

        # Validate after warmup
        res = evaluate(model, val_dl, device, criterion)
        console.print(
            f"[bold cyan]warmup[/] | val_acc={res.acc:.4f} | val_loss={res.loss:.4f} "
            f"({res.correct}/{res.total})"
        )
        best_val_acc = res.acc
        best_epoch = 0
        save_best(model, opt, sched=None, epoch=best_epoch)

        # -------------------- Phase 2: full fine-tune --------------------- #
        for p in model.parameters():
            p.requires_grad = True

        # Smaller fine-tune micro-batch for laptop VRAM + gradient accumulation
        console.print(
            f"[bold]Fine-tune[/]: bs={FT_BATCH_SIZE}, accum_steps={ACCUM_STEPS} "
            f"(effective ≈ {FT_BATCH_SIZE * ACCUM_STEPS})"
        )
        train_dl_ft = DataLoader(
            train_dl.dataset,
            batch_size=FT_BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
            prefetch_factor=2,
        )

        opt = optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=FT_LR,
            weight_decay=FT_WD,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, EPOCHS - 1))

        for epoch in range(1, EPOCHS + 1):
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
                f"({res.correct}/{res.total}) | lr={scheduler.get_last_lr()[0]:.2e}"
            )

            improved = res.acc > best_val_acc + 1e-4
            if improved:
                best_val_acc = res.acc
                best_epoch = epoch
                epochs_no_improve = 0
                save_best(model, opt, scheduler, epoch)
                console.print(
                    f"[bold green]new best[/] val_acc={best_val_acc:.4f} (epoch {best_epoch}) "
                    f"→ saved {BEST_WEIGHTS.name}"
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    console.print(
                        f"[bold yellow]Early stopping[/]: no improvement for "
                        f"{PATIENCE} epoch(s). Best at epoch {best_epoch} "
                        f"with val_acc={best_val_acc:.4f}."
                    )
                    break

    console.print(f"[bold green]Best weights saved →[/] {BEST_WEIGHTS.resolve()}")
    console.print(f"[bold green]Best checkpoint saved →[/] {BEST_CKPT.resolve()}")


if __name__ == "__main__":
    main()
