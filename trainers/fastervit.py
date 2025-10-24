# ruff: noqa: INP001
"""Supervised training script for FasterViT-2-224 on a Real/Fake dataset.

Expected layout (ImageFolder-compatible):
    DATA_ROOT/
        Train/{Real,Fake}/...
        Validation/{Real,Fake}/...

Training regime:
- Warm up by training only the classification head.
- Then fine-tune the whole network with a lower LR.
- Uses AMP on CUDA (if available) and channels_last for throughput.

This script saves:
- Best weights (by Validation accuracy): FasterVitModel.pth
- A full training checkpoint (model+optimizer+scheduler+epoch) for resume:
  checkpoints/fastervit_best.ckpt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from fastervit import create_model
from PIL import Image
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from train_env import (
    apply_seed,
    create_console,
    env_int,
    env_path,
    env_str,
    maybe_load_checkpoint,
    prepare_training_environment,
    require_num_classes,
    save_best_checkpoint,
    save_latest_checkpoint,
)

# ---------------------------- Config --------------------------------- #

# Adjust for your environment if needed.
DATA_ROOT: Path = Path.home() / "code" / "DeepfakeDetection" / "data" / "Dataset"
MODEL_NAME: str = "faster_vit_2_224"

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

# Output filenames (paths resolved at runtime via :mod:`train_env`).
BEST_WEIGHTS_NAME: str = "FasterVitModel.pth"
BEST_CKPT_NAME: str = "best.ckpt"
LATEST_CKPT_NAME: str = "latest.ckpt"

# --------------------------------------------------------------------- #

console = create_console()


def _ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert grayscale inputs to RGB to match ImageNet pretraining."""

    if getattr(image, "mode", "RGB") != "RGB":
        return image.convert("RGB")  # type: ignore[no-any-return]
    return image


@dataclass(frozen=True)
class EvalResult:
    """Simple container for evaluation metrics."""

    acc: float
    total: int
    correct: int


def get_loaders(
    data_root: Path,
    train_split: str,
    val_split: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    *,
    expected_classes: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train/validation loaders. FasterViT expects ImageNet norm."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    small_images = img_size <= 64

    if small_images:
        train_t = transforms.Compose(
            [
                transforms.Lambda(_ensure_rgb),
                transforms.Resize(img_size + 4, interpolation=InterpolationMode.BILINEAR),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ],
        )
        val_t = transforms.Compose(
            [
                transforms.Lambda(_ensure_rgb),
                transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
            ],
        )
    else:
        resize_shorter = max(img_size + 32, int(img_size * 1.15))
        train_t = transforms.Compose(
            [
                transforms.Lambda(_ensure_rgb),
                transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.ToTensor(),
                normalize,
            ],
        )
        val_t = transforms.Compose(
            [
                transforms.Lambda(_ensure_rgb),
                transforms.Resize(resize_shorter, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
            ],
        )

    train_ds = datasets.ImageFolder(data_root / train_split, transform=train_t)
    if expected_classes is not None:
        require_num_classes(train_ds, expected_classes, split=train_split)
    val_ds = datasets.ImageFolder(data_root / val_split, transform=val_t)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        **({"prefetch_factor": 2} if num_workers > 0 else {}),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        **({"prefetch_factor": 2} if num_workers > 0 else {}),
    )
    return train_dl, val_dl


def evaluate(model: nn.Module, dl: DataLoader, device: str) -> EvalResult:
    """Compute top-1 accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch_x, batch_y in dl:
            inputs = batch_x.to(device, non_blocking=True).to(
                memory_format=torch.channels_last,
            )
            targets = batch_y.to(device, non_blocking=True)
            logits = model(inputs)
            pred = logits.argmax(1)
            correct += (pred == targets).sum().item()
            total += targets.numel()
    acc = correct / max(1, total)
    return EvalResult(acc=acc, total=total, correct=correct)


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
) -> None:
    """Single-epoch training loop with AMP and live throughput reporting."""
    model.train()
    start = perf_counter()
    opt.zero_grad(set_to_none=True)

    # Track whether we have pending grads at the end (for non-divisible steps)
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

        # Progress bar: show loss (unscaled) and images/sec
        elapsed = perf_counter() - start
        seen = min(i * dl.batch_size, len(dl.dataset))
        ips = seen / max(1e-6, elapsed)
        shown_loss = float(loss.item() * (max(1, accum_steps)))
        progress.update(
            task,
            advance=1,
            description=f"train | loss={shown_loss:.4f} | {ips:.0f} img/s",
        )

    # Flush any leftover grads if the last micro-batch didn't hit a step
    if pending_steps > 0:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)


def main() -> None:  # noqa: PLR0915
    """Entrypoint: data, model, warmup, fine-tune, early stop, save best."""
    env = prepare_training_environment(
        weights_name=BEST_WEIGHTS_NAME,
        best_checkpoint_name=BEST_CKPT_NAME,
        latest_checkpoint_name=LATEST_CKPT_NAME,
    )
    apply_seed(env.seed)

    data_root = env_path("DD_DATA_ROOT", DATA_ROOT)
    train_split = env_str("DD_TRAIN_SPLIT", "Train")
    val_split = env_str("DD_VAL_SPLIT", "Validation")
    batch_size = env_int("DD_BATCH_SIZE", BATCH_SIZE)
    epochs = env_int("DD_EPOCHS", EPOCHS)
    img_size = env_int("DD_IMG_SIZE", IMG_SIZE)
    num_workers = env_int("DD_NUM_WORKERS", NUM_WORKERS)
    num_classes = env_int("DD_NUM_CLASSES", 2)

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    if env.device_override:
        requested = env.device_override
        if requested.startswith("cuda") and not torch.cuda.is_available():
            console.print(
                "[bold yellow]⚠️  Requested CUDA device not available[/]; falling back to CPU",
            )
            device = "cpu"
            use_cuda = False
        else:
            device = requested
            use_cuda = requested.startswith("cuda")
    torch.backends.cudnn.benchmark = use_cuda and env.seed is None

    if not (data_root / train_split).exists() or not (data_root / val_split).exists():
        console.print(f"[bold red]Dataset not found under[/] {data_root}")
        console.print(
            f"Expected: {data_root}/{train_split}/<class> and {data_root}/{val_split}/<class>",
        )
        raise SystemExit(1)

    try:
        train_dl, val_dl = get_loaders(
            data_root,
            train_split,
            val_split,
            img_size,
            batch_size,
            num_workers,
            expected_classes=num_classes,
        )
    except ValueError as exc:
        console.print(
            "[bold red]Class configuration mismatch[/]",
            f"→ {exc}",
        )
        console.print(
            "Update `data.num_classes` in your YAML to match the dataset. "
            "For MNIST, set it to 10.",
        )
        raise SystemExit(1) from exc
    console.print(
        f"[bold]Data[/]: train={len(train_dl.dataset)} | val={len(val_dl.dataset)} | "
        f"bs={batch_size} | steps/epoch={len(train_dl)}",
    )

    model = create_model(MODEL_NAME, pretrained=True)
    in_features = model.head.in_features  # type: ignore[attr-defined]
    model.head = nn.Linear(in_features, num_classes)  # type: ignore[attr-defined]
    model.to(memory_format=torch.channels_last)
    model = model.to(device)

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

    best_val_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    warmup_done = env.resume_checkpoint is not None

    with progress:
        if not warmup_done:
            for param in model.parameters():
                param.requires_grad = False
            for name, param in model.named_parameters():
                if "head" in name:
                    param.requires_grad = True

            head_params = [p for p in model.parameters() if p.requires_grad]
            warm_opt = optim.AdamW(head_params, lr=HEAD_LR, weight_decay=HEAD_WD)

            warm_task = progress.add_task(
                "warmup (head only)",
                total=len(train_dl),
                extra="",
            )
            console.print("[bold]Warmup (head only)[/]")
            train_one_epoch(
                model=model,
                dl=train_dl,
                opt=warm_opt,
                scaler=scaler,
                criterion=criterion,
                device=device,
                use_cuda_amp=use_cuda,
                progress=progress,
                task=warm_task,
                accum_steps=1,
            )

            res = evaluate(model, val_dl, device)
            console.print(
                f"[bold cyan]warmup[/] | val_acc={res.acc:.4f} ({res.correct}/{res.total})",
            )
            best_val_acc = res.acc
            best_epoch = 0
            warmup_done = True

        for param in model.parameters():
            param.requires_grad = True

        ft_batch_size = 32
        effective_batch = 128
        accum_steps_ft = max(1, effective_batch // ft_batch_size)
        console.print(
            f"[bold]Fine-tune[/]: bs={ft_batch_size}, accum_steps={accum_steps_ft} "
            f"(effective ≈ {ft_batch_size * accum_steps_ft})",
        )

        train_dl_ft = DataLoader(
            train_dl.dataset,
            batch_size=ft_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2,
        )

        opt = optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=FT_LR,
            weight_decay=FT_WD,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - 1))

        start_epoch = 0
        resume_state = maybe_load_checkpoint(
            env,
            model=model,
            optimizer=opt,
            scheduler=scheduler,
        )
        if resume_state is not None:
            start_epoch = int(resume_state.get("epoch", 0))
            best_val_acc = float(resume_state.get("best_val_acc", best_val_acc))
            best_epoch = int(resume_state.get("best_epoch", best_epoch))
            warmup_done = bool(resume_state.get("warmup_done", warmup_done))
            epochs_no_improve = max(0, start_epoch - best_epoch)
            console.print(
                f"[bold green]Resumed[/] from epoch {start_epoch} using {env.resume_checkpoint}",
            )

        for epoch in range(start_epoch + 1, epochs + 1):
            task = progress.add_task(f"epoch {epoch}", total=len(train_dl_ft), extra="")
            train_one_epoch(
                model=model,
                dl=train_dl_ft,
                opt=opt,
                scaler=scaler,
                criterion=criterion,
                device=device,
                use_cuda_amp=use_cuda,
                progress=progress,
                task=task,
                accum_steps=accum_steps_ft,
            )
            scheduler.step()

            res = evaluate(model, val_dl, device)
            console.print(
                f"[bold cyan]epoch {epoch}[/] | val_acc={res.acc:.4f} "
                f"({res.correct}/{res.total}) | lr={scheduler.get_last_lr()[0]:.2e}",
            )

            improved = res.acc > best_val_acc + 1e-4
            if improved:
                best_val_acc = res.acc
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            state = save_latest_checkpoint(
                env,
                model=model,
                optimizer=opt,
                scheduler=scheduler,
                epoch=epoch,
                best_val_acc=best_val_acc,
                best_epoch=best_epoch,
                extra={"warmup_done": warmup_done},
            )

            if improved:
                save_best_checkpoint(env, state)
                console.print(
                    f"[bold green]↑ new best[/] val_acc={best_val_acc:.4f} "
                    f"(epoch {best_epoch}) → saved {env.best_weights_path.name}",
                )
            elif epochs_no_improve >= PATIENCE:
                console.print(
                    f"[bold yellow]Early stopping[/]: no improvement for {PATIENCE} epoch(s). "
                    f"Best at epoch {best_epoch} with val_acc={best_val_acc:.4f}.",
                )
                break

    console.print(f"[bold green]Best weights saved →[/] {env.best_weights_path.resolve()}")
    console.print(
        f"[bold green]Best checkpoint saved →[/] {env.best_checkpoint_path.resolve()}",
    )


if __name__ == "__main__":
    main()
