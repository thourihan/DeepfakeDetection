# ruff: noqa: INP001
"""Supervised training script for RepViT (via timm) on an ImageFolder dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import timm
import torch
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

from orchestration.train_env import (
    apply_seed,
    create_console,
    env_float,
    env_int,
    env_path,
    env_str,
    load_transform_toggles,
    maybe_load_checkpoint,
    prepare_training_environment,
    require_num_classes,
    save_best_checkpoint,
    save_latest_checkpoint,
)

console = create_console()

DATA_ROOT: Path = Path.cwd() / "data"
DEFAULT_MODEL_ID: str = "repvit_m1.dist_in1k"

DEFAULT_EPOCHS: int = 10
DEFAULT_BATCH_SIZE: int = 128
DEFAULT_IMG_SIZE: int = 224
DEFAULT_NUM_WORKERS: int = 8

DEFAULT_LR: float = 2e-4
DEFAULT_WD: float = 5e-2
DEFAULT_PATIENCE: int = 5
DEFAULT_ACCUM_STEPS: int = 1

BEST_WEIGHTS_NAME: str = "RepViTModel.pth"
BEST_CKPT_NAME: str = "best.ckpt"
LATEST_CKPT_NAME: str = "latest.ckpt"


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if getattr(image, "mode", "RGB") != "RGB":
        return image.convert("RGB")  # type: ignore[no-any-return]
    return image


@dataclass(frozen=True)
class EvalResult:
    acc: float
    loss: float
    total: int
    correct: int


def build_loaders(
    *,
    data_root: Path,
    train_split: str,
    val_split: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    expected_classes: int,
) -> tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    defaults = {
        "ensure_rgb": True,
        "train_resize": False,
        "train_random_crop": False,
        "train_center_crop": False,
        "train_random_resized_crop": True,
        "train_random_horizontal_flip": True,
        "train_random_rotation": False,
        "train_color_jitter": True,
        "train_random_erasing": False,
        "train_to_tensor": True,
        "train_normalize": True,
        "val_resize": True,
        "val_center_crop": True,
        "val_to_tensor": True,
        "val_normalize": True,
    }
    toggles = load_transform_toggles(
        defaults,
        required=("train_to_tensor", "train_normalize", "val_to_tensor", "val_normalize"),
    )

    train_ops: list[object] = []
    if toggles.get("ensure_rgb", True):
        train_ops.append(transforms.Lambda(_ensure_rgb))
    if toggles.get("train_random_resized_crop", True):
        train_ops.append(transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)))
    else:
        train_ops.append(
            transforms.Resize(int(img_size * 1.15), interpolation=InterpolationMode.BILINEAR),
        )
        train_ops.append(transforms.CenterCrop(img_size))
    if toggles.get("train_random_horizontal_flip", True):
        train_ops.append(transforms.RandomHorizontalFlip())
    if toggles.get("train_random_rotation", False):
        train_ops.append(transforms.RandomRotation(10))
    if toggles.get("train_color_jitter", False):
        train_ops.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.05))
    if toggles.get("train_to_tensor", True):
        train_ops.append(transforms.ToTensor())
    if toggles.get("train_normalize", True):
        train_ops.append(normalize)
    if toggles.get("train_random_erasing", False):
        train_ops.append(
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value=0,
            ),
        )

    val_ops: list[object] = []
    if toggles.get("ensure_rgb", True):
        val_ops.append(transforms.Lambda(_ensure_rgb))
    if toggles.get("val_resize", True):
        val_ops.append(
            transforms.Resize(int(img_size * 1.15), interpolation=InterpolationMode.BILINEAR),
        )
    if toggles.get("val_center_crop", True):
        val_ops.append(transforms.CenterCrop(img_size))
    if toggles.get("val_to_tensor", True):
        val_ops.append(transforms.ToTensor())
    if toggles.get("val_normalize", True):
        val_ops.append(normalize)

    train_ds = datasets.ImageFolder(data_root / train_split, transform=transforms.Compose(train_ops))
    require_num_classes(train_ds, expected_classes, split=train_split, dataset_root=data_root)
    val_ds = datasets.ImageFolder(data_root / val_split, transform=transforms.Compose(val_ops))
    require_num_classes(val_ds, expected_classes, split=val_split, dataset_root=data_root)

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


def evaluate(model: nn.Module, dl: DataLoader, device: str, criterion: nn.Module) -> EvalResult:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for batch_x, batch_y in dl:
            inputs = batch_x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            targets = batch_y.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            pred = logits.argmax(1)
            correct += (pred == targets).sum().item()
            total += targets.numel()
            loss_sum += float(loss.item()) * targets.size(0)
    acc = correct / max(1, total)
    return EvalResult(acc=acc, loss=loss_sum / max(1, total), total=total, correct=correct)


def train_one_epoch(  # noqa: PLR0913
    *,
    model: nn.Module,
    dl: DataLoader,
    opt: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: str,
    use_cuda_amp: bool,
    progress: Progress,
    task: TaskID,
    accum_steps: int,
) -> float:
    model.train()
    start = perf_counter()
    opt.zero_grad(set_to_none=True)

    loss_sum = 0.0
    seen_total = 0
    pending = 0
    amp_device = "cuda" if use_cuda_amp else "cpu"

    for i, (batch_x, batch_y) in enumerate(dl, 1):
        inputs = batch_x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = batch_y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=amp_device, enabled=use_cuda_amp):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if accum_steps > 1:
                loss = loss / accum_steps

        scaler.scale(loss).backward()
        pending += 1

        if pending == accum_steps:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            pending = 0

        bsz = targets.size(0)
        seen_total += bsz
        loss_sum += float(loss.item()) * bsz * max(1, accum_steps)

        elapsed = perf_counter() - start
        seen = min(i * dl.batch_size, len(dl.dataset))
        ips = seen / max(1e-6, elapsed)
        shown_loss = float(loss.item() * max(1, accum_steps))
        progress.update(task, advance=1, description=f"train | loss={shown_loss:.4f} | {ips:.0f} img/s")

    if pending > 0:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    return loss_sum / max(1, seen_total)


def main() -> None:  # noqa: PLR0915
    env = prepare_training_environment(
        weights_name=BEST_WEIGHTS_NAME,
        best_checkpoint_name=BEST_CKPT_NAME,
        latest_checkpoint_name=LATEST_CKPT_NAME,
    )
    apply_seed(env.seed)

    data_root = env_path("DATA_ROOT", DATA_ROOT)
    train_split = env_str("TRAIN_SPLIT", "train")
    val_split = env_str("VAL_SPLIT", "val")
    batch_size = env_int("BATCH_SIZE", DEFAULT_BATCH_SIZE)
    epochs = env_int("EPOCHS", DEFAULT_EPOCHS)
    img_size = env_int("IMG_SIZE", DEFAULT_IMG_SIZE)
    num_workers = env_int("NUM_WORKERS", DEFAULT_NUM_WORKERS)
    num_classes = env_int("NUM_CLASSES", 2)

    lr = env_float("LR", DEFAULT_LR)
    wd = env_float("WEIGHT_DECAY", DEFAULT_WD)
    accum_steps = env_int("ACCUM_STEPS", DEFAULT_ACCUM_STEPS)
    early_stop_patience = env_int("EARLY_STOP_PATIENCE", DEFAULT_PATIENCE)

    model_id = env_str("MODEL_NAME", DEFAULT_MODEL_ID)

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    if env.device_override:
        requested = env.device_override
        if requested.startswith("cuda") and not torch.cuda.is_available():
            console.print("[bold yellow]⚠️  CUDA requested but unavailable[/]: using CPU")
            device = "cpu"
            use_cuda = False
        else:
            device = requested
            use_cuda = requested.startswith("cuda")
    torch.backends.cudnn.benchmark = use_cuda and env.seed is None

    if not (data_root / train_split).exists() or not (data_root / val_split).exists():
        console.print(f"[bold red]Dataset not found under[/] {data_root}")
        raise SystemExit(1)

    train_dl, val_dl = build_loaders(
        data_root=data_root,
        train_split=train_split,
        val_split=val_split,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        expected_classes=num_classes,
    )

    console.print(
        f"[bold]Model[/]: {model_id} | classes={num_classes} | img={img_size} | bs={batch_size}"
    )
    model = timm.create_model(model_id, pretrained=True, num_classes=num_classes)
    model.to(memory_format=torch.channels_last)
    model = model.to(device)

    criterion: nn.Module = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=use_cuda)

    opt = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    best_val_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    resume_state = maybe_load_checkpoint(env, model=model, optimizer=opt, scheduler=scheduler)
    start_epoch = 0
    if resume_state is not None:
        start_epoch = int(resume_state.get("epoch", 0))
        best_val_acc = float(resume_state.get("best_val_acc", best_val_acc))
        best_epoch = int(resume_state.get("best_epoch", best_epoch))
        epochs_no_improve = max(0, start_epoch - best_epoch)
        console.print(f"[bold green]Resumed[/] from epoch {start_epoch} ({env.resume_checkpoint})")

    with progress:
        for epoch in range(start_epoch + 1, epochs + 1):
            task = progress.add_task(f"epoch {epoch}", total=len(train_dl))
            train_loss = train_one_epoch(
                model=model,
                dl=train_dl,
                opt=opt,
                scaler=scaler,
                criterion=criterion,
                device=device,
                use_cuda_amp=use_cuda,
                progress=progress,
                task=task,
                accum_steps=max(1, accum_steps),
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
                extra={"model_id": model_id},
            )

            if improved:
                save_best_checkpoint(env, state)
                console.print(
                    f"[bold green]new best[/] val_acc={best_val_acc:.4f} "
                    f"(epoch {best_epoch}) → saved {env.best_weights_path.name}"
                )
            elif epochs_no_improve >= early_stop_patience:
                console.print(
                    f"[bold yellow]Early stopping[/]: best at epoch {best_epoch} "
                    f"with val_acc={best_val_acc:.4f}."
                )
                break

    console.print(f"[bold green]Best weights saved →[/] {env.best_weights_path.resolve()}")
    console.print(f"[bold green]Best checkpoint saved →[/] {env.best_checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
