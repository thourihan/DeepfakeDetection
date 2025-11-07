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

# Configuration constants (could be moved to a config file).
DATA_ROOT: Path = Path.home() / "code" / "DeepfakeDetection" / "data" / "Dataset"
MODEL_NAME: str = "efficientformerv2_s1"
EPOCHS: int = 5
BATCH_SIZE: int = 128
IMG_SIZE: int = 224
NUM_WORKERS: int = 8
BEST_WEIGHTS_NAME: str = "EfficientFormerV2_S1.pth"
BEST_CKPT_NAME: str = "best.ckpt"
LATEST_CKPT_NAME: str = "latest.ckpt"

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

console = create_console()


def _ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert grayscale frames to RGB for ImageNet-pretrained backbones."""

    if getattr(image, "mode", "RGB") != "RGB":
        return image.convert("RGB")  # type: ignore[no-any-return]
    return image


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
    """Build train/validation loaders with light augmentations on train."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    small_images = img_size <= 64

    defaults = {
        "ensure_rgb": True,
        "train_resize": True,
        "train_random_crop": small_images,
        "train_center_crop": False,
        "train_random_resized_crop": not small_images,
        "train_random_horizontal_flip": True,
        "train_random_rotation": False,
        "train_color_jitter": not small_images,
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

    if small_images:
        if toggles.get("train_resize", True):
            train_ops.append(
                transforms.Resize(img_size + 4, interpolation=InterpolationMode.BILINEAR),
            )
        if toggles.get("train_random_crop", True):
            train_ops.append(transforms.RandomCrop(img_size))
        elif toggles.get("train_center_crop", False):
            train_ops.append(transforms.CenterCrop(img_size))
    else:
        resize_shorter = max(img_size + 32, int(img_size * 1.15))
        if toggles.get("train_random_resized_crop", True):
            train_ops.append(transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)))
        else:
            if toggles.get("train_resize", True):
                train_ops.append(
                    transforms.Resize(resize_shorter, interpolation=InterpolationMode.BILINEAR),
                )
            if toggles.get("train_center_crop", True):
                train_ops.append(transforms.CenterCrop(img_size))

    if toggles.get("train_random_horizontal_flip", True):
        train_ops.append(transforms.RandomHorizontalFlip())
    if toggles.get("train_random_rotation", False):
        train_ops.append(transforms.RandomRotation(10))
    if toggles.get("train_color_jitter", False):
        train_ops.append(transforms.ColorJitter(0.1, 0.1, 0.1, 0.05))
    if toggles.get("train_to_tensor", True):
        train_ops.append(transforms.ToTensor())
    if toggles.get("train_normalize", True):
        train_ops.append(normalize)
    if toggles.get("train_random_erasing", False):
        train_ops.append(
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
        )

    val_ops: list[object] = []
    if toggles.get("ensure_rgb", True):
        val_ops.append(transforms.Lambda(_ensure_rgb))
    if toggles.get("val_resize", True):
        resize_target = max(img_size + 32, int(img_size * 1.15)) if not small_images else img_size
        val_ops.append(transforms.Resize(resize_target, interpolation=InterpolationMode.BILINEAR))
    if toggles.get("val_center_crop", True):
        val_ops.append(transforms.CenterCrop(img_size))
    if toggles.get("val_to_tensor", True):
        val_ops.append(transforms.ToTensor())
    if toggles.get("val_normalize", True):
        val_ops.append(normalize)

    train_t = transforms.Compose(train_ops)
    val_t = transforms.Compose(val_ops)

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
    env = prepare_training_environment(
        weights_name=BEST_WEIGHTS_NAME,
        best_checkpoint_name=BEST_CKPT_NAME,
        latest_checkpoint_name=LATEST_CKPT_NAME,
    )
    apply_seed(env.seed)

    data_root = env_path("DATA_ROOT", DATA_ROOT)
    train_split = env_str("TRAIN_SPLIT", "Train")
    val_split = env_str("VAL_SPLIT", "Validation")
    batch_size = env_int("BATCH_SIZE", BATCH_SIZE)
    epochs = env_int("EPOCHS", EPOCHS)
    img_size = env_int("IMG_SIZE", IMG_SIZE)
    num_workers = env_int("NUM_WORKERS", NUM_WORKERS)
    num_classes = env_int("NUM_CLASSES", 2)

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

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes, img_size=img_size)
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
    warmup_done = env.resume_checkpoint is not None

    with progress:
        if not warmup_done:
            for name, param in model.named_parameters():
                param.requires_grad = ("classifier" in name) or ("head" in name)

            head_params = [p for p in model.parameters() if p.requires_grad]
            warm_opt = optim.AdamW(head_params, lr=3e-4, weight_decay=5e-2)

            warmup_task = progress.add_task("warmup", total=len(train_dl), extra="")
            console.print("[bold]Warmup (head only)[/]")
            start = perf_counter()

            model.train()
            for i, (batch_x, batch_y) in enumerate(train_dl, 1):
                inputs = batch_x.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last,
                )
                targets = batch_y.to(device, non_blocking=True)
                warm_opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                    loss = criterion(model(inputs), targets)
                scaler.scale(loss).backward()
                scaler.step(warm_opt)
                scaler.update()

                elapsed = perf_counter() - start
                seen = min(i * train_dl.batch_size, len(train_dl.dataset))
                ips = seen / max(1e-6, elapsed)
                progress.update(
                    warmup_task,
                    advance=1,
                    description=f"warmup | loss={loss.item():.4f}",
                    extra=f"{ips:.0f} img/s",
                )

            best_val_acc = evaluate(model, val_dl, device)
            console.print(f"[bold cyan]warmup[/] | val_acc={best_val_acc:.4f}")
            best_epoch = 0
            warmup_done = True

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
            console.print(
                f"[bold green]Resumed[/] from epoch {start_epoch} using {env.resume_checkpoint}",
            )

        for epoch in range(start_epoch + 1, epochs + 1):
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
            val_acc = evaluate(model, val_dl, device)
            console.print(f"[bold cyan]epoch {epoch}[/] | val_acc={val_acc:.4f}")

            improved = val_acc > best_val_acc + 1e-4
            if improved:
                best_val_acc = val_acc
                best_epoch = epoch

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
                    f"[bold green]new best[/] val_acc={best_val_acc:.4f} "
                    f"(epoch {best_epoch}) → saved {env.best_weights_path.name}",
                )

    console.print(f"[bold green]Best weights saved →[/] {env.best_weights_path.resolve()}")
    console.print(
        f"[bold green]Best checkpoint saved →[/] {env.best_checkpoint_path.resolve()}",
    )


if __name__ == "__main__":
    main()
