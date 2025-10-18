from __future__ import annotations
"""
Minimal evaluation + Grad-CAM visualization for EfficientFormerV2 on a
Real/Fake test set. Expects a folder layout compatible with
torchvision.datasets.ImageFolder:

    DATA_ROOT/
        Real/
            *.jpg|*.png|...
        Fake/
            *.jpg|*.png|...

Notes:
- Transforms intentionally omit normalization to mirror training/eval.
- Grad-CAM uses the model's predicted class by default; override with
  `class_idx` if you need a specific target.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


DATA_ROOT: Path = Path.home() / "code" / "DeepfakeDetection" / "data" / "Dataset" / "Test"

# timm model identifier and local weights to load.
MODEL_NAME: str = "efficientformerv2_s1"
WEIGHTS_PATH: Path = (
    Path.home()
    / "code"
    / "DeepfakeDetection"
    / "models"
    / "efficientformer_v2"
    / "EfficientFormerV2_S1.pth"
)

# TODO: Move constants below to a config file.
# Evaluation and CAM parameters.
BATCH_SIZE: int = 128
IMG_SIZE: int = 224
NUM_WORKERS: int = 8
N_CAM_SAMPLES: int = 5
OUTPUT_DIR: Path = Path.home() / "code" / "DeepfakeDetection" / "outputs" / "cam_samples"

console = Console()


@dataclass(frozen=True)
class EvalResult:
    """Simple container for evaluation metrics."""
    accuracy: float
    total: int
    correct: int


def get_test_loader(root: Path, img_size: int) -> DataLoader:
    """Create a DataLoader over `root` using ImageFolder.

    Transforms:
        - Resize -> CenterCrop -> ToTensor (no normalization).
          Keep this consistent with training/evaluation.
    """
    if not root.exists():
        console.print(f"[bold red]Test folder not found:[/] {root}")
        console.print("Expected structure: Dataset/Test/{Real,Fake}")
        raise SystemExit(1)

    t = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),  # consistent with training/eval (no normalization)
        ],
    )
    ds = datasets.ImageFolder(root, transform=t)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    return dl


def build_model(model_name: str, num_classes: int, device: str) -> nn.Module:
    """Instantiate timm model, load weights, switch to eval+channels_last."""
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    sd = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.to(memory_format=torch.channels_last)
    model = model.to(device)
    model.eval()
    return model


def evaluate_with_progress(model: nn.Module, dl: DataLoader, device: str) -> EvalResult:
    """Run a single pass over the test set with a progress bar."""
    correct, total = 0, 0
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
    with progress, torch.inference_mode():
        task = progress.add_task("eval (test set)", total=len(dl), extra="")
        for i, (x, y) in enumerate(dl, 1):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            acc_so_far = correct / max(1, total)
            progress.update(task, advance=1, extra=f"acc={acc_so_far:.4f}")
    return EvalResult(accuracy=correct / max(1, total), total=total, correct=correct)


def find_last_conv_layer(module: nn.Module) -> nn.Module:
    """Return the last nn.Conv2d encountered in the module.

    Some modern backbones use few explicit convs; adjust if needed.
    """
    last_conv: nn.Module | None = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM target.")
    return last_conv


def sample_test_images(root: Path, n: int) -> list[Path]:
    """Uniformly sample up to `n` image files from subtree rooted at `root`."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_files = [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    if not all_files:
        raise SystemExit(f"No images found under {root}")
    return random.sample(all_files, k=min(n, len(all_files)))


def load_image_for_cam(path: Path, img_size: int) -> Tuple[torch.Tensor, np.ndarray]:
    """Load an image for inference and CAM overlay.

    Returns:
        x: (1, 3, H, W) tensor after Resize+CenterCrop+ToTensor.
        rgb: HWC float32 in [0,1] for overlay composition.

    Keep the spatial transforms synchronized with the model input to avoid
    CAM misalignment.
    """
    img = Image.open(path).convert("RGB")
    t = transforms.Compose(
        [transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor()]
    )
    x = t(img).unsqueeze(0)  # (1, 3, H, W)
    # The overlay path uses a direct PIL resize to IMG_SIZE; this matches the
    # spatial size of `x` after the Compose above.
    rgb = np.array(img.resize((img_size, img_size))).astype(np.float32) / 255.0  # HWC in [0,1]
    return x, rgb


def make_cam_overlays(
    model: nn.Module,
    device: str,
    class_idx: int | None,
    target_layer: nn.Module,
    paths: Iterable[Path],
    out_dir: Path,
) -> None:
    """Generate side-by-side original/CAM overlays for the given image paths.

    - `class_idx=None` targets the model's argmax per image.
    - Uses a context manager to guarantee hook cleanup.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    # Ensure hooks are released promptly (avoids __del__ warnings).
    with GradCAM(model=model, target_layers=[target_layer]) as cam, progress:
        task = progress.add_task("grad-cam", total=len(list(paths)))
        for i, path in enumerate(paths, 1):
            x, rgb = load_image_for_cam(path, IMG_SIZE)
            x = x.to(device, non_blocking=True)

            with torch.inference_mode():
                logits = model(x)
                pred = int(logits.argmax(1).item())

            targets = [ClassifierOutputTarget(pred if class_idx is None else class_idx)]
            grayscale_cam = cam(input_tensor=x, targets=targets)[0]  # (H, W) float32 in [0,1]
            overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)  # uint8 RGB

            orig = (rgb * 255).astype(np.uint8)
            side_by_side = np.concatenate([orig, overlay], axis=1)
            out_path = out_dir / f"cam_{i:02d}_{path.parent.name}_{path.stem}.png"
            Image.fromarray(side_by_side).save(out_path)

            progress.update(task, advance=1)


def main() -> None:
    """Entry point: device setup, evaluation, and CAM generation."""
    # Device
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    if use_cuda:
        console.print("[bold green]✅ CUDA available[/]: using GPU")
        console.print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        console.print("[bold yellow]⚠️  CUDA not available[/]: falling back to CPU")

    # Data & model
    dl = get_test_loader(DATA_ROOT, IMG_SIZE)
    num_classes = len(dl.dataset.classes)
    console.print(f"[bold]Classes[/]: {dl.dataset.classes} (num_classes={num_classes})")

    if not WEIGHTS_PATH.exists():
        console.print(f"[bold red]Weights not found:[/] {WEIGHTS_PATH}")
        raise SystemExit(1)

    model = build_model(MODEL_NAME, num_classes, device)

    # Evaluate with a progress bar
    res = evaluate_with_progress(model, dl, device)
    console.print(f"[bold cyan]TEST[/] | acc={res.accuracy:.4f} ({res.correct}/{res.total})")

    # Grad-CAM on random images from Test
    target_layer = find_last_conv_layer(model)
    samples = sample_test_images(DATA_ROOT, N_CAM_SAMPLES)
    console.print(
        f"[bold]Grad-CAM[/]: generating overlays for {len(samples)} random test images → {OUTPUT_DIR}"
    )
    make_cam_overlays(
        model, device, class_idx=None, target_layer=target_layer, paths=samples, out_dir=OUTPUT_DIR
    )
    console.print("[bold green]Done.[/]")


if __name__ == "__main__":
    main()
