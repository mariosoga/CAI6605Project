import os
import random
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torchvision.transforms import v2

from core.PlantModel import MultiTaskEffNet


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Optional: make hash-based ops deterministic too
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def seed_worker(worker_id: int):
    """
    Use with DataLoader(..., worker_init_fn=seed_worker, generator=...)
    to keep augmentations/dataloader consistent across runs.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_deterministic(enable: bool = True):
    """
    Extra guard for deterministic behavior on newer PyTorch.
    """
    try:
        torch.use_deterministic_algorithms(enable)
    except Exception:
        pass


# -----------------------------
# Device scripts
# -----------------------------
def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Running on GPU: {name}")
        return torch.device("cuda")
    # Apple Silicon
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Running on Apple MPS")
        return torch.device("mps")
    print("Running on CPU")
    return torch.device("cpu")


# -----------------------------
# Filesystem
# -----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_floats(csv: str) -> List[float]:
    return [float(x.strip()) for x in csv.split(",") if x.strip()]


# ----------------------------
# Transforms
# ----------------------------
def build_transforms(img_size: int = 224, mode: str = 'train', mean: List[float] = (0.485, 0.456, 0.406),
                     std: List[float] = (0.229, 0.224, 0.225)):
    if mode == 'train':
        return v2.Compose([
            v2.Resize((img_size, img_size), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(10),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])
    elif mode == 'val':
        return v2.Compose([
            v2.Resize((img_size, img_size), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])
    else:
        raise ValueError(f"Unknown mode {mode}")


def load_model(ckpt_path: Path, device: torch.device) -> Tuple[
    nn.Module, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    chkpt_args = ckpt.get('args', {})
    model_name = chkpt_args.get('model', 'efficientnet_b0')

    class_names = None

    species_list = ckpt.get('species_list')
    disease_list = ckpt.get('disease_list')
    if species_list is None or disease_list is None:
        raise RuntimeError("Checkpoint missing 'species list' or 'disease list for multitask model'")

    model = MultiTaskEffNet(model_name, num_s=len(species_list), num_d=len(disease_list))

    model.load_state_dict(ckpt['model_state'], strict=True)
    model.to(device).eval()

    meta: Dict[str, Any] = {"class_names": class_names,
                            "species_list": species_list,
                            "disease_list": disease_list,
                            "args": chkpt_args}
    return model, meta


# -----------------------------
# Visualization
# -----------------------------
def plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        out_path: Path,
        normalize: Optional[str] = None,  # 'true' | 'pred' | 'all' | None
        cmap: str = "Blues",
        show_values: bool = True,
):
    """
    Save a (optionally normalized) confusion matrix image.
    - normalize=None: raw counts
    - normalize='true': row-normalized (per true class)
    - normalize='pred': column-normalized (per predicted class)
    - normalize='all' : matrix normalized by total samples
    """
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_display = cm.astype(np.float64)

    if normalize is not None:
        norm = normalize.lower()
        if norm == "true":  # rows sum to 1
            row_sums = cm_display.sum(axis=1, keepdims=True).clip(min=1e-12)
            cm_display = cm_display / row_sums
        elif norm == "pred":  # cols sum to 1
            col_sums = cm_display.sum(axis=0, keepdims=True).clip(min=1e-12)
            cm_display = cm_display / col_sums
        elif norm == "all":
            total = cm_display.sum().clip(min=1e-12)
            cm_display = cm_display / total
        else:
            # unrecognized -> fallback to counts
            pass

    # Figure size scales with number of classes (cap to avoid huge canvases)
    n = len(class_names)
    side = min(0.45 * n + 4.0, 20.0)
    fig, ax = plt.subplots(figsize=(side, side))

    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix" + (f" (norm={normalize})" if normalize else ""), pad=12)
    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted", labelpad=8)
    ax.set_ylabel("True", labelpad=8)

    # Gridlines for readability
    ax.set_xlim(-0.5, n - 1 + 0.5)
    ax.set_ylim(n - 1 + 0.5, -0.5)
    ax.set_aspect("auto")

    # Annotate cells
    if show_values:
        # choose format: percentages if normalized, else counts
        use_pct = normalize is not None
        fmt = ".1%" if use_pct else "d"
        # for percentages, the data is [0,1]; for counts, it's ints
        for i in range(n):
            for j in range(n):
                val = cm_display[i, j]
                text = f"{val:{fmt}}" if use_pct else f"{int(val)}"
                ax.text(
                    j, i, text,
                    ha="center", va="center",
                    color="white" if im.norm(val) > 0.5 else "black",
                    fontsize=8
                )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
