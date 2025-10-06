# grad_cam_hard_examples.py
# Lightweight, blog-style Grad-CAM scripts for EfficientNet-B0 (timm or torchvision)
# and your MultiTaskEffNet wrapper. No OpenCV required.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn

from core.utils import build_transforms, ensure_dir


def denorm_to_rgb(x: torch.Tensor, mean: List[float], std: List[float]) -> np.ndarray:
    mean_t = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    img = (x * std_t + mean_t).clamp(0, 1)[0].permute(1, 2, 0).detach().cpu().numpy()
    return img


def jet_colormap(gray: np.ndarray) -> np.ndarray:
    H, W = gray.shape
    c = np.zeros((H, W, 3), dtype=np.float32)
    c[..., 0] = np.clip(1.5 - np.abs(4 * gray - 3), 0, 1)  # R
    c[..., 1] = np.clip(1.5 - np.abs(4 * gray - 2), 0, 1)  # G
    c[..., 2] = np.clip(1.5 - np.abs(4 * gray - 1), 0, 1)  # B
    return c


def overlay_heatmap(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = jet_colormap(cam)  # H,W,3
    out = (1 - alpha) * rgb + alpha * heat
    return np.clip(out, 0, 1)


# ----------------------------
# Model adaptation
# ----------------------------
class CamSingleHead(nn.Module):
    """
    Wrap a multi-head model to expose a single classification output for CAM.
    Assumes wrapped model has attributes: .backbone, .disease_head, .health_head
    """

    def __init__(self, mt_model: nn.Module, task: str = 'disease'):
        super().__init__()
        assert hasattr(mt_model, "backbone"), "Expected a multi-task model with .backbone"
        self.backbone = mt_model.backbone  # keep reference so we can select conv layers
        self.task = task
        self.mt = mt_model

    def forward(self, x):
        feats = self.backbone(x)
        if self.task == 'health':
            return self.mt.health_head(feats)
        elif self.task == 'disease':
            return self.mt.disease_head(feats)
        elif self.task == 'species':
            return self.mt.species_head(feats)
        else:
            raise NotImplementedError


def resolve_target_layer(model: nn.Module, target_layer_path: Optional[str]) -> nn.Module:
    """
    If target_layer_path is provided, follow the dotted path from the (wrapped) model.
    Otherwise, choose a sensible default for EfficientNet-B0:
      - timm:       *.conv_head
      - torchvision *.features[-1]
    For MultiTaskEffNet, we look inside .backbone.
    """

    def get_by_path(m: nn.Module, dotted: str) -> nn.Module:
        cur = m
        for tok in dotted.split("."):
            cur = cur[int(tok)] if tok.isdigit() else getattr(cur, tok)
        return cur

    if target_layer_path:
        return get_by_path(model, target_layer_path)

    # try timm conv_head first
    if hasattr(model, "conv_head"):
        return getattr(model, "conv_head")

    # if this is a wrapper or a torchvision model, features is a Sequential
    parent = model
    if hasattr(model, "backbone"):  # wrapper or mt model passed directly
        parent = model.backbone

    # For timm EfficientNet
    if hasattr(parent, "conv_head"):
        return getattr(parent, "conv_head")
    if hasattr(parent, "features") and isinstance(parent.features, nn.Sequential):
        return parent.features[-1]

    raise ValueError("Could not auto-resolve a target layer; pass --grad_cam_layer explicitly.")


@torch.no_grad()
def top2_from_logits(logits: torch.Tensor) -> Tuple[int, int]:
    p = torch.softmax(logits, dim=1)[0]
    k = 2 if p.numel() >= 2 else 1
    idx = torch.topk(p, k=k).indices.tolist()
    if len(idx) == 1:
        idx.append(idx[0])
    return idx[0], idx[1]


# ----------------------------
# Main entry
# ----------------------------
def run_gradcam_for_hard_examples(
        model: nn.Module,
        records: List[Dict[str, Any]],
        class_names: List[str],
        out_dir: Path,
        k: int = 50,
        img_size: int = 224,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        cam_alpha: float = 0.45,
        target_layer_path: Optional[str] = None,
        multitask: bool = False,
        task: str = 'disease',  # if multitask=True
):
    """
    Generates Grad-CAM overlays ONLY for hard examples:
      - Overconfident Wrong: CAM for predicted and true classes
      - Uncertain Correct:   CAM for true(=pred) and runner-up
    Saves into:
      out_dir/hard_examples/overconfident_wrong_cam/*.png
      out_dir/hard_examples/uncertain_correct_cam/*.png
    """
    device = next(model.parameters()).device
    model.eval()

    # build single-head view if needed
    if multitask:
        cam_model = CamSingleHead(model, task=task).to(device)
    else:
        cam_model = model  # single-head classifier

    # resolve layer on the cam_model view
    target_layer = resolve_target_layer(cam_model, target_layer_path)

    cam = GradCAM(model=cam_model, target_layers=[target_layer])

    tfm = build_transforms(img_size, mode="val", mean=mean, std=std)

    ow = [r for r in records if r["pred_idx"] != r["true_idx"]]
    ow.sort(key=lambda r: r["pred_conf"], reverse=True)

    uc = [r for r in records if r["pred_idx"] == r["true_idx"]]
    uc.sort(key=lambda r: r["true_conf"])  # lowest confidence first

    ow_dir = out_dir / "hard_examples" / "overconfident_wrong_cam"
    uc_dir = out_dir / "hard_examples" / "uncertain_correct_cam"
    ensure_dir(ow_dir)
    ensure_dir(uc_dir)

    def load_x(p: Path) -> torch.Tensor:
        pil = Image.open(p).convert("RGB")
        return tfm(pil).unsqueeze(0).to(device)

    def save_overlay(x: torch.Tensor, gray: np.ndarray, out_path: Path):
        rgb = denorm_to_rgb(x, mean, std)
        vis = overlay_heatmap(rgb, gray, alpha=cam_alpha)
        Image.fromarray((vis * 255).astype(np.uint8)).save(out_path)

    # Overconfident Wrong: predicted & true
    for r in ow[:k]:
        p = Path(r["path"])
        x = load_x(p)
        base = f"true={class_names[r['true_idx']]}__pred={class_names[r['pred_idx']]}__p={r['pred_conf']:.3f}"

        targets_pred = [ClassifierOutputTarget(int(r["pred_idx"]))]
        targets_true = [ClassifierOutputTarget(int(r["true_idx"]))]

        gray_pred = cam(input_tensor=x, targets=targets_pred)[0]
        gray_true = cam(input_tensor=x, targets=targets_true)[0]

        save_overlay(x, gray_pred, ow_dir / f"{base}__CAM_pred.png")
        save_overlay(x, gray_true, ow_dir / f"{base}__CAM_true.png")

    # Uncertain Correct: true (=pred) & runner-up
    for r in uc[:k]:
        p = Path(r["path"])
        x = load_x(p)
        with torch.no_grad():
            logits = cam_model(x)
        pred_idx, runnerup = top2_from_logits(logits)

        base = f"true={class_names[r['true_idx']]}__pred={class_names[r['pred_idx']]}__p={r['true_conf']:.3f}"

        targets_true = [ClassifierOutputTarget(int(r["true_idx"]))]
        targets_alt = [ClassifierOutputTarget(int(runnerup))]

        gray_true = cam(input_tensor=x, targets=targets_true)[0]
        gray_alt = cam(input_tensor=x, targets=targets_alt)[0]

        save_overlay(x, gray_true, uc_dir / f"{base}__CAM_true(pred).png")
        save_overlay(x, gray_alt, uc_dir / f"{base}__CAM_runnerup.png")

    print(f"[Grad-CAM] saved:\n  {ow_dir}\n  {uc_dir}")
