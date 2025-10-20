import re
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib

try:
    matplotlib.use("TkAgg")  # or 'Qt5Agg' if available
except Exception:
    matplotlib.use("Agg")  # headless fallback

from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from core.utils import set_seed, get_device, ensure_dir, build_transforms, load_model, parse_floats

from core.gradcam_utils import (
    CamSingleHead,
    resolve_target_layer,
    overlay_heatmap,
    denorm_to_rgb,
)


# Filename parsing helpers
def parse_labels_from_stem(stem: str) -> Tuple[Optional[str], Optional[str]]:
    m_true = re.search(r"true=(.*?)(?:__|$)", stem, flags=re.IGNORECASE)
    m_pred = re.search(r"pred=(.*?)(?:__|$)", stem, flags=re.IGNORECASE)
    true_label = m_true.group(1) if m_true else None
    pred_label = m_pred.group(1) if m_pred else None
    return true_label, pred_label


def choose_left_title(stem: str, model_pred: str) -> str:
    true_label, pred_label = parse_labels_from_stem(stem)
    if true_label:
        return true_label
    if pred_label:
        return pred_label
    return model_pred


# Plotting helpers
def _hide_axes(ax):
    ax.set_xticks([]);
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_side_by_side(left_img,
                      right_img,
                      left_title: str,
                      right_title: str,
                      main_title: str | None = None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(left_img);
    ax[0].set_title(left_title, pad=6)
    ax[1].imshow(right_img);
    ax[1].set_title(right_title, pad=6)
    for a in ax: _hide_axes(a)
    if main_title:
        fig.suptitle(main_title, y=0.98)
    fig.tight_layout()
    return fig


def plot_triptych(left_img, mid_img, probs: np.ndarray, class_names: List[str], pred_idx: int,
                  left_title: str, mid_title: str = "Grad-CAM Overlay", topk: int = 5, main_title: str | None = None):
    """
    Create a 1x3 figure:
      [ ClassName (Original) | Grad-CAM Overlay | Prediction Panel (Top-k bar chart) ]
    """
    # Top-k selection
    topk = int(min(topk, probs.size))
    topk_idx = np.argsort(probs)[-topk:][::-1]
    topk_labels = [class_names[i] for i in topk_idx]
    topk_vals = probs[topk_idx] * 100.0

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1, 1.2]})

    # Left: Original with class title
    ax[0].imshow(left_img)
    ax[0].set_title(left_title, pad=6)
    _hide_axes(ax[0])

    # Middle: Grad-CAM Overlay
    ax[1].imshow(mid_img)
    ax[1].set_title(mid_title, pad=6)
    _hide_axes(ax[1])

    # Right: Prediction Panel (bar chart)
    ax[2].barh(range(topk), topk_vals)
    ax[2].set_yticks(range(topk), labels=topk_labels)
    ax[2].invert_yaxis()
    ax[2].set_xlim(0, 100)
    ax[2].set_xlabel("Confidence (%)")
    ax[2].grid(axis="x", linestyle=":", linewidth=0.5)
    pred_label = class_names[pred_idx]
    ax[2].set_title(f"Prediction: {pred_label} ({probs[pred_idx] * 100:.1f}%)", pad=6)

    if main_title:
        fig.suptitle(main_title, y=0.98)
    fig.tight_layout()
    return fig


def save_fig(fig: "matplotlib.figure.Figure", out_path: Path, dpi: int = 200):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.as_posix(), bbox_inches="tight", pad_inches=0.15, dpi=dpi, facecolor="white")
    plt.close(fig)


# Image helpers
def get_canny_edge(img_rgb01: np.ndarray, threshold1: int = 30, threshold2: int = 80) -> np.ndarray:
    """
    Compute white-on-black Canny edges for an RGB float image in [0,1].
    Returns (H,W,3) float in [0,1].
    """
    gray = cv2.cvtColor((img_rgb01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = 255 - cv2.Canny(gray, threshold1, threshold2)
    edge = np.stack([edge] * 3, axis=-1) / 255.0
    return edge


def main():
    # ---- Paths ----
    base_path = Path(r"../outputs")
    ckpt_path = base_path / "best.pt"

    # Input images: exported hard examples (PNG/JPG/etc.)
    hard_examples_dir = base_path / "hard_examples" / "overconfident_wrong"

    # Outputs: clean figures
    out_cam_dir = base_path / "hard_examples" / "overconfident_wrong_cam"
    out_cam3_dir = base_path / "hard_examples" / "overconfident_wrong_cam3"
    out_edges_dir = base_path / "hard_examples" / "overconfident_wrong_edges"
    ensure_dir(out_cam_dir)
    ensure_dir(out_cam3_dir)
    ensure_dir(out_edges_dir)

    # ---- Repro / device ----
    set_seed(42)
    device = get_device()

    model, meta = load_model(ckpt_path, device)

    # class_names = meta["class_names"]
    # species_list = meta["species_list"]
    disease_list = meta["disease_list"]
    ckpt_args = meta["args"]
    multitask = ckpt_args.get("multitask", False)
    mean = parse_floats(ckpt_args.get("mean", '0.485,0.456,0.406'))
    std = parse_floats(ckpt_args.get("std", '0.229,0.224,0.225'))

    # Wrap for single-head CAM if multitask (use disease head by default)
    cam_model = CamSingleHead(model, task="disease").to(device) if multitask else model

    # ---- Grad-CAM target layer ----
    target_layer = resolve_target_layer(cam_model, target_layer_path=None)
    cam = GradCAM(model=cam_model, target_layers=[target_layer])

    # ---- Preprocess / Denorm ----
    transform = build_transforms(img_size=224, mode="val")

    # ---- Gather files from hard_examples dir ----
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    image_files: List[Path] = []
    for pat in exts:
        image_files.extend(hard_examples_dir.glob(pat))
    image_files = sorted(image_files)

    if not image_files:
        print(f"No images found in: {hard_examples_dir}")
        return

    print(f"Found {len(image_files)} images in {hard_examples_dir}")

    # ---- Process all images ----
    for img_path in image_files:
        try:
            # Load & preprocess
            pil = Image.open(img_path).convert("RGB")
            x = transform(pil).unsqueeze(0).to(device)

            # Inference (use disease head for multitask)
            with torch.no_grad():
                if multitask:
                    logits = model(x)[2]  # disease logits
                else:
                    logits = model(x)

            # Predictions
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (C,)
            pred_idx = int(np.argmax(probs))
            print(f"{disease_list[pred_idx]}: {probs[pred_idx]}")

            pred_name = disease_list[pred_idx]
            # Denormalize for visualization
            img_vis = denorm_to_rgb(x, mean, list(std))  # (H,W,3) in [0,1]

            # Optional: Canny edges
            edges = get_canny_edge(img_vis)

            # Grad-CAM overlay (predicted class)
            grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred_idx)])[0]
            if grayscale_cam is None:
                raise ValueError("Grad-CAM returned None. Check target layer or model output.")

            cam_overlay = overlay_heatmap(edges, grayscale_cam, alpha=0.25)

            # ----- Choose LEFT title (class name) -----
            left_title = choose_left_title(img_path.stem, model_pred=pred_name)

            # --- Save ClassName | Grad-CAM ---
            fig_cam = plot_side_by_side(
                img_vis,
                cam_overlay,
                left_title=left_title,
                right_title="Grad-CAM Overlay",
                main_title=f"Pred: {pred_name}"
            )
            out_cam_path = out_cam_dir / f"{img_path.stem}__pred={pred_name}__cam.png"
            save_fig(fig_cam, out_cam_path, dpi=200)

            # --- Save ClassName | Grad-CAM | Prediction Panel ---
            fig_cam3 = plot_triptych(
                img_vis,
                cam_overlay,
                probs=probs,
                class_names=disease_list,
                pred_idx=pred_idx,
                left_title=left_title,
                mid_title="Grad-CAM Overlay",
                topk=5,
                main_title=None
            )
            out_cam3_path = out_cam3_dir / f"{img_path.stem}__pred={pred_name}__cam_predpanel.png"
            save_fig(fig_cam3, out_cam3_path, dpi=200)

            # --- Save ClassName | Canny Edges ---
            fig_edges = plot_side_by_side(
                img_vis,
                edges,
                left_title=left_title,
                right_title="Canny Edges",
                main_title=None
            )
            out_edge_path = out_edges_dir / f"{img_path.stem}__edges.png"
            save_fig(fig_edges, out_edge_path, dpi=200)

            print(f"[OK] {img_path.name}  ->  {out_cam_path.name}, {out_cam3_path.name}, {out_edge_path.name}")

        except Exception as e:
            print(f"[ERR] {img_path.name}: {e}")

    print(f"\nDone.\n  CAM overlays: {out_cam_dir}\n  Triptychs: {out_cam3_dir}\n  Canny edges: {out_edges_dir}")


if __name__ == "__main__":
    main()
