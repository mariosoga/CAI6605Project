import argparse
from pathlib import Path

import torch
from PIL import Image

from core.utils import get_device, set_seed, load_model, build_transforms, parse_floats

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

@torch.no_grad()
def infer_single(model, multitask, img_path: Path, tfm, device, species_list, disease_list, class_names, topk: int = 3):
    img = Image.open(img_path).convert('RGB')
    x = tfm(img).unsqueeze(0).to(device)

    if multitask:
        sp_logits, hl_logits, dz_logits = model(x)
        sp_probs = torch.softmax(sp_logits, dim=1)[0]
        hl_probs = torch.softmax(hl_logits, dim=1)[0]  # health: 0=Sick, 1=Healthy
        dz_probs = torch.softmax(dz_logits, dim=1)[0]

        sp_topk = torch.topk(sp_probs, k=min(topk, sp_probs.numel()))
        dz_topk = torch.topk(dz_probs, k=min(topk, dz_probs.numel()))

        sp_idx = sp_probs.argmax().item()
        hl_idx = hl_probs.argmax().item()  # 1=Healthy
        pred_health = "Healthy" if hl_idx == 1 else "Sick"

        line = f"[MULTI] {img_path.name}  ->  Species: {species_list[sp_idx]}  |  Health: {pred_health} ({hl_probs[hl_idx]:.3f})"
        if hl_idx == 0:  # Sick -> show disease
            dz_idx = dz_probs.argmax().item()
            line += f"  |  Disease: {disease_list[dz_idx]} ({dz_probs[dz_idx]:.3f})"
        print(line)

        # Pretty top-k
        sp_k = ", ".join([f"{species_list[i]}:{sp_probs[i]:.2f}" for i in sp_topk.indices.tolist()])
        dz_k = ", ".join([f"{disease_list[i]}:{dz_probs[i]:.2f}" for i in dz_topk.indices.tolist()])
        print(f"    species@{len(sp_topk.indices)} -> {sp_k}")
        print(f"    disease@{len(dz_topk.indices)} -> {dz_k}")
    else:
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top = torch.topk(probs, k=min(topk, probs.numel()))
        p_idx = probs.argmax().item()
        print(f"[SINGLE] {img_path.name}  ->  Class: {class_names[p_idx]} ({probs[p_idx]:.3f})")
        top_str = ", ".join([f"{class_names[i]}:{probs[i]:.2f}" for i in top.indices.tolist()])
        print(f"    top@{len(top.indices)} -> {top_str}")


def images_from_dir(d: Path):
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def main():
    ap = argparse.ArgumentParser("Test/Inference for Smart Plant Care Assistant")
    ap.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint (best.pt)')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--image_path', type=str, help='Path to a single image')
    g.add_argument('--image_dir', type=str, help='Path to a directory of images (recursively)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--topk', type=int, default=3)
    args = ap.parse_args()

    set_seed(args.seed)

    device = get_device()

    image_path = args.image_path
    image_dir = args.image_dir

    ckpt_path = Path(args.ckpt)
    model, meta = load_model(ckpt_path, device)
    ckpt_args = (meta.get("args") or {})

    mean = parse_floats(ckpt_args.get("norm_mean") or ckpt_args.get("mean") or [0.485, 0.456, 0.406])
    std = parse_floats(ckpt_args.get("norm_std") or ckpt_args.get("std") or [0.229, 0.224, 0.225])
    species_list = meta['species_list']
    disease_list = meta['disease_list']
    class_names = meta['class_names']

    chkpt_args = meta.get("args", {}) or {}
    multitask = chkpt_args['multitask']

    img_size = int(ckpt_args.get("img_size", 224))
    batch_size = int(ckpt_args.get("batch_size", 32))

    tfm = build_transforms(img_size, "val", mean, std)

    paths = []
    if image_path is not None:
        paths = [Path(image_path)]
    else:
        d = Path(image_dir)
        paths = list(images_from_dir(d))
        if not paths:
            print(f"No images found in {d}")
            return

    print(f"Loaded model from: {ckpt_path}")
    if multitask:
        print(
            f"Heads: species={len(species_list)} | health=2 | disease={len(disease_list)}  (health: 0=Sick,1=Healthy)\n")
    else:
        print(f"Single-task classes: {len(class_names)}\n")

    for p in paths:
        try:
            infer_single(model, multitask, p, tfm, device, species_list, disease_list, class_names, topk=args.topk)
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")


if __name__ == '__main__':
    main()
