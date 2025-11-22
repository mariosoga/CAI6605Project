# python3 -m scripts.fgsm --ckpt outputs/best.pt --data_dir data/dataset/test --out_dir outputs/fgsm_test

# === FGSM evaluation summary ===
# Total samples evaluated: 1561, total skipped: 0
# epsilon=0.000 -> accuracy: 74.89%  (167/223)
# epsilon=0.050 -> accuracy: 4.93%  (11/223)
# epsilon=0.100 -> accuracy: 13.90%  (31/223)
# epsilon=0.150 -> accuracy: 23.77%  (53/223)
# epsilon=0.200 -> accuracy: 23.32%  (52/223)
# epsilon=0.250 -> accuracy: 18.39%  (41/223)
# epsilon=0.300 -> accuracy: 13.90%  (31/223)


import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from core.data import FolderLabeledDataset
from core.utils import load_model, get_device, parse_floats, ensure_dir, set_seed
import re
_norm_rx_multi_underscore = re.compile(r"_+")


def normalize_name(raw_name):
    cleaned_name = str(raw_name).strip()
    cleaned_name = cleaned_name.replace("-", " ")
    cleaned_name = cleaned_name.replace("/", " ")
    
    words_joined_by_underscore = "_".join(cleaned_name.split())
    normalized_with_single_underscores = _norm_rx_multi_underscore.sub("_", words_joined_by_underscore)

    normalized_final = normalized_with_single_underscores.lower()
    return normalized_final

def build_norm_maps(original_names):
    normalized_to_original: Dict[str, str] = {}
    original_to_index: Dict[str, int] = {}

    if not original_names:
        return normalized_to_original, original_to_index

    for index, original_name in enumerate(original_names):

        original_to_index[original_name] = index

        normalized_name = normalize_name(original_name)

        if normalized_name not in normalized_to_original:
            normalized_to_original[normalized_name] = original_name

    return normalized_to_original, original_to_index

def find_idx_from_norm(raw_label, normalized_to_original, original_to_index):
    if raw_label is None:
        return None

    normalized_label = normalize_name(raw_label)

    canonical_name = normalized_to_original.get(normalized_label)
    if canonical_name is None:
        return None

    return original_to_index.get(canonical_name)

def unnormalize_tensor(normalized_image, channel_means,channel_stds):
    mean_tensor = torch.tensor(channel_means, dtype=normalized_image.dtype, device=normalized_image.device).view(-1, 1, 1)
    std_tensor = torch.tensor(channel_stds, dtype=normalized_image.dtype, device=normalized_image.device).view(-1, 1, 1)
    
    unnormalized_image = normalized_image * std_tensor + mean_tensor

    return unnormalized_image

def save_image_tensor_to_file(image_tensor, output_path) :
    image_cpu = image_tensor.detach().cpu().clamp(0.0, 1.0)

    image_hwc = image_cpu.permute(1, 2, 0).numpy()

    image_uint8 = (image_hwc * 255.0).round().astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(image_uint8).save(str(output_path))

# Generate FGSM adversarial images for a test dataset and evaluate model accuracy.
def run_fgsm(checkpoint_path, test_root, output_root, epsilons: List[float], batch_size, seed,target_head = "auto"):
    set_seed(seed)
    device = get_device()
    print(f"[info] device: {device}")

    if not checkpoint_path.exists():
        candidates = list(checkpoint_path.parent.glob("*.pth")) + \
                     list(checkpoint_path.parent.glob("*.pt")) + \
                     list(checkpoint_path.parent.glob("*.safetensors"))
        if not candidates and Path("outputs").exists():
            candidates = list(Path("outputs").glob("*.pth")) + \
                         list(Path("outputs").glob("*.pt")) + \
                         list(Path("outputs").glob("*.safetensors"))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            chosen = candidates[0]
            print(f"[warning] requested checkpoint '{checkpoint_path}' not found; using '{chosen}'")
            checkpoint_path = chosen
        else:
            raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found and no alternatives in outputs/")

    # Load model and metadata
    model, meta = load_model(checkpoint_path, device)
    model.eval()

    ckpt_args = (meta.get("args") or {})
    mean = parse_floats(ckpt_args.get("norm_mean") or ckpt_args.get("mean") or [0.485, 0.456, 0.406])
    std = parse_floats(ckpt_args.get("norm_std") or ckpt_args.get("std") or [0.229, 0.224, 0.225])
    img_size = int(ckpt_args.get("img_size", 224))

    species_list = meta.get("species_list")
    disease_list = meta.get("disease_list")
    class_names = meta.get("class_names")
    multitask = (species_list is not None and disease_list is not None)
    print(f"[info] multitask={multitask}, img_size={img_size}")

    # Decide which head to attack
    if multitask:
        head_to_attack = "species" if target_head == "auto" else target_head
    else:
        head_to_attack = "single"
    print(f"[info] attacking head: {head_to_attack}")


    # Prepare dataset + loaders
    dataset = FolderLabeledDataset(test_root, img_size, mean, std, multitask=multitask)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    baseline_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    output_root.mkdir(parents=True, exist_ok=True)
    epsilons = sorted(epsilons)

    if multitask:
        normalized_to_original, original_to_index = build_norm_maps(species_list)
    else:
        normalized_to_original, original_to_index = build_norm_maps(class_names)

    # Baseline evaluation (no perturbation)
    print("[info] computing baseline accuracy on same dataset (no perturbation)...")
    baseline_stats = {"correct": 0, "total": 0, "skipped": 0}
    model.eval()
    with torch.no_grad():
        for images, paths, truth in baseline_loader:
            images = images.to(device)
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            batch_size_actual = images.shape[0]

            for b in range(batch_size_actual):
                if multitask:
                    label_raw = truth["species"][b]
                    mapped_idx = find_idx_from_norm(label_raw, normalized_to_original, original_to_index)
                else:
                    label_raw = truth["combined"][b]
                    mapped_idx = find_idx_from_norm(label_raw, normalized_to_original, original_to_index)

                if mapped_idx is None:
                    baseline_stats["skipped"] += 1
                    continue

                baseline_stats["total"] += 1
                if int(preds[b]) == int(mapped_idx):
                    baseline_stats["correct"] += 1

    total_baseline = baseline_stats["total"]
    baseline_accuracy = 0.0 if total_baseline == 0 else 100.0 * baseline_stats["correct"] / total_baseline
    print(f"[info] baseline: {baseline_stats['correct']}/{total_baseline} = {baseline_accuracy:.2f}% (skipped={baseline_stats['skipped']})")

    # FGSM attack 
    acc_by_epsilon: Dict[float, Dict[str, int]] = {e: {"correct": 0, "total": 0} for e in epsilons}
    total_skipped = 0
    total_evaluated = 0

    for batch_idx, (images, paths, truth) in enumerate(eval_loader):
        images = images.to(device)
        batch_len = images.shape[0]

        valid_local_indices: List[int] = []
        ground_truth_indices: List[int] = []
        for b in range(batch_len):
            if multitask:
                label_raw = truth["species"][b]
                mapped_idx = find_idx_from_norm(label_raw, normalized_to_original, original_to_index)
            else:
                label_raw = truth["combined"][b]
                mapped_idx = find_idx_from_norm(label_raw, normalized_to_original, original_to_index)

            if mapped_idx is None:
                continue

            valid_local_indices.append(b)
            ground_truth_indices.append(int(mapped_idx))

        if not valid_local_indices:
            total_skipped += batch_len
            continue

        images_selected = images[valid_local_indices].clone().detach().requires_grad_(True)
        targets_tensor = torch.tensor(ground_truth_indices, dtype=torch.long, device=device)

        model.zero_grad()
        outputs_selected = model(images_selected)
        logits_selected = outputs_selected[0] if isinstance(outputs_selected, (tuple, list)) else outputs_selected
        loss = F.cross_entropy(logits_selected, targets_tensor)
        loss.backward()
        grads_selected = images_selected.grad.data  

        std_tensor = torch.tensor(std, dtype=images.dtype, device=device).view(-1, 1, 1)
        mean_tensor = torch.tensor(mean, dtype=images.dtype, device=device).view(-1, 1, 1)

        for local_idx, original_batch_index in enumerate(valid_local_indices):
            original_image_norm = images_selected[local_idx].detach()
            signed_grad = grads_selected[local_idx].sign()
            image_path = Path(paths[original_batch_index])
            try:
                relative_path = image_path.relative_to(test_root)
            except Exception:
                relative_path = image_path.name

            for eps in epsilons:
                perturb_norm = (eps / std_tensor) * signed_grad
                perturbed_norm = original_image_norm + perturb_norm

                perturbed_pixel = unnormalize_tensor(perturbed_norm, mean, std)
                perturbed_pixel = torch.clamp(perturbed_pixel, 0.0, 1.0)
                perturbed_norm_clamped = (perturbed_pixel - mean_tensor) / std_tensor

                # Save perturbed image
                epsilon_subdir = output_root / f"eps_{eps:.3f}"
                save_dest = epsilon_subdir / relative_path
                save_image_tensor_to_file(perturbed_pixel, save_dest)

                with torch.no_grad():
                    inp = perturbed_norm_clamped.unsqueeze(0).to(device)
                    outp = model(inp)
                    logits_eval = outp[0] if isinstance(outp, (tuple, list)) else outp
                    predicted_index = int(torch.argmax(logits_eval, dim=-1).item())

                acc_by_epsilon[eps]["total"] += 1
                total_evaluated += 1
                if predicted_index == ground_truth_indices[local_idx]:
                    acc_by_epsilon[eps]["correct"] += 1

        images_selected.grad.detach_()
        images_selected.requires_grad_(False)
        model.zero_grad()

        if (batch_idx + 1) % 50 == 0:
            print(f"[info] processed {batch_idx + 1} batches")

    print("\n=== FGSM evaluation summary ===")
    print(f"Total samples evaluated: {total_evaluated}, total skipped: {total_skipped}")
    for eps in epsilons:
        total = acc_by_epsilon[eps]["total"]
        correct = acc_by_epsilon[eps]["correct"]
        accuracy_pct = 0.0 if total == 0 else 100.0 * correct / total
        print(f"epsilon={eps:.3f} -> accuracy: {accuracy_pct:.2f}%  ({correct}/{total})")

    csv_path = output_root / "fgsm_summary.csv"
    with open(csv_path, "w") as fh:
        fh.write("epsilon,correct,total,accuracy_pct\n")
        for eps in epsilons:
            total = acc_by_epsilon[eps]["total"]
            correct = acc_by_epsilon[eps]["correct"]
            accuracy_pct = 0.0 if total == 0 else 100.0 * correct / total
            fh.write(f"{eps},{correct},{total},{accuracy_pct:.4f}\n")

    print(f"[info] saved summary to {csv_path}")

def parse_eps_list(epsilons):
    epsilons = epsilons.strip()

    if epsilons.startswith("[") and epsilons.endswith("]"):
        epsilons = epsilons[1:-1]  

    raw_parts = epsilons.split(",")
    parts = []
    for part in raw_parts:
        cleaned = part.strip()
        if cleaned != "":
            parts.append(cleaned)

    floats = []
    for p in parts:
        floats.append(float(p))

    return floats

def main():
    
    # Example command: python3 -m scripts.fgsm --ckpt outputs/best.pt --data_dir data/dataset/test --out_dir outputs/fgsm_test
    p = argparse.ArgumentParser(description="Generate FGSM adversarial images for test set and evaluate accuracy.")
    p.add_argument("--ckpt", required=True, help="Best Model")
    p.add_argument("--data_dir", required=True, help="Root test folder (folder-per-class)")
    p.add_argument("--out_dir", default="outputs/fgsm", help="Folder Preturbed Images")
    p.add_argument("--epsilons", default="[0, .05, .1, .15, .2, .25, .3]", help="Epislon List")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    epsilons = parse_eps_list(args.epsilons)

    run_fgsm(ckpt_path, data_dir, out_dir, epsilons, batch_size=args.batch_size, seed=args.seed)

if __name__ == "__main__":
    main()
