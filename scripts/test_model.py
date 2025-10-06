# test_model.py
import argparse
import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from sympy.testing.runtests import Reporter
from torch.utils.data import DataLoader

# Project imports
from core.data import FolderLabeledDataset
from core.evaluation_reporter import EvaluationReporter
from core.report import (
    write_classification_report, write_binary_report,
    save_confusion_matrix_img
)
from core.utils import get_device, ensure_dir, set_seed, parse_floats, load_model

# ---------- Helpers for robust label matching ----------

_norm_rx_multi_underscore = re.compile(r"_+")


def _normalize_name(s: str) -> str:
    """
    Normalize label strings to improve matching between dataset folder names
    and checkpoint label lists that may differ in case/spacing/hyphens.
    """
    s = str(s).strip()
    s = s.replace("-", " ").replace("/", " ")
    s = "_".join(s.split())  # collapse whitespace to single underscores
    s = _norm_rx_multi_underscore.sub("_", s)
    return s.lower()


def _build_norm_maps(names: Optional[List[str]]) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Given a list of canonical names (as saved in checkpoint), build:
      - norm_to_name: map from normalized string -> original canonical name
      - name_to_idx:  map from original canonical name -> index
    """
    norm_to_name: Dict[str, str] = {}
    name_to_idx: Dict[str, int] = {}
    if not names:
        return norm_to_name, name_to_idx
    for i, n in enumerate(names):
        name_to_idx[n] = i
        nn = _normalize_name(n)
        # keep first occurrence if duplicates normalize the same
        norm_to_name.setdefault(nn, n)
    return norm_to_name, name_to_idx


def _find_idx_from_norm(
        raw_label: str,
        norm_to_name: Dict[str, str],
        name_to_idx: Dict[str, int]
) -> Optional[int]:
    nn = _normalize_name(raw_label)
    canonical = norm_to_name.get(nn)
    if canonical is None:
        return None
    return name_to_idx.get(canonical)


# ---------- Top-k ----------

def _topk(prob: torch.Tensor, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (idxs, confs) for top-k per row.
    NOTE: torch.topk returns (values, indices). We swap to (indices, values).
    """
    k = min(k, prob.shape[-1])
    confs, idxs = torch.topk(prob, k=k, dim=-1)  # values first, indices second
    return idxs, confs


def _fmt_topk(idxs: List[int], confs: List[float], names: List[str]) -> str:
    out = []
    for i, c in zip(idxs, confs):
        name = names[i] if 0 <= i < len(names) else f"<{i}>"
        out.append(f"{name} ({c:.3f})")
    return " | ".join(out)


# ---------- Inference (single-task) ----------

@torch.no_grad()
def infer_and_eval_single(
        model,
        loader: DataLoader,
        device: torch.device,
        class_names: List[str],
        topk: int = 3,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    stats = {
        "seen": 0,
        "skipped_unknown_truth": 0,
    }

    class_norm_to_name, class_name_to_idx = _build_norm_maps(class_names)

    for batch, paths, truth in loader:
        batch = batch.to(device, non_blocking=True)
        logits = model(batch)
        prob = torch.softmax(logits, dim=-1)

        idxs, confs = _topk(prob, k=topk)

        for i, p in enumerate(paths):
            stats["seen"] += 1

            top_idxs = idxs[i].tolist()
            top_confs = [float(x) for x in confs[i].tolist()]
            pred_idx = int(top_idxs[0])
            pred_name = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else f"<{pred_idx}>"

            # Ground truth from dataset
            gt_raw = truth["combined"][i]
            gt_idx = _find_idx_from_norm(gt_raw, class_norm_to_name, class_name_to_idx)
            if gt_idx is None:
                stats["skipped_unknown_truth"] += 1
                continue

            y_true.append(gt_idx)
            y_pred.append(pred_idx)

            rows.append({
                "path": str(p),
                "true_label": class_names[gt_idx],
                "pred_label": pred_name,
                "pred_conf": top_confs[0],
                "topk": _fmt_topk(top_idxs, top_confs, class_names),
            })

    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)
    return y_true_arr, y_pred_arr, rows, stats


# ---------- Inference (multitask) ----------

@torch.no_grad()
def infer_and_eval_multitask(
        model,
        loader: DataLoader,
        device: torch.device,
        species_list: List[str],
        disease_list: List[str],
        health_labels: Optional[List[str]] = None,
        topk: int = 3,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, int]]:
    """
    Multitask heads: species (multiclass), health (binary), disease (multiclass)
    """
    if not health_labels:
        health_labels = ["Sick", "Healthy"]

    # Build normalized lookup maps for robust matching
    sp_norm_to_name, sp_name_to_idx = _build_norm_maps(species_list)
    dz_norm_to_name, dz_name_to_idx = _build_norm_maps(disease_list)
    hl_norm_to_name, hl_name_to_idx = _build_norm_maps(health_labels)

    def _safe_health_name(idx: int) -> str:
        return health_labels[idx] if 0 <= idx < len(health_labels) else f"<{idx}>"

    rows: List[Dict[str, Any]] = []
    y_true_d: List[int] = []
    y_pred_d: List[int] = []
    y_true_h: List[int] = []
    y_pred_h: List[int] = []
    y_true_s: List[int] = []
    y_pred_s: List[int] = []

    stats = {
        "seen": 0,
        "skipped_unknown_species": 0,
        "skipped_unknown_disease": 0,
        "skipped_unknown_health": 0,
    }

    for batch, paths, truth in loader:
        batch = batch.to(device, non_blocking=True)

        # Expect model to return three logits tensors (species, health, disease)
        sp_logits, hl_logits, dz_logits = model(batch)

        sp_prob = torch.softmax(sp_logits, dim=-1)
        hl_prob = torch.softmax(hl_logits, dim=-1)
        dz_prob = torch.softmax(dz_logits, dim=-1)

        sp_idxs, sp_confs = _topk(sp_prob, k=topk)
        dz_idxs, dz_confs = _topk(dz_prob, k=topk)
        hl_top = torch.argmax(hl_prob, dim=-1)
        hl_conf = torch.gather(hl_prob, 1, hl_top.unsqueeze(1)).squeeze(1)

        for i, p in enumerate(paths):
            stats["seen"] += 1

            # Predictions
            sp_top_idxs = sp_idxs[i].tolist()
            sp_top_confs = [float(x) for x in sp_confs[i].tolist()]
            dz_top_idxs = dz_idxs[i].tolist()
            dz_top_confs = [float(x) for x in dz_confs[i].tolist()]

            sp0 = int(sp_top_idxs[0])
            dz0 = int(dz_top_idxs[0])

            pred_species = species_list[sp0] if 0 <= sp0 < len(species_list) else f"<{sp0}>"
            pred_disease = disease_list[dz0] if 0 <= dz0 < len(disease_list) else f"<{dz0}>"

            pred_health_idx = int(hl_top[i].item())
            pred_health = _safe_health_name(pred_health_idx)
            pred_health_conf = float(hl_conf[i].item())

            # Ground truth (parsed from folder names by the dataset)
            gt_sp_raw = truth["species"][i]
            gt_dz_raw = truth["disease"][i]
            gt_hl_raw = truth["health"][i]  # e.g., "Sick" / "Healthy"

            gt_sp_idx = _find_idx_from_norm(gt_sp_raw, sp_norm_to_name, sp_name_to_idx)
            if gt_sp_idx is None:
                stats["skipped_unknown_species"] += 1
                continue

            gt_dz_idx = _find_idx_from_norm(gt_dz_raw, dz_norm_to_name, dz_name_to_idx)
            if gt_dz_idx is None:
                stats["skipped_unknown_disease"] += 1
                continue

            gt_hl_idx = _find_idx_from_norm(gt_hl_raw, hl_norm_to_name, hl_name_to_idx)
            if gt_hl_idx is None:
                stats["skipped_unknown_health"] += 1
                continue

            # Collect metrics
            y_true_d.append(gt_dz_idx)
            y_pred_d.append(dz0)

            y_true_h.append(gt_hl_idx)
            y_pred_h.append(pred_health_idx)

            y_true_s.append(gt_sp_idx)
            y_pred_s.append(sp0)

            rows.append({
                "path": str(p),

                "true_species": species_list[gt_sp_idx],
                "pred_species": pred_species,
                "species_conf": sp_top_confs[0],
                "species_topk": _fmt_topk(sp_top_idxs, sp_top_confs, species_list),

                "true_health": health_labels[gt_hl_idx],
                "pred_health": pred_health,
                "health_conf": pred_health_conf,

                "true_disease": disease_list[gt_dz_idx],
                "pred_disease": pred_disease,
                "disease_conf": dz_top_confs[0],
                "disease_topk": _fmt_topk(dz_top_idxs, dz_top_confs, disease_list),
            })

    y_true_d_arr = np.array(y_true_d, dtype=int)
    y_pred_d_arr = np.array(y_pred_d, dtype=int)
    y_true_h_arr = np.array(y_true_h, dtype=int)
    y_pred_h_arr = np.array(y_pred_h, dtype=int)
    y_pred_s_arr = np.array(y_pred_s, dtype=int)
    y_true_s_arr = np.array(y_true_s, dtype=int)

    return y_true_d_arr, y_pred_d_arr, y_true_h_arr, y_pred_h_arr, y_true_s_arr, y_pred_s_arr, rows, stats


# ---------- Main ----------

def main():
    p = argparse.ArgumentParser(description="Evaluate test set organized by label folders")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root folder containing subfolders named with ground-truth labels")
    p.add_argument("--out_dir", type=str, default="outputs/test_eval")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--healthy_keyword", type=str, default="healthy",
                   help="Case-insensitive token to mark 'Healthy' in folder names (for multitask health)")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    topk = int(args.topk)

    set_seed(args.seed)
    ensure_dir(out_dir)
    device = get_device()

    model, meta = load_model(ckpt_path, device)

    ckpt_args = (meta.get("args") or {})
    mean = parse_floats(ckpt_args.get("norm_mean") or ckpt_args.get("mean") or [0.485, 0.456, 0.406])
    std = parse_floats(ckpt_args.get("norm_std") or ckpt_args.get("std") or [0.229, 0.224, 0.225])
    img_size = int(ckpt_args.get("img_size", 224))
    batch_size = int(ckpt_args.get("batch_size", 32))

    class_names = meta.get("class_names")  # for single-task
    species_list = meta.get("species_list")
    disease_list = meta.get("disease_list")
    health_labels = meta.get("health_labels") or ["Sick", "Healthy"]

    # Determine multitask from presence of species & disease lists
    multitask = (species_list is not None and disease_list is not None)

    if multitask:
        print(f"Mode: multitask, img_size={img_size}, device={device}")
        ds = FolderLabeledDataset(
            data_dir, img_size, mean, std,
            multitask=True, healthy_keyword=args.healthy_keyword
        )
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        y_true_d, y_pred_d, y_true_h, y_pred_h, y_true_s, y_pred_s, rows, stats = infer_and_eval_multitask(
            model, dl, device,
            species_list=species_list,
            disease_list=disease_list,
            health_labels=health_labels,
            topk=topk
        )

        ensure_dir(out_dir)

        reporter = EvaluationReporter(out_dir)
        # Disease (multiclass) report + confusion matrix
        rpt = reporter.write_classification_report(y_true_d, y_pred_d, disease_list, "test_report_disease.txt")
        print(rpt)

        cm_path = reporter.save_confusion_matrix_img(y_true_d, y_pred_d, disease_list, "disease_")
        print(f"Saved disease confusion matrix to {cm_path}")

        # species (multiclass) report + confusion matrix
        rpt2 = reporter.write_classification_report(y_true_s, y_pred_s, species_list, "test_report_species.txt")
        print(rpt2)

        cm_path = reporter.save_confusion_matrix_img(y_true_s, y_pred_s, species_list, "species_")
        print(f"Saved species confusion matrix to {cm_path}")

        # Health (binary) report (no label_names arg in your API)
        rpt_h = reporter.write_classification_report(y_true_h, y_pred_h, health_labels, "test_report_health.txt")
        print(rpt_h)

    else:
        if not class_names:
            raise RuntimeError("Checkpoint missing 'class_names' for single-task evaluation.")
        print(f"Mode: single-task, img_size={img_size}, device={device}")
        ds = FolderLabeledDataset(data_dir, img_size, mean, std, multitask=False)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        y_true, y_pred, rows, stats = infer_and_eval_single(
            model, dl, device, class_names=class_names, topk=topk
        )

        ensure_dir(out_dir)

        # Reports
        rpt_path = out_dir / "test_report.txt"
        rpt = write_classification_report(y_true, y_pred, class_names, rpt_path)
        print(rpt)
        cm_path = save_confusion_matrix_img(y_true, y_pred, class_names, out_dir, "")
        print(f"Saved confusion matrix to {cm_path}")

        print("Eval artifacts in", out_dir)
        print(f"Stats: seen={stats['seen']}, skipped_unknown_truth={stats['skipped_unknown_truth']}")


if __name__ == "__main__":
    main()
