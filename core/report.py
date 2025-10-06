from pathlib import Path
from typing import List, Dict, Any
from sklearn.metrics import classification_report, confusion_matrix
from .utils import plot_confusion_matrix


def write_classification_report(y_true, y_pred, class_names: List[str], out_path: Path):
    # Explicit labels to lock the class order and avoid "missing class" issues
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        labels=labels,
        digits=4,
        zero_division=0
    )
    out_path.write_text(report)
    return report


def write_binary_report(
        y_true,
        y_pred,
        out_path: Path,
        # IMPORTANT: index 0 -> "Sick", index 1 -> "Healthy" (matches your dataset convention 0=Sick, 1=Healthy)
        labels=('Sick', 'Healthy')
):
    report = classification_report(
        y_true, y_pred,
        target_names=list(labels),
        labels=[0, 1],
        digits=4,
        zero_division=0
    )
    out_path.write_text(report)
    return report


def save_confusion_matrix_img(y_true, y_pred, class_names: List[str], out_dir: Path, prefix: str):
    cm_path = out_dir / f"{prefix}confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    return cm_path


def export_val_predictions_csv(records, class_names: List[str], out_csv: Path):
    with out_csv.open('w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "pred_label", "pred_conf", "true_conf", "top2_label", "top2_conf"])
        for r in records:
            writer.writerow([
                r["path"],
                class_names[r["true_idx"]],
                class_names[r["pred_idx"]],
                f"{r['pred_conf']:.6f}",
                f"{r['true_conf']:.6f}",
                class_names[r["top2_idx"]] if "top2_idx" in r else "",
                f"{r['top2_conf']:.6f}" if "top2_conf" in r else "",
            ])


def export_val_predictions_csv_binary(
        records,
        out_csv: Path,
        # Match convention: 0=Sick, 1=Healthy
        labels=('Sick', 'Healthy')
):
    with out_csv.open('w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "pred_label", "pred_conf", "true_conf"])
        for r in records:
            writer.writerow([
                r["path"],
                labels[r["true_idx"]],
                labels[r["pred_idx"]],
                f"{r['pred_conf']:.6f}",
                f"{r['true_conf']:.6f}",
            ])


def save_hard_examples(out_dir: Path, records: List[Dict[str, Any]], class_names: List[str], k: int = 50, prefix: str = ''):
    from .utils import ensure_dir
    import shutil
    wrong = [r for r in records if r["pred_idx"] != r["true_idx"]]
    wrong.sort(key=lambda r: r["pred_conf"], reverse=True)
    ow_dir = out_dir / "hard_examples" / f"{prefix}overconfident_wrong"
    ensure_dir(ow_dir)
    for r in wrong[:k]:
        src = Path(r["path"])
        name = f"true={class_names[r['true_idx']]}__pred={class_names[r['pred_idx']]}__p={r['pred_conf']:.3f}{src.suffix}"
        dst = ow_dir / name
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass

    correct = [r for r in records if r["pred_idx"] == r["true_idx"]]
    correct.sort(key=lambda r: r["true_conf"])
    uc_dir = out_dir / "hard_examples" / f"{prefix}uncertain_correct"
    ensure_dir(uc_dir)
    for r in correct[:k]:
        src = Path(r["path"])
        name = f"true={class_names[r['true_idx']]}__pred={class_names[r['pred_idx']]}__p={r['true_conf']:.3f}{src.suffix}"
        dst = uc_dir / name
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass


def save_hard_examples_binary(
        out_dir: Path,
        records: List[Dict[str, Any]],
        k: int = 50,
        # Match convention: 0=Sick, 1=Healthy
        labels=('Sick', 'Healthy')
):
    from .utils import ensure_dir
    import shutil
    wrong = [r for r in records if r["pred_idx"] != r["true_idx"]]
    wrong.sort(key=lambda r: r["pred_conf"], reverse=True)
    ow_dir = out_dir / "hard_examples_binary" / "overconfident_wrong"
    ensure_dir(ow_dir)
    for r in wrong[:k]:
        src = Path(r["path"])
        name = f"true={labels[r['true_idx']]}__pred={labels[r['pred_idx']]}__p={r['pred_conf']:.3f}{src.suffix}"
        dst = ow_dir / name
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass

    correct = [r for r in records if r["pred_idx"] == r["true_idx"]]
    correct.sort(key=lambda r: r["true_conf"])
    uc_dir = out_dir / "hard_examples_binary" / "uncertain_correct"
    ensure_dir(uc_dir)
    for r in correct[:k]:
        src = Path(r["path"])
        name = f"true={labels[r['true_idx']]}__pred={labels[r['pred_idx']]}__p={r['true_conf']:.3f}{src.suffix}"
        dst = uc_dir / name
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass


def save_top_confusion_pairs(out_dir: Path, y_true, y_pred, class_names: List[str], records: List[Dict[str, Any]],
                             m: int = 5, examples_per_pair: int = 8):
    from .utils import ensure_dir
    import shutil
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i == j: continue
            count = int(cm[i, j])
            if count > 0:
                pairs.append(((i, j), count))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:m]

    pairs_dir = out_dir / "confusions"
    ensure_dir(pairs_dir)
    with (pairs_dir / "top_pairs.txt").open('w') as f:
        for (i, j), cnt in top:
            support_i = int((y_true == i).sum())
            rate = (cnt / max(1, support_i)) * 100.0
            f.write(f"{class_names[i]} -> {class_names[j]}: {cnt} / {support_i} ({rate:.1f}%)\n")

    for (i, j), cnt in top:
        pair_dir = pairs_dir / f"{class_names[i]}__to__{class_names[j]}"
        ensure_dir(pair_dir)
        examples = [r for r in records if r["true_idx"] == i and r["pred_idx"] == j]
        examples.sort(key=lambda r: r["pred_conf"], reverse=True)
        for r in examples[:examples_per_pair]:
            src = Path(r["path"])
            name = f"p={r['pred_conf']:.3f}__true={class_names[i]}__pred={class_names[j]}{src.suffix}"
            dst = pair_dir / name
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass
