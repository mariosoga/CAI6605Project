from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple, Optional

from sklearn.metrics import classification_report, confusion_matrix

# Local utils
from .utils import plot_confusion_matrix, ensure_dir


class EvaluationReporter:

    def __init__(self, out_dir: Path, binary_labels: Tuple[str, str] = ("Sick", "Healthy"),
                 default_prefix: str = "") -> None:
        self.out_dir = Path(out_dir)
        self.binary_labels = binary_labels
        self.default_prefix = default_prefix
        ensure_dir(self.out_dir)

    # ------------------------------
    # Reports
    # ------------------------------

    def write_classification_report(
            self, y_true, y_pred, class_names: List[str], out_name: str = "classification_report.txt",
            labels: Optional[Iterable[int]] = None
    ) -> str:

        lbls = labels or list(range(len(class_names)))
        report = classification_report(
            y_true, y_pred, target_names=class_names, labels=list(lbls), digits=4, zero_division=0)
        (self.out_dir / out_name).write_text(report)
        return report

    def write_binary_report(
            self, y_true, y_pred, out_name: str = "classification_report_binary.txt",
            labels: Optional[Tuple[str, str]] = None
    ) -> str:

        lbls = labels or self.binary_labels
        report = classification_report(
            y_true, y_pred, target_names=list(lbls), labels=[0, 1], digits=4, zero_division=0
        )
        (self.out_dir / out_name).write_text(report)
        return report

    # ------------------------------
    # Confusion Matrix
    # ------------------------------

    def save_confusion_matrix_img(
            self, y_true, y_pred, class_names: List[str], prefix: Optional[str] = None, out_name: Optional[str] = None
    ) -> Path:
        pfx = self._prefix(prefix)
        cm_name = out_name or f"{pfx}confusion_matrix.png"
        cm_path = self.out_dir / cm_name
        plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
        return cm_path

    # ------------------------------
    # CSV Exports
    # ------------------------------

    def export_val_predictions_csv(
            self, records: List[Dict[str, Any]], class_names: List[str], out_name: str = "val_predictions.csv"
    ) -> Path:
        out_csv = self.out_dir / out_name
        import csv
        with out_csv.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["path", "true_label", "pred_label", "pred_conf", "true_conf", "top2_label", "top2_conf"])
            for r in records:
                writer.writerow([
                    str(r["path"]),
                    class_names[r["true_idx"]],
                    class_names[r["pred_idx"]],
                    f"{r.get('pred_conf', 0.0):.6f}",
                    f"{r.get('true_conf', 0.0):.6f}",
                    class_names[r["top2_idx"]] if "top2_idx" in r else "",
                    f"{r.get('top2_conf', 0.0):.6f}" if "top2_conf" in r else "",
                ])
        return out_csv

    def export_val_predictions_csv_binary(
            self, records: List[Dict[str, Any]], out_name: str = "val_predictions_binary.csv",
            labels: Optional[Tuple[str, str]] = None
    ) -> Path:

        lbls = labels or self.binary_labels
        out_csv = self.out_dir / out_name
        import csv
        with out_csv.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["path", "true_label", "pred_label", "pred_conf", "true_conf"])
            for r in records:
                writer.writerow([
                    str(r["path"]),
                    lbls[r["true_idx"]],
                    lbls[r["pred_idx"]],
                    f"{r.get('pred_conf', 0.0):.6f}",
                    f"{r.get('true_conf', 0.0):.6f}",
                ])
        return out_csv

    # ------------------------------
    # Hard Examples
    # ------------------------------

    def save_hard_examples(
            self, records: List[Dict[str, Any]], class_names: List[str], k: int = 50, prefix: Optional[str] = None
    ) -> None:

        import shutil

        pfx = self._prefix(prefix)
        # Overconfident wrong
        wrong = [r for r in records if r["pred_idx"] != r["true_idx"]]
        wrong.sort(key=lambda r: r.get("pred_conf", 0.0), reverse=True)

        ow_dir = self.out_dir / "hard_examples" / f"{pfx}overconfident_wrong"
        ensure_dir(ow_dir)
        for r in wrong[:k]:
            src = Path(r["path"])
            name = f"true={class_names[r['true_idx']]}__pred={class_names[r['pred_idx']]}__p={r.get('pred_conf', 0.0):.3f}{src.suffix}"
            dst = ow_dir / name
            self._safe_copy(src, dst, shutil)

        # Uncertain correct
        correct = [r for r in records if r["pred_idx"] == r["true_idx"]]
        correct.sort(key=lambda r: r.get("true_conf", 0.0))  # ascending -> most uncertain first

        uc_dir = self.out_dir / "hard_examples" / f"{pfx}uncertain_correct"
        ensure_dir(uc_dir)
        for r in correct[:k]:
            src = Path(r["path"])
            name = f"true={class_names[r['true_idx']]}__pred={class_names[r['pred_idx']]}__p={r.get('true_conf', 0.0):.3f}{src.suffix}"
            dst = uc_dir / name
            self._safe_copy(src, dst, shutil)

    def save_hard_examples_binary(
            self, records: List[Dict[str, Any]], k: int = 50, labels: Optional[Tuple[str, str]] = None) -> None:

        import shutil

        lbls = labels or self.binary_labels

        # Overconfident wrong
        wrong = [r for r in records if r["pred_idx"] != r["true_idx"]]
        wrong.sort(key=lambda r: r.get("pred_conf", 0.0), reverse=True)

        ow_dir = self.out_dir / "hard_examples_binary" / "overconfident_wrong"
        ensure_dir(ow_dir)
        for r in wrong[:k]:
            src = Path(r["path"])
            name = f"true={lbls[r['true_idx']]}__pred={lbls[r['pred_idx']]}__p={r.get('pred_conf', 0.0):.3f}{src.suffix}"
            dst = ow_dir / name
            self._safe_copy(src, dst, shutil)

        # Uncertain correct
        correct = [r for r in records if r["pred_idx"] == r["true_idx"]]
        correct.sort(key=lambda r: r.get("true_conf", 0.0))

        uc_dir = self.out_dir / "hard_examples_binary" / "uncertain_correct"
        ensure_dir(uc_dir)
        for r in correct[:k]:
            src = Path(r["path"])
            name = f"true={lbls[r['true_idx']]}__pred={lbls[r['pred_idx']]}__p={r.get('true_conf', 0.0):.3f}{src.suffix}"
            dst = uc_dir / name
            self._safe_copy(src, dst, shutil)

    # ------------------------------
    # Confusion Pairs
    # ------------------------------

    def save_top_confusion_pairs(
            self, y_true, y_pred, class_names: List[str], records: List[Dict[str, Any]], m: int = 5,
            examples_per_pair: int = 8
    ) -> None:
        import shutil

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        pairs: List[Tuple[Tuple[int, int], int]] = []

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i == j:
                    continue
                count = int(cm[i, j])
                if count > 0:
                    pairs.append(((i, j), count))

        pairs.sort(key=lambda x: x[1], reverse=True)
        top = pairs[:m]

        pairs_dir = self.out_dir / "confusions"
        ensure_dir(pairs_dir)
        with (pairs_dir / "top_pairs.txt").open('w', encoding="utf-8") as f:
            import numpy as np
            y_true_arr = np.asarray(y_true)
            for (i, j), cnt in top:
                support_i = int((y_true_arr == i).sum())
                rate = (cnt / max(1, support_i)) * 100.0
                f.write(f"{class_names[i]} -> {class_names[j]}: {cnt} / {support_i} ({rate:.1f}%)\n")

        # Copy examples per confusion pair
        for (i, j), _cnt in top:
            pair_dir = pairs_dir / f"{class_names[i]}__to__{class_names[j]}"
            ensure_dir(pair_dir)
            examples = [r for r in records if r["true_idx"] == i and r["pred_idx"] == j]
            examples.sort(key=lambda r: r.get("pred_conf", 0.0), reverse=True)
            for r in examples[:examples_per_pair]:
                src = Path(r["path"])
                name = f"p={r.get('pred_conf', 0.0):.3f}__true={class_names[i]}__pred={class_names[j]}{src.suffix}"
                dst = pair_dir / name
                self._safe_copy(src, dst, shutil)

    # ------------------------------
    # Helpers
    # ------------------------------

    def _prefix(self, prefix: Optional[str]) -> str:
        """Resolve an explicit prefix or fall back to the default."""
        return (prefix if prefix is not None else self.default_prefix) or ""

    @staticmethod
    def _safe_copy(src: Path, dst: Path, shutil_module) -> None:
        try:
            shutil_module.copy2(src, dst)
        except Exception:
            # Swallow copy errors (missing source, permissions, etc.)
            pass
