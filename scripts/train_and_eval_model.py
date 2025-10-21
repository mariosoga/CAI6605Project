import argparse
import time
from pathlib import Path

import timm
import torch
from torch import nn

from core.PlantModel import MultiTaskEffNet
from core.data import make_datasets, build_loaders
from core.early_stop import EarlyStopper
from core.evaluate import evaluate, evaluate_mtl  # <-- use evaluate_mtl (3-head)
from core.evaluation_reporter import EvaluationReporter
from core.gradcam_utils import run_gradcam_for_hard_examples
from core.train import train, train_mtl  # <-- use train_mtl (3-head). keep train for single-task fallback
from core.utils import set_seed, get_device, ensure_dir, parse_floats


def main():
    data_dir = r"/data/dataset/color"

    parser = argparse.ArgumentParser(description="Smart Plant Observation Tool — Training Entry (Multi-task)")
    parser.add_argument('--data_dir', type=str, default=data_dir)
    parser.add_argument('--out_dir', type=str, default='outputs')
    parser.add_argument('--model', type=str, default='efficientnet_b0')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_split', type=float, default=0.15)

    # early stopping & scheduler
    parser.add_argument('--early_stop_patience', type=int, default=3)
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0)
    parser.add_argument('--lr_scheduler', type=str, choices=['plateau', 'none'], default='plateau')
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--lr_patience', type=int, default=2)
    parser.add_argument('--lr_min', type=float, default=1e-6)

    # multi-task
    parser.add_argument('--multitask', action='store_true', help='Enable multi-head (species, health, disease)')
    parser.add_argument('--healthy_keyword', type=str, default='healthy',
                        help="Substring to detect healthy classes (case-insensitive)")
    parser.add_argument('--health_loss_weight', type=float, default=1.0)

    # analysis options
    parser.add_argument('--save_hard_examples', action='store_true')
    parser.add_argument('--hard_k', type=int, default=50)
    parser.add_argument('--save_confusion_pairs', action='store_true')
    parser.add_argument('--pairs_m', type=int, default=5)
    parser.add_argument('--examples_per_pair', type=int, default=8)

    # Grad-CAM options
    parser.add_argument('--grad_cam', action='store_true', help='Generate Grad-CAM overlays for hard examples')
    parser.add_argument('--grad_cam_k', type=int, default=50, help='Top-K per bucket for Grad-CAM')
    parser.add_argument('--grad_cam_task', type=str, choices=['disease', 'health'], default='disease',
                        help='Which head to visualize (CamSingleHead supports disease/health)')
    parser.add_argument('--grad_cam_layer', type=str, default='',
                        help='Dotted conv layer path (e.g., "conv_head" or "backbone.conv_head"). Empty = auto.')
    parser.add_argument('--norm_mean', type=str, default='0.485,0.456,0.406', help='Preprocess mean (csv)')
    parser.add_argument('--norm_std', type=str, default='0.229,0.224,0.225', help='Preprocess std (csv)')
    parser.add_argument('--cam_alpha', type=float, default=0.45, help='Overlay alpha for heatmap')

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # -----------------------
    # Data
    # -----------------------
    train_ds, val_ds, train_sampler, val_sampler, idx_to_class = make_datasets(
        Path(args.data_dir), args.img_size,
        val_split=args.val_split, seed=args.seed,
        multitask=args.multitask,
        healthy_keyword=args.healthy_keyword
    )
    t_loader, v_loader = build_loaders(train_ds, val_ds, train_sampler, val_sampler, batch_size=args.batch_size)

    # Prepare class name lists
    if args.multitask:
        # disease_list includes 'healthy'; health is binary (0=Sick, 1=Healthy)
        species_list = getattr(train_ds, "species_list")
        disease_list = getattr(train_ds, "disease_list")
        health_names = ['Sick', 'Healthy']  # index 0 -> Sick, 1 -> Healthy
    else:
        # single-task: original combined labels
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        num_classes = len(class_names)

    # -----------------------
    # Model, loss, optim
    # -----------------------
    if args.multitask:
        num_species = len(species_list)
        num_diseases = len(disease_list)
        model = MultiTaskEffNet(args.model, num_s=num_species, num_d=num_diseases).to(device)

        species_criterion = nn.CrossEntropyLoss()
        health_criterion = nn.CrossEntropyLoss()
        disease_criterion = nn.CrossEntropyLoss()
    else:
        model = timm.create_model(args.model, pretrained=True, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler
    scheduler = None
    if args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience,
            threshold=1e-4, min_lr=args.lr_min
        )

    early_stopper = EarlyStopper(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        mode='min'
    )

    best_val_acc = 0.0  # disease acc for checkpointing (kept for continuity)
    start_time = time.time()

    # placeholders for reports
    records_h = []
    y_pred_h = y_true_h = None
    val_acc_h = None

    for epoch in range(1, args.epochs + 1):
        prev_lr = optimizer.param_groups[0]['lr']

        if args.multitask:
            # Train (3-head). Weights: species=1.0, health=args.health_loss_weight, disease=0.5
            train_metrics = train_mtl(
                model, t_loader, optimizer, device,
                w_species=1.0, w_health=args.health_loss_weight, w_disease=0.5, use_amp=True
            )
            # Validate
            val_metrics = evaluate_mtl(
                model, v_loader, species_criterion, health_criterion, disease_criterion, device,
                w_species=1.0, w_health=args.health_loss_weight, w_disease=0.5
            )

            v_loss = val_metrics["loss"]
            val_acc_d = val_metrics["disease"]["acc_overall"]
            val_acc_h = val_metrics["health"]["acc"]

            # keep for reports & gradcam
            records_d = val_metrics["disease"]["records"]
            records_h = val_metrics["health"]["records"]
            records_s = val_metrics["species"]["records"]
            y_true_d, y_pred_d = val_metrics["disease"]["y_true"], val_metrics["disease"]["y_pred"]
            y_true_h, y_pred_h = val_metrics["health"]["y_true"], val_metrics["health"]["y_pred"]
            y_true_s, y_pred_s = val_metrics["species"]["y_true"], val_metrics["species"]["y_pred"]

            cur_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch:02d}/{args.epochs} | lr={cur_lr:.2e} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"(sp={train_metrics['species_acc']:.3f} hl={train_metrics['health_acc']:.3f} "
                f"dz_all={train_metrics['disease_acc_overall']:.3f} dz_sick={train_metrics['disease_acc_sick_only']:.3f}) | "
                f"val_loss={v_loss:.4f} "
                f"(sp={val_metrics['species']['acc']:.3f} hl={val_acc_h:.3f} "
                f"dz_all={val_acc_d:.3f} dz_sick={val_metrics['disease']['acc_sick_only']:.3f}) | "
                f"elapsed={time.time() - start_time:.2f}s"
            )

            # choose disease acc for checkpointing (same behavior as before)
            val_acc = val_acc_d

        else:
            # Single-task (legacy)
            t_loss, train_acc = train(model, t_loader, criterion, optimizer, device)
            v_loss, val_acc, y_true_d, y_pred_d, records_d = evaluate(model, v_loader, criterion, device)
            cur_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch:02d}/{args.epochs} | lr={cur_lr:.2e} | "
                f"train_loss={t_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={v_loss:.4f} acc={val_acc:.4f} | "
                f"elapsed={time.time() - start_time:.2f}s"
            )

        # Scheduler step
        if scheduler is not None:
            scheduler.step(v_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < prev_lr - 1e-12:
                print(f"ReduceLROnPlateau: lr {prev_lr:.2e} -> {new_lr:.2e}")

        # Checkpoint
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                'model_state': model.state_dict(),
                # for backward compatibility, store disease names under 'class_names'
                'class_names': (
                    disease_list if args.multitask else [idx_to_class[i] for i in range(len(idx_to_class))]),
                'args': vars(args),
            }
            if args.multitask:
                ckpt['species_list'] = species_list
                ckpt['disease_list'] = disease_list
            torch.save(ckpt, out_dir / 'best.pt')

        if early_stopper.step(v_loss):
            print(f"Early stopping triggered after epoch {epoch}. Best val (disease) acc: {best_val_acc:.4f}")
            break

    print(f"elapsed time: {time.time() - start_time:.2f}s")

    # -----------------------
    # Reports (use disease for multiclass report)
    # -----------------------
    # Instantiate once per evaluation run (you can pass a default prefix like 'disease_' or 'species_')
    reporter = EvaluationReporter(out_dir / "run_001")

    # Multiclass report + CM + CSV
    reporter.write_classification_report(y_true_d, y_pred_d, class_names=class_names,
                                         out_name="classification_report_disease.txt")


    reporter.save_confusion_matrix_img(y_true_d, y_pred_d, class_names=class_names,
                                       prefix="disease_")  # uses default_prefix

    reporter.export_val_predictions_csv(records_d, class_names=class_names, out_name="val_predictions_disease.csv")

    # --- Health binary report (if multitask) ---
    if args.multitask and y_true_h is not None and y_pred_h is not None:
        # Binary report + CSV
        reporter.write_binary_report(y_true_h, y_pred_h, out_name="classification_report_health.txt")
        reporter.export_val_predictions_csv_binary(records_h, out_name="val_predictions_health.csv")

    if y_true_s is not None and y_pred_s is not None:

        reporter.write_classification_report(y_true_s, y_pred_s, class_names=species_list,
                                             out_name="classification_report_species.txt")

        reporter.save_confusion_matrix_img(y_true_s, y_pred_s, class_names=species_list, prefix="species_")
        reporter.export_val_predictions_csv(records_s, class_names=species_list, out_name="val_predictions_species.csv")

    # --- Hard examples ---
    if args.save_hard_examples:
        # Hard examples
        reporter.save_hard_examples(records_d, class_names=class_names, k=args.hard_k, prefix="disease_")

        if args.multitask:
            reporter.save_hard_examples_binary(records_h, k=args.hard_k)  # binary
        if y_true_s is not None and y_pred_s is not None:
            reporter.save_hard_examples(records_s, class_names=species_list, k=args.hard_k, prefix="species_")

    if args.save_confusion_pairs:
        reporter.save_top_confusion_pairs(
            y_true_d, y_pred_d, class_names=class_names, records=records_d, m=args.pairs_m, examples_per_pair=args.examples_per_pair)

    # -----------------------
    # Grad-CAM on hard examples
    # -----------------------
    if args.grad_cam:
        mean = parse_floats(args.norm_mean)
        std = parse_floats(args.norm_std)

        if args.multitask:
            task = args.grad_cam_task  # {'disease','species', 'health'}
            if task == 'disease':
                use_records = records_d
                use_class_names = disease_list
            elif task == 'species':
                use_records = records_s
                use_class_names = species_list
            else:
                use_records = records_h
                use_class_names = ['Sick', 'Healthy']  # indexes: 0,1
        else:
            task = 'disease'
            use_records = records_d
            use_class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

        print(f"[Grad-CAM] Generating overlays for hard examples (task={task})...")
        run_gradcam_for_hard_examples(
            model=model,
            records=use_records,
            class_names=use_class_names,
            out_dir=out_dir,
            k=args.grad_cam_k,
            img_size=args.img_size,
            mean=mean,
            std=std,
            cam_alpha=args.cam_alpha,
            target_layer_path=args.grad_cam_layer.strip() or None,
            multitask=args.multitask,
            task=task
        )

    print(f"Done. Best (disease) val acc: {best_val_acc:.4f}. Artifacts -> {out_dir}")

if __name__ == '__main__':
    main()
