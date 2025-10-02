import argparse
import time
import timm
import torch
from pathlib import Path
from torch import nn

from core.train import train_one_epoch_single, train_one_epoch_multi
from core.utils import set_seed, get_device, ensure_dir
from core.data import make_datasets, build_loaders
from core.early_stop import EarlyStopper
from core.report import (write_classification_report, write_binary_report,
                         save_confusion_matrix_img, export_val_predictions_csv, export_val_predictions_csv_binary,
                         save_hard_examples, save_hard_examples_binary, save_top_confusion_pairs)
from core.evaluate import evaluate, evaluate_multitask


class MultiTaskEffNet(nn.Module):
    def __init__(self, model_name: str, num_disease_classes: int):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.disease_head = nn.Linear(feat_dim, num_disease_classes)
        self.health_head = nn.Linear(feat_dim, 2)

    def forward(self, x):
        feats = self.backbone(x)
        return self.disease_head(feats), self.health_head(feats)


def main():
    data_dir = "C:\\Users\\y-pol\\PyCharmMiscProject\\plant_care_assistant\\data\\plantvillage dataset\\color"
    parser = argparse.ArgumentParser(description="Smart Plant Care Assistant — Training Entry (Multi-task)")
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
    parser.add_argument('--multitask', action='store_true', help='Enable disease + healthy/sick heads')
    parser.add_argument('--healthy_keyword', type=str, default='healthy',
                        help="Substring to detect healthy classes (case-insensitive)")
    parser.add_argument('--health_loss_weight', type=float, default=0.5)

    # analysis options
    parser.add_argument('--save_hard_examples', action='store_true')
    parser.add_argument('--hard_k', type=int, default=50)
    parser.add_argument('--save_confusion_pairs', action='store_true')
    parser.add_argument('--pairs_m', type=int, default=5)
    parser.add_argument('--examples_per_pair', type=int, default=8)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Data
    train_ds, val_ds, train_sampler, val_sampler, idx_to_class = make_datasets(Path(args.data_dir), args.img_size,
                                                                               val_split=args.val_split, seed=args.seed,
                                                                               multitask=args.multitask,
                                                                               healthy_keyword=args.healthy_keyword)
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_names)
    t_loader, v_loader = build_loaders(train_ds, val_ds, train_sampler, val_sampler, batch_size=args.batch_size)

    # Model, loss, optim
    if args.multitask:
        model = MultiTaskEffNet(args.model, num_disease_classes=num_classes).to(device)
        d_criterion = nn.CrossEntropyLoss()
        h_criterion = nn.CrossEntropyLoss()
    else:
        model = timm.create_model(args.model, pretrained=True, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    scheduler = None
    if args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience,
            threshold=1e-4, min_lr=args.lr_min
        )

    early_stopper = EarlyStopper(patience=args.early_stop_patience, min_delta=args.early_stop_min_delta, mode='min')

    best_val_acc = 0.0  # disease acc for checkpointing
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        prev_lr = optimizer.param_groups[0]['lr']
        if args.multitask:
            t_loss, train_acc_d, train_acc_h = train_one_epoch_multi(model, t_loader, d_criterion, h_criterion,
                                                                     optimizer, device, args.health_loss_weight)
            (v_loss, v_acc_d, y_true_d, y_pred_d, records_d, val_acc_h, y_true_h, y_pred_h,
             records_h) = evaluate_multitask(model, v_loader, d_criterion, h_criterion, device, args.health_loss_weight)
            cur_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch:02d}/{args.epochs} |"
                f" lr={cur_lr:.2e} |"
                f" train_loss={t_loss:.4f} |"
                f" val_loss={v_loss:.4f} |"
                f" disease_acc={v_acc_d:.4f} |"
                f" health_acc={val_acc_h:.4f} |"
                f" elapsed_time={time.time() - start_time:.2f}")

            if scheduler is not None:
                scheduler.step(v_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr < prev_lr - 1e-12:
                    print(f"ReduceLROnPlateau: reducing learning rate {prev_lr:.2e} -> {new_lr:.2e}")

            if v_acc_d >= best_val_acc:
                best_val_acc = v_acc_d
                torch.save({'model_state': model.state_dict(), 'class_names': class_names, 'args': vars(args)},
                           out_dir / 'best.pt')

            if early_stopper.step(v_loss):
                print(f"Early stopping triggered after epoch {epoch}. Best disease val acc: {best_val_acc:.4f}")
                break
        else:
            t_loss, train_acc = train_one_epoch_single(model, t_loader, criterion, optimizer, device)
            v_loss, val_acc, y_true_d, y_pred_d, records_d = evaluate(model, v_loader, criterion, device)
            cur_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch:02d}/{args.epochs} |"
                f" lr={cur_lr:.2e} |"
                f" train_loss={t_loss:.4f} acc={train_acc:.4f} |"
                f" val_loss={v_loss:.4f} acc={val_acc:.4f} |"
                f"elapsed_time={time.time() - start_time:.2f}")

            if scheduler is not None:
                scheduler.step(v_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr < prev_lr - 1e-12:
                    print(f"ReduceLROnPlateau: reducing learning rate {prev_lr:.2e} -> {new_lr:.2e}")

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save({'model_state': model.state_dict(), 'class_names': class_names, 'args': vars(args)},
                           out_dir / 'best.pt')

            if early_stopper.step(v_loss):
                print(f"Early stopping triggered after epoch {epoch}. Best val acc so far: {best_val_acc:.4f}")
                break

    print(F"elapsed time: {time.time() - start_time:.2f}")

    # Reports
    if args.multitask:
        report_path = out_dir / 'classification_report_disease.txt'
        report = write_classification_report(y_true_d, y_pred_d, class_names, report_path)
        print(report)

        cm_path = save_confusion_matrix_img(y_true_d, y_pred_d, class_names, out_dir)
        print(f"Saved disease confusion matrix to {cm_path}")
        export_val_predictions_csv(records_d, class_names, out_dir / 'val_predictions_disease.csv')

        report_h_path = out_dir / 'classification_report_health.txt'
        report_h = write_binary_report(y_true_h, y_pred_h, report_h_path)
        print(report_h)

        export_val_predictions_csv_binary(records_h, out_csv=out_dir / 'val_predictions_health.csv')

        if args.save_hard_examples:
            save_hard_examples(out_dir, records_d, class_names, k=args.hard_k)
            save_hard_examples_binary(out_dir, records_h, k=args.hard_k)
            print(f"Saved hard examples under {out_dir / 'hard_examples'} and {out_dir / 'hard_examples_binary'}")

        if args.save_confusion_pairs:
            save_top_confusion_pairs(out_dir, y_true_d, y_pred_d, class_names, records_d, m=args.pairs_m,
                                     examples_per_pair=args.examples_per_pair)
            print(f"Saved top confusion pairs under {out_dir / 'confusions'}")
    else:
        report_path = out_dir / 'classification_report.txt'
        report = write_classification_report(y_true_d, y_pred_d, class_names, report_path)
        print(report)

        cm_path = save_confusion_matrix_img(y_true_d, y_pred_d, class_names, out_dir)
        print(f"Saved confusion matrix to {cm_path}")
        export_val_predictions_csv(records_d, class_names, out_dir / 'val_predictions.csv')

    print(f"Done. Best val disease acc: {best_val_acc:.4f}. Artifacts in: {out_dir}")


if __name__ == '__main__':
    main()
