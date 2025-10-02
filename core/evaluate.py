import numpy as np
import torch


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    y_true_all, y_pred_all = [], []
    records = []
    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1].to(device)
        paths = batch[2]
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        top2_probs, top2_idx = torch.topk(probs, k=2, dim=1)
        for i in range(len(paths)):
            t = int(labels[i].cpu().item())
            p = int(preds[i].cpu().item())
            pr_true = float(probs[i, t].cpu().item())
            pr_pred = float(probs[i, p].cpu().item())
            t2i = int(top2_idx[i, 1].cpu().item()) if top2_idx.size(1) > 1 else p
            t2p = float(top2_probs[i, 1].cpu().item()) if top2_probs.size(1) > 1 else pr_pred
            records.append({
                "path": paths[i],
                "true_idx": t,
                "pred_idx": p,
                "pred_conf": pr_pred,
                "true_conf": pr_true,
                "top2_idx": t2i,
                "top2_conf": t2p,
            })
            y_true_all.append(t)
            y_pred_all.append(p)

    val_loss = running_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return val_loss, acc, np.array(y_true_all), np.array(y_pred_all), records


@torch.no_grad()
def evaluate_multitask(model, loader, disease_criterion, health_criterion, device, health_loss_weight: float = 0.5):
    model.eval()
    loss_sum, total, d_correct, h_correct = 0.0, 0, 0, 0
    y_true_d, y_pred_d, y_true_h, y_pred_h = [], [], [], []
    records_d, records_h = [], []
    for batch in loader:
        imgs = batch[0].to(device)
        labels_d = batch[1].to(device).long()
        labels_h = batch[2].to(device).long()
        paths = batch[3]
        out_d, out_h = model(imgs)
        loss_d = disease_criterion(out_d, labels_d)
        loss_h = health_criterion(out_h, labels_h)
        loss = loss_d + health_loss_weight * loss_h

        probs_d = torch.softmax(out_d, dim=1)
        preds_d = probs_d.argmax(dim=1)
        probs_h = torch.softmax(out_h, dim=1)
        preds_h = probs_h.argmax(dim=1)

        bs = imgs.size(0)
        loss_sum += float(loss.item()) * bs
        total += bs
        d_correct += (preds_d == labels_d).sum().item()
        h_correct += (preds_h == labels_h).sum().item()

        top2_probs_d, top2_idx_d = torch.topk(probs_d, k=2, dim=1)
        for i in range(bs):
            td = int(labels_d[i].cpu().item())
            pd = int(preds_d[i].cpu().item())
            prd_true = float(probs_d[i, td].cpu().item())
            prd_pred = float(probs_d[i, pd].cpu().item())
            t2i = int(top2_idx_d[i, 1].cpu().item()) if top2_idx_d.size(1) > 1 else pd
            t2p = float(top2_probs_d[i, 1].cpu().item()) if top2_probs_d.size(1) > 1 else prd_pred
            records_d.append({
                "path": paths[i],
                "true_idx": td,
                "pred_idx": pd,
                "pred_conf": prd_pred,
                "true_conf": prd_true,
                "top2_idx": t2i,
                "top2_conf": t2p,
            })
            y_true_d.append(td); y_pred_d.append(pd)

            th = int(labels_h[i].cpu().item())
            ph = int(preds_h[i].cpu().item())
            prh_true = float(probs_h[i, th].cpu().item())
            prh_pred = float(probs_h[i, ph].cpu().item())
            records_h.append({
                "path": paths[i],
                "true_idx": th,
                "pred_idx": ph,
                "pred_conf": prh_pred,
                "true_conf": prh_true,
            })
            y_true_h.append(th); y_pred_h.append(ph)

    loss_avg = loss_sum / max(1, total)
    return (loss_avg,
            d_correct / max(1, total), np.array(y_true_d), np.array(y_pred_d), records_d,
            h_correct / max(1, total), np.array(y_true_h), np.array(y_pred_h), records_h)
