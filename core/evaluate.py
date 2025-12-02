import numpy as np
import torch


@torch.no_grad()
def evaluate_mtl(model,
                 loader,
                 species_criterion,
                 health_criterion,
                 disease_criterion,
                 device,
                 w_species: float = 1.0,
                 w_health: float = 1.0,
                 w_disease: float = 0.5):  # classifying disease is harder, weight = 0.5 to reduce influence
    model.eval()

    loss_sum, total = 0.0, 0

    # Accumulators
    sp_correct = 0
    hl_correct = 0
    dz_correct_overall = 0
    dz_correct_sick = 0
    sick_count = 0

    y_true_sp, y_pred_sp = [], []
    y_true_hl, y_pred_hl = [], []
    y_true_dz, y_pred_dz = [], []

    records_sp, records_hl, records_dz = [], [], []

    for batch in loader:
        imgs, target, paths, *_ = batch
        imgs = imgs.to(device)

        sp = target["species"].to(device).long()
        hl = target["health"].to(device).long()      # 1=Healthy, 0=Sick
        dz = target["disease"].to(device).long()

        # Forward
        sp_logits, hl_logits, dz_logits = model(imgs)

        # Loss (sum, we'll average later)
        loss_sp = species_criterion(sp_logits, sp)
        loss_hl = health_criterion(hl_logits, hl)
        loss_dz = disease_criterion(dz_logits, dz)
        loss = w_species * loss_sp + w_health * loss_hl + w_disease * loss_dz

        bs = imgs.size(0)
        loss_sum += float(loss.item()) * bs
        total += bs

        # Predictions / probs
        sp_probs = torch.softmax(sp_logits, dim=1)
        hl_probs = torch.softmax(hl_logits, dim=1)
        dz_probs = torch.softmax(dz_logits, dim=1)

        sp_pred = sp_probs.argmax(1)
        hl_pred = hl_probs.argmax(1)
        dz_pred = dz_probs.argmax(1)

        # Accuracies
        sp_correct += (sp_pred == sp).sum().item()
        hl_correct += (hl_pred == hl).sum().item()
        dz_correct_overall += (dz_pred == dz).sum().item()

        sick_mask = (hl == 0)
        if sick_mask.any():
            dz_correct_sick += (dz_pred[sick_mask] == dz[sick_mask]).sum().item()
            sick_count += int(sick_mask.sum().item())

        # Store y_true / y_pred
        y_true_sp.extend([int(v) for v in sp.cpu().tolist()])
        y_pred_sp.extend([int(v) for v in sp_pred.cpu().tolist()])

        y_true_hl.extend([int(v) for v in hl.cpu().tolist()])
        y_pred_hl.extend([int(v) for v in hl_pred.cpu().tolist()])

        y_true_dz.extend([int(v) for v in dz.cpu().tolist()])
        y_pred_dz.extend([int(v) for v in dz_pred.cpu().tolist()])

        # Records (top-2 for species/disease; health is 2-class so top-2 is trivial)
        topk_sp = min(2, sp_probs.size(1))
        topk_dz = min(2, dz_probs.size(1))
        top2_probs_sp, top2_idx_sp = torch.topk(sp_probs, k=topk_sp, dim=1)
        top2_probs_dz, top2_idx_dz = torch.topk(dz_probs, k=topk_dz, dim=1)

        for i in range(bs):
            # Species record
            t_sp = int(sp[i].item())
            p_sp = int(sp_pred[i].item())
            pr_sp_true = float(sp_probs[i, t_sp].item())
            pr_sp_pred = float(sp_probs[i, p_sp].item())
            if topk_sp > 1:
                t2i_sp = int(top2_idx_sp[i, 1].item())
                t2p_sp = float(top2_probs_sp[i, 1].item())
            else:
                t2i_sp, t2p_sp = p_sp, pr_sp_pred
            records_sp.append({
                "path": paths[i],
                "true_idx": t_sp,
                "pred_idx": p_sp,
                "pred_conf": pr_sp_pred,
                "true_conf": pr_sp_true,
                "top2_idx": t2i_sp,
                "top2_conf": t2p_sp,
            })

            # Health record (binary; we still log top-2 for consistency)
            t_hl = int(hl[i].item())
            p_hl = int(hl_pred[i].item())
            pr_hl_true = float(hl_probs[i, t_hl].item())
            pr_hl_pred = float(hl_probs[i, p_hl].item())
            # For binary, top-2 is just the other class
            t2i_hl = 1 - p_hl
            t2p_hl = float(hl_probs[i, t2i_hl].item())
            records_hl.append({
                "path": paths[i],
                "true_idx": t_hl,
                "pred_idx": p_hl,
                "pred_conf": pr_hl_pred,
                "true_conf": pr_hl_true,
                "top2_idx": t2i_hl,
                "top2_conf": t2p_hl,
            })

            # Disease record
            t_dz = int(dz[i].item())
            p_dz = int(dz_pred[i].item())
            pr_dz_true = float(dz_probs[i, t_dz].item())
            pr_dz_pred = float(dz_probs[i, p_dz].item())
            if topk_dz > 1:
                t2i_dz = int(top2_idx_dz[i, 1].item())
                t2p_dz = float(top2_probs_dz[i, 1].item())
            else:
                t2i_dz, t2p_dz = p_dz, pr_dz_pred
            records_dz.append({
                "path": paths[i],
                "true_idx": t_dz,
                "pred_idx": p_dz,
                "pred_conf": pr_dz_pred,
                "true_conf": pr_dz_true,
                "top2_idx": t2i_dz,
                "top2_conf": t2p_dz,
            })

    metrics = {
        "loss": loss_sum / max(1, total),
        "species": {
            "acc": sp_correct / max(1, total),
            "y_true": np.array(y_true_sp, dtype=np.int64),
            "y_pred": np.array(y_pred_sp, dtype=np.int64),
            "records": records_sp,
        },
        "health": {
            "acc": hl_correct / max(1, total),
            "y_true": np.array(y_true_hl, dtype=np.int64),
            "y_pred": np.array(y_pred_hl, dtype=np.int64),
            "records": records_hl,
        },
        "disease": {
            "acc_overall": dz_correct_overall / max(1, total),
            "acc_sick_only": (dz_correct_sick / sick_count) if sick_count > 0 else 0.0,
            "y_true": np.array(y_true_dz, dtype=np.int64),
            "y_pred": np.array(y_pred_dz, dtype=np.int64),
            "records": records_dz,
        },
    }
    return metrics
