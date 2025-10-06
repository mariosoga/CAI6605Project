import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# -----------------------------
# single-task
# -----------------------------
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels, *_ in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        running_loss += float(loss.item()) * bs
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs

    return running_loss / total, correct / total

# -----------------------------
# Multi-task (3 heads): species, health, disease
# health target convention: 1 = Healthy, 0 = Sick
# -----------------------------
def train_mtl(model,
              loader,
              optimizer,
              device,
              w_species: float = 1.0,
              w_health:  float = 1.0,
              w_disease: float = 0.5,
              use_amp: bool = False):
    model.train()
    scaler = GradScaler(enabled=use_amp)

    loss_sum = 0.0
    total = 0
    sp_correct = 0
    hl_correct = 0
    dz_correct_overall = 0
    dz_correct_sick = 0
    sick_count = 0

    for imgs, target, *_ in loader:
        imgs = imgs.to(device, non_blocking=True)
        sp = target["species"].to(device, non_blocking=True).long()
        hl = target["health"].to(device, non_blocking=True).long()     # 1=Healthy, 0=Sick
        dz = target["disease"].to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            sp_logits, hl_logits, dz_logits = model(imgs)
            loss_sp = F.cross_entropy(sp_logits, sp)
            loss_hl = F.cross_entropy(hl_logits,  hl)
            loss_dz = F.cross_entropy(dz_logits, dz)
            loss = w_species*loss_sp + w_health*loss_hl + w_disease*loss_dz

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = imgs.size(0)
        loss_sum += float(loss.item()) * bs
        total += bs

        # metrics
        sp_pred = sp_logits.argmax(1)
        hl_pred = hl_logits.argmax(1)
        dz_pred = dz_logits.argmax(1)

        sp_correct += (sp_pred == sp).sum().item()
        hl_correct += (hl_pred == hl).sum().item()
        dz_correct_overall += (dz_pred == dz).sum().item()

        sick_mask = (hl == 0)
        if sick_mask.any():
            dz_correct_sick += (dz_pred[sick_mask] == dz[sick_mask]).sum().item()
            sick_count += int(sick_mask.sum().item())

    metrics = {
        "loss": loss_sum / max(1, total),
        "species_acc": sp_correct / max(1, total),
        "health_acc":  hl_correct / max(1, total),
        "disease_acc_overall": dz_correct_overall / max(1, total),
        "disease_acc_sick_only": (dz_correct_sick / sick_count) if sick_count > 0 else 0.0,
    }
    return metrics
