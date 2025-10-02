def train_one_epoch_single(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()  # clears previous gradients before computing new ones

        # compute predictions and loss
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # aggregate batch losses
        batch_size = imgs.size(0)
        batch_loss = loss.item() * batch_size
        running_loss += batch_loss

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    return train_loss, train_acc


def train_one_epoch_multi(model, loader, d_criterion, h_criterion, optimizer, device, h_loss_weight: float):
    model.train()
    loss_sum, d_correct, h_correct, total = 0.0, 0, 0, 0
    for batch in loader:
        imgs = batch[0].to(device)
        labels_d = batch[1].to(device).long()
        labels_h = batch[2].to(device).long()
        optimizer.zero_grad()

        # compute predictions and loss
        out_d, out_h = model(imgs)
        loss_d = d_criterion(out_d, labels_d)
        loss_h = h_criterion(out_h, labels_h)

        loss = loss_d + h_loss_weight * loss_h

        # Backpropagation
        loss.backward()
        optimizer.step()

        # aggregate batch losses
        batch_size = imgs.size(0)
        batch_loss = float(loss.item()) * imgs.size(0)
        loss_sum += batch_loss

        d_correct += (out_d.argmax(dim=1) == labels_d).sum().item()
        h_correct += (out_h.argmax(dim=1) == labels_h).sum().item()
        total += batch_size

    train_loss = loss_sum / total
    train_d_acc = d_correct / total
    train_h_acc = h_correct / total

    return train_loss, train_d_acc, train_h_acc
