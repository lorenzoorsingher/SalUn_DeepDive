import torch


def rand_label(model, image, target, idx, criterion, loader, device):
    # Assign random labels to forget data
    ds = loader.dataset.dataset
    forget_tensor = torch.tensor(ds.FORGET).to(device)
    which_is_in = (idx.unsqueeze(1) == forget_tensor).any(dim=1)
    # rand_targets = torch.randint(1, len(ds.classes), target.shape).to(device)
    rand_targets = torch.randint(0, len(ds.classes), target.shape).to(device)
    # rand_targets = (target + rand_targets) % len(ds.classes)
    target[which_is_in] = rand_targets[which_is_in]

    output = model(image)
    loss = criterion(output, target)
    loss = loss.mean()
    return loss


def grad_ascent(model, image, target, idx, criterion, loader, device):
    output = model(image)
    loss = criterion(output, target)

    ds = loader.dataset.dataset
    forget_tensor = torch.tensor(ds.FORGET).to(device)
    which_is_in = (idx.unsqueeze(1) == forget_tensor).any(dim=1)
    loss[which_is_in] *= -1
    loss = loss.mean()

    return loss


def grad_ascent_small(model, image, target, idx, criterion, loader, device):
    output = model(image)
    loss = -criterion(output, target)
    loss = loss.mean()

    return loss


def retrain(model, image, target, idx, criterion, loader, device):
    output = model(image)
    loss = criterion(output, target)
    loss = loss.mean()

    return loss
