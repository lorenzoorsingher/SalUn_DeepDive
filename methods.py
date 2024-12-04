import torch
from tqdm import tqdm


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


def compute_mask(model, forget_loader, unlearn_lr, saliency_threshold=0.5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    optimizer = torch.optim.SGD(model.parameters(), lr=unlearn_lr)
    criterion = torch.nn.CrossEntropyLoss()

    gradients = {}
    for name, param in model.named_parameters():
        gradients[name] = 0

    print("Computing Gradients...")
    for batch_idx, batch in enumerate(tqdm(forget_loader, desc="Processing batches")):
        # breakpoint()

        image = batch["image"].to(device)
        target = batch["label"].to(device)

        output = model(image)
        loss = -criterion(output, target)  # Negative loss for unlearning

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    sorted_dict_positions = {}
    hard_dict = {}
    all_elements = -torch.cat([tensor.flatten() for tensor in gradients.values()])

    threshold_index = int(len(all_elements) * saliency_threshold)
    # Calculate positions of all elements
    positions = torch.argsort(all_elements)
    ranks = torch.argsort(positions)

    print("Computing Saliency Mask...")
    start_index = 0
    for key, tensor in tqdm(gradients.items(), desc="Processing tensors"):
        num_elements = tensor.numel()
        # tensor_positions = positions[start_index: start_index + num_elements]
        tensor_ranks = ranks[start_index : start_index + num_elements]

        sorted_positions = tensor_ranks.reshape(tensor.shape)
        sorted_dict_positions[key] = sorted_positions

        # Set the corresponding elements to 1
        threshold_tensor = torch.zeros_like(tensor_ranks)
        threshold_tensor[tensor_ranks < threshold_index] = 1
        threshold_tensor = threshold_tensor.reshape(tensor.shape)
        hard_dict[key] = threshold_tensor
        start_index += num_elements  # More efficient masking

    return hard_dict
