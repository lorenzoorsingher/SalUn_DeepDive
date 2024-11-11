import copy
import random
import torch
from functools import partial
from tqdm import tqdm

from method import utils
from method.metrics import evaluate_after_unlearning


def compute_mask(model, forget_loader, criterion, args):
    gradients = {}
    model.eval()
    model.zero_grad()

    for name, param in model.named_parameters():
        gradients[name] = 0

    print("Computing Gradients...")
    for batch in forget_loader:
        image = batch[-3]
        target = batch[-2]

        image = image.to(args.device)
        target = target.to(args.device)

        output = model(image)
        loss = -criterion(output, target)

        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

        model.zero_grad()

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    mask = {}
    all_elements = -torch.cat([w.flatten() for w in gradients.values()])

    # calculate number of elements to keep
    threshold_index = int(len(all_elements) * args.saliency_threshold)

    # calculate positions of all elements
    positions = torch.argsort(all_elements)
    ranks = torch.argsort(positions)

    print("Computing Saliency Mask...")
    start_index = 0
    for key, w in gradients.items():
        num_elements = w.numel()
        weight_ranks = ranks[start_index : start_index + num_elements]

        # set the corresponding elements to 1
        threshold_tensor = torch.zeros_like(weight_ranks)
        threshold_tensor[weight_ranks < threshold_index] = 1
        threshold_tensor = threshold_tensor.reshape(w.shape)
        mask[key] = threshold_tensor
        start_index += num_elements

    return mask


def random_labeling_big(model, datasets, use_mask, run, args):
    assert args.world_size == 1, "SalUn is not compatible with distributed training"
    assert args.task == "classification", "SalUn is not compatible with multilabel classification"

    train_data = datasets.get_train_data(args.use_train_aug)
    forget_dataset = datasets.get_unlearning_data(train=args.use_train_aug)["forget"]
    forget_indices = set(forget_dataset.indices)
    forget_targets = {i: random.randint(0, args.num_classes - 1) for i in forget_dataset.indices}

    generic_loader = partial(
        torch.utils.data.DataLoader,
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
    )

    train_sampler = utils.get_sampler(train_data, shuffle=True)
    forget_sampler = utils.get_sampler(forget_dataset, shuffle=True, weights=None)

    train_loader = generic_loader(train_data, sampler=train_sampler)
    forget_loader = generic_loader(forget_dataset, sampler=forget_sampler)

    utils.print_info(args, model, train_loader)

    criterion = utils.get_criterion(args.criterion).to(args.device)

    if use_mask:
        mask = compute_mask(model, forget_loader, criterion, args)

    unlearned_model = copy.deepcopy(model)
    unlearned_model = unlearned_model.to(args.device)

    criterion = utils.get_criterion(args.criterion).to(args.device)

    optimizer = utils.get_optimizer(unlearned_model, args)
    scheduler = utils.get_scheduler(optimizer, args)

    num_digits = len(str(args.epochs))
    for epoch in range(args.epochs):
        unlearned_model.train()

        # enable returning indices of training samples
        train_loader.dataset.dataset.return_indexes = True

        for indices, image, target, sensitive in tqdm(
            train_loader, leave=False, dynamic_ncols=True
        ):
            with torch.autocast(device_type="cuda", dtype=args.dtype):
                image = image.to(device=args.device, dtype=args.dtype)
                target = target.to(args.device)
                sensitive = sensitive.to(args.device)

                # use random labeling for forget data
                forget_samples = [
                    i for i, idx in enumerate(indices) if idx.item() in forget_indices
                ]
                if len(forget_samples) > 0:
                    forget_samples = torch.tensor(forget_samples)
                    random_targets = torch.tensor(
                        [
                            forget_targets[idx.item()]
                            for idx in indices
                            if idx.item() in forget_indices
                        ],
                        device=args.device,
                    )
                    target[forget_samples] = random_targets

                output = unlearned_model(image)
                loss = criterion(output, target)

            loss.backward()

            if use_mask:
                for name, param in unlearned_model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()
            optimizer.zero_grad()

            if args.debug:
                break

        if epoch % args.evaluate_every == 0:
            (
                retain_loss,
                retain_acc,
                retain_losses,
                forget_loss,
                forget_acc,
                forget_losses,
                val_loss,
                val_acc,
                val_losses,
                test_loss,
                test_acc,
                test_losses,
            ) = evaluate_after_unlearning(unlearned_model, datasets, criterion, args=args)
            print(
                f"| Epoch: {str(epoch+1).zfill(num_digits)}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.4f} | Retain Loss: {retain_loss:.4f} | Retain Acc: {retain_acc:.2f} | Forget Loss: {forget_loss:.4f} | Forget Acc: {forget_acc:.2f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f} |"
            )

        scheduler.step()

        if args.debug:
            break

    return unlearned_model


def random_labeling_small(model, datasets, use_mask, run, args):
    assert args.world_size == 1, "SalUn is not compatible with distributed training"
    assert args.task == "classification", "SalUn is not compatible with multilabel classification"

    train_dataset = datasets.get_train_data()
    unlearning_datasets = datasets.get_unlearning_data(train=args.use_train_aug)
    retain_dataset = unlearning_datasets["retain"]
    forget_dataset = unlearning_datasets["forget"]

    generic_loader = partial(
        torch.utils.data.DataLoader,
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
    )

    train_sampler = utils.get_sampler(train_dataset, shuffle=False, weights=None)
    retain_sampler = utils.get_sampler(retain_dataset, shuffle=True, weights=None)
    forget_sampler = utils.get_sampler(forget_dataset, shuffle=True, weights=None)

    train_loader = generic_loader(train_dataset, sampler=train_sampler)
    retain_loader = generic_loader(retain_dataset, sampler=retain_sampler)
    forget_loader = generic_loader(forget_dataset, sampler=forget_sampler)

    utils.print_info(args, model, train_loader)

    criterion = utils.get_criterion(args.criterion).to(args.device)

    if use_mask:
        mask = compute_mask(model, forget_loader, criterion, args)

    unlearned_model = copy.deepcopy(model)
    unlearned_model = unlearned_model.to(args.device)

    criterion = utils.get_criterion(args.criterion).to(args.device)

    optimizer = utils.get_optimizer(unlearned_model, args)
    scheduler = utils.get_scheduler(optimizer, args)

    num_digits = len(str(args.epochs))
    for epoch in range(args.epochs):
        unlearned_model.train()
        for desc, loader in zip(["Forget", "Retain"], [forget_loader, retain_loader]):
            for image, target, sensitive in tqdm(
                loader, desc=f"{desc} Step", leave=False, dynamic_ncols=True
            ):
                with torch.autocast(device_type="cuda", dtype=args.dtype):
                    image = image.to(device=args.device, dtype=args.dtype)
                    if desc == "Forget":
                        target = torch.randint(0, args.num_classes, target.size())
                    target = target.to(args.device)
                    sensitive = sensitive.to(args.device)

                    output = unlearned_model(image)
                    loss = criterion(output, target)

                loss.backward()

                if use_mask:
                    for name, param in unlearned_model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]

                optimizer.step()
                optimizer.zero_grad()

                if args.debug:
                    break

        if args.debug:
            break

        if epoch % args.evaluate_every == 0:
            (
                retain_loss,
                retain_acc,
                retain_losses,
                forget_loss,
                forget_acc,
                forget_losses,
                val_loss,
                val_acc,
                val_losses,
                test_loss,
                test_acc,
                test_losses,
            ) = evaluate_after_unlearning(unlearned_model, datasets, criterion, args=args)
            print(
                f"| Epoch: {str(epoch+1).zfill(num_digits)}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.4f} | Retain Loss: {retain_loss:.4f} | Retain Acc: {retain_acc:.2f} | Forget Loss: {forget_loss:.4f} | Forget Acc: {forget_acc:.2f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f} |"
            )

        scheduler.step()

    return unlearned_model


def rl(model, datasets, run, args):
    random_labeling = random_labeling_small if args.dataset != "cifar100" else random_labeling_big
    return random_labeling(model, datasets, use_mask=False, run=run, args=args)


def salun(model, datasets, run, args):
    random_labeling = random_labeling_small if args.dataset != "cifar100" else random_labeling_big
    return random_labeling(model, datasets, use_mask=True, run=run, args=args)
