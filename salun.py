import copy
import random
import torch
from functools import partial
from tqdm import tqdm
import time
from torchvision import transforms



import torch

def compute_mask(model, forget_loader, unlearn_lr, saliency_threshold=0.5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=unlearn_lr)
    criterion = torch.nn.CrossEntropyLoss()

    gradients = {}
    for name, param in model.named_parameters():
        gradients[name] = torch.zeros_like(param.data)

    print("Computing Gradients...")
    for batch_idx, batch in enumerate(forget_loader):
        image, target = batch[0].to(device), batch[1].to(device)
        output = model(image)
        loss = -criterion(output, target)  # Negative loss for unlearning

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data / len(forget_loader)  # Average gradients

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    mask = {}
    all_elements = torch.cat([w.flatten() for w in gradients.values()])

    # Dynamic threshold calculation (example - you can adjust this)
    threshold_index = int(len(all_elements) * saliency_threshold) 
    threshold_value = torch.kthvalue(all_elements, threshold_index).values

    print("Computing Saliency Mask...")
    for key, w in gradients.items():
        mask[key] = (w >= threshold_value).float()  # More efficient masking

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


def random_labeling(model, dataset , mask, use_mask = True, epochs =10):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{0}")
    else:
        device = torch.device("cpu")
    
    criterion = torch.nn.CrossEntropyLoss()
    unlearned_model = copy.deepcopy(model)
    unlearned_model = unlearned_model.to(device)
    model.train()
    #assign random labels to the forget data
    data_copy = copy.deepcopy(dataset.data)
    forget_idx = dataset.FORGET
    forget_data_random_label = []
    #assign random label except the original one
    for idx in forget_idx:
        class_to_forget = dataset[idx]["label"]
        num = random.randint(0,len(dataset.classes)-1)
        if(num == class_to_forget):
            num = (num+1)%(10)
        image = data_copy[idx][0]
        image_tensor = transforms.ToTensor()(image)
        new_data = (image_tensor, num)
        forget_data_random_label.append(new_data)
    #create a dataloader for the random labeled data
    random_labeled_loader = torch.utils.data.DataLoader(forget_data_random_label, batch_size=32, shuffle=True)
    optimizer = torch.optim.AdamW(unlearned_model.parameters(), lr=0.1)
    for epoch in range(epochs):
        for batch in tqdm(random_labeled_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            image = batch[0]
            target = batch[1]
            image = image.to(device)
            target = target.to(device)

            output = unlearned_model(image)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            if use_mask:
                for name, param in unlearned_model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            optimizer.step()
        
    return unlearned_model

def salun(model, datasets, run, args):
    random_labeling = random_labeling_small if args.dataset != "cifar100" else random_labeling_big
    return random_labeling(model, datasets, use_mask=True, run=run, args=args)

