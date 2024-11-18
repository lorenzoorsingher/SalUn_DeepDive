import datetime
import os
import torch
import numpy as np
import wandb

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from dotenv import load_dotenv

from datasets import UnlearnCifar10, UnlearnCifar100, UnlearnSVNH
from utils import get_model, compute_topk, load_checkpoint


def eval_unlearning(model, test_loader, forget_loader, DEVICE):

    model.eval()

    retain_acc = 0
    forget_acc = 0

    for loader, desc in zip([test_loader, forget_loader], ["Retain", "Forget"]):

        for data in tqdm(loader):

            image = data["image"]
            target = data["label"]
            image = image.to(DEVICE)
            target = target.to(DEVICE)

            output = model(image)

            acc = compute_topk(target, output, 1)

            if desc == "Retain":
                retain_acc += acc
            else:
                forget_acc += acc

    retain_acc /= len(test_loader.dataset)
    forget_acc /= len(forget_loader.dataset)

    return retain_acc, forget_acc

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
        image, target = batch[0].to(device), batch[1].to(device)
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
    all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

    threshold_index = int(len(all_elements) * saliency_threshold)
    # Calculate positions of all elements
    positions = torch.argsort(all_elements)
    ranks = torch.argsort(positions)

    print("Computing Saliency Mask...")
    start_index = 0
    for key, tensor in tqdm(gradients.items(),desc="Processing tensors"):
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



if __name__ == "__main__":

    SAVE_PATH = "checkpoints/"
    LOG = False
    CLASS_TO_FORGET = 0
    LOAD = "checkpoints/resnet18_cifar10_best.pt"
    LOAD_MASK = False
    MASK_PATH = f"checkpoints/mask_resnet18_cifar10_{CLASS_TO_FORGET}.pt"
    USE_MASK = True
    
    model, config, transform, opt = load_checkpoint(LOAD)

    DSET = config["dataset"]
    MODEL = config["model"]

    PAT = 4
    EPOCHS = 10
    LR = 0.001
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MASK_THRESHOLD = 0.5
    if LOG:
        load_dotenv()
        WANDB_SECRET = os.getenv("WANDB_SECRET")
        wandb.login(key=WANDB_SECRET)

    split = [0.7, 0.2, 0.1]

    if DSET == "cifar10":
        dataset = UnlearnCifar10(split=split, transform=transform, class_to_forget=CLASS_TO_FORGET)
    elif DSET == "cifar100":
        dataset = UnlearnCifar100(split=split, transform=transform)
    elif DSET == "svnh":
        dataset = UnlearnSVNH(split=split, transform=transform)

    classes = dataset.classes

    train_set = Subset(dataset, dataset.TRAIN)
    # val_set = Subset(dataset, dataset.VAL)
    test_set = Subset(dataset, dataset.TEST)
    forget_set = Subset(dataset, dataset.FORGET)
    retain_set = Subset(dataset, dataset.RETAIN)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    forget_loader = DataLoader(forget_set, batch_size=32, shuffle=False)
    retain_loader = DataLoader(retain_set, batch_size=32, shuffle=False)

    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    model_savefile = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            }
    if(LOAD_MASK == False):
        mask = compute_mask(model, forget_loader, unlearn_lr=LR, saliency_threshold=MASK_THRESHOLD, device=DEVICE)
        torch.save(mask, MASK_PATH)
    else:
        mask = torch.load(MASK_PATH)
    

    if LOG:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
        wandb.init(
            project="TrendsAndApps",
            name=run_name,
            config={
                "model": MODEL,
                "dataset": DSET,
            },
        )

    ret_acc, for_acc = eval_unlearning(model, test_loader, forget_loader, DEVICE)
    print(f"Retain: {round(ret_acc,2)}, Forget: {round(for_acc,2)}")

    print("[MAIN] Unlearning model")

    forget_tensor = torch.tensor(dataset.FORGET).to(DEVICE)

    for epoch in range(EPOCHS):

        print(f"Epoch {epoch}")

        model.train()

        for data in tqdm(train_loader):

            image = data["image"]
            target = data["label"]
            idx = data["idx"]

            image = image.to(DEVICE)
            target = target.to(DEVICE)
            idx = idx.to(DEVICE)

            # Assign random labels to forget data
            which_is_in = idx.unsqueeze(1) == forget_tensor
            which_is_in = which_is_in.any(dim=1)
            rand_targets = torch.randint(0, len(classes), target.shape).to(DEVICE)
            target[which_is_in] = rand_targets[which_is_in]

            output = model(image)
            loss = criterion(output, target)

            loss.backward()

            if USE_MASK:
                for name, param in model.named_parameters():
                    if name in mask:
                        param.grad *= mask[name]
            optimizer.step()
            optimizer.zero_grad()
            model_savepath = f"{SAVE_PATH}{MODEL}_{DSET}_best_RL.pt"
            model_savefile = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                }   
            torch.save(model_savefile, model_savepath)

        ret_acc, for_acc = eval_unlearning(model, test_loader, forget_loader, DEVICE)
        print(f"Retain: {round(ret_acc,2)}, Forget: {round(for_acc,2)}")
