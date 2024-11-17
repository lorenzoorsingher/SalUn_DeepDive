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


if __name__ == "__main__":

    SAVE_PATH = "checkpoints/"
    LOG = False

    LOAD = "checkpoints/resnet18_cifar10_best.pt"

    model, config, transform, opt = load_checkpoint(LOAD)

    DSET = config["dataset"]
    MODEL = config["model"]

    PAT = 4
    EPOCHS = 100
    LR = 0.001

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if LOG:
        load_dotenv()
        WANDB_SECRET = os.getenv("WANDB_SECRET")
        wandb.login(key=WANDB_SECRET)

    split = [0.7, 0.2, 0.1]

    if DSET == "cifar10":
        dataset = UnlearnCifar10(split=split, transform=transform, class_to_forget=0)
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

            optimizer.step()
            optimizer.zero_grad()

        ret_acc, for_acc = eval_unlearning(model, test_loader, forget_loader, DEVICE)
        print(f"Retain: {round(ret_acc,2)}, Forget: {round(for_acc,2)}")
