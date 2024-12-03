import datetime
import json
import os
import torch
import numpy as np
import wandb
import argparse

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from dotenv import load_dotenv

from datasets import UnlearnCifar10, UnlearnCifar100, UnlearnSVNH
from utils import get_model, compute_topk, load_checkpoint


def train_loop(model, train_loader, criterion, optimizer, device):

    model.train()

    losses = []

    for _, data in enumerate(tqdm(train_loader)):

        image = data["image"]
        label = data["label"]

        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        output = model(image)

        loss = criterion(output, label)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    return np.mean(losses)


def test_loop(model, loader, criterion, device):

    model.eval()

    losses = []
    top1 = 0
    top5 = 0
    for idx, data in enumerate(tqdm(loader)):

        image = data["image"]
        labels = data["label"]

        image = image.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():

            output = model(image)
            loss = criterion(output, labels)

            losses.append(loss.item())

            top1 += compute_topk(labels, output, 1)
            top5 += compute_topk(labels, output, 5)

    top1_acc = top1 / len(loader.dataset)
    top5_acc = top5 / len(loader.dataset)

    return np.mean(losses), top1_acc, top5_acc


def get_args():

    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--comment", type=str, default="pretrained")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()

    SAVE_PATH = "checkpoints/"
    LOG = False

    # resnet18 vit_tiny_patch16_224 ... use timm.list_models() to get all models
    MODEL = args.model

    # cifar10 cifar100 svnh ...
    DSET = args.dataset

    comment = args.comment

    PAT = 10
    EPOCHS = 200
    LR = 0.001

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if LOG:
        load_dotenv()
        WANDB_SECRET = os.getenv("WANDB_SECRET")
        wandb.login(key=WANDB_SECRET)

    split = [0.7, 0.2, 0.1]
    transform = None

    if DSET == "cifar10":
        dataset = UnlearnCifar10(split=split, transform=transform, unlearning_ratio=0.1)
    elif DSET == "cifar100":
        dataset = UnlearnCifar100(
            split=split, transform=transform, unlearning_ratio=0.1
        )
    elif DSET == "svnh":
        dataset = UnlearnSVNH(split=split, transform=transform, unlearning_ratio=0.1)

    classes = dataset.classes

    train_set = Subset(dataset, dataset.TRAIN)
    val_set = Subset(dataset, dataset.VAL)
    test_set = Subset(dataset, dataset.TEST)
    forget_set = Subset(dataset, dataset.FORGET)
    retain_set = Subset(dataset, dataset.RETAIN)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=8)
    forget_loader = DataLoader(forget_set, batch_size=32, shuffle=False)
    retain_loader = DataLoader(retain_set, batch_size=32, shuffle=False)
    # breakpoint()

    forget_targets = f"{SAVE_PATH}{MODEL}_{DSET}_{comment}_forget.json"
    with open(forget_targets, "w") as f:
        json.dump(dataset.FORGET, f)

    model, config, transform = get_model(MODEL, len(classes), True)

    config = {
        "model": MODEL,
        "dataset": DSET,
        "nclasses": len(classes),
    }

    model = model.to(DEVICE)
    dataset.set_transform(transform)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.2, patience=5, verbose=True
    )

    print("Training model")

    if LOG:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
        wandb.init(
            project="TrendsAndApps",
            name=run_name,
            config=config,
        )

    best_acc = 0
    pat = PAT
    for epoch in range(EPOCHS):

        loss = train_loop(model, retain_loader, criterion, optimizer, DEVICE)

        val_loss, val_top1, val_top5 = test_loop(model, val_loader, criterion, DEVICE)
        for_loss, for_top1, for_top5 = test_loop(
            model, forget_loader, criterion, DEVICE
        )

        scheduler.step(val_top1)

        print(f"lr: {optimizer.param_groups[0]['lr']}")

        if LOG:
            wandb.log(
                {
                    "train_loss": loss,
                    "val_loss": val_loss,
                    "val_top1": val_top1,
                    "val_top5": val_top5,
                }
            )
        print(
            f"Epoch: {epoch}, Loss: {round(loss,2)}, Val Loss: {round(val_loss,2)}, Val Top1: {round(val_top1,2)} Val Top5: {round(val_top5,2)} PAT: {pat}"
        )
        print(
            f"Epoch: {epoch}, Loss: {round(loss,2)}, For Loss: {round(for_loss,2)}, For Top1: {round(for_top1,2)} For Top5: {round(for_top5,2)} PAT: {pat}"
        )

        if val_top1 > best_acc:
            best_acc = val_top1
            pat = PAT

            model_savefile = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            }

            model_savepath = f"{SAVE_PATH}{MODEL}_{DSET}_{comment}_forget.pt"

            torch.save(model_savefile, model_savepath)

        else:
            pat -= 1
            if pat == 0:
                break

    test_loss, test_top1, test_top5 = test_loop(model, test_loader, criterion, DEVICE)

    print(
        f"Test Loss: {round(test_loss,2)}, Test Top1: {round(test_top1,2)} Test Top5: {round(test_top5,2)}"
    )

    if LOG:
        wandb.log(
            {
                "test_loss": test_loss,
                "test_top1": test_top1,
                "test_top5": test_top5,
            }
        )
