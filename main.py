import torch
import numpy as np

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets import UnlearnCifar10, UnlearnCifar100, UnlearnSVNH
from utils import get_model, compute_topk


def train_loop(model, train_loader, criterion, optimizer, device):

    model.train()

    losses = []

    for idx, data in enumerate(tqdm(train_loader)):

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

        image = image.to(device)
        labels = labels.to(device)

        with torch.no_grad():

            output = model(image)
            loss = criterion(output, labels)

            losses.append(loss.item())

            top1 += compute_topk(labels, output, 1)
            top5 += compute_topk(labels, output, 5)

    top1_acc = top1 / len(loader.dataset)
    top5_acc = top5 / len(loader.dataset)

    return np.mean(losses), top1_acc, top5_acc


if __name__ == "__main__":

    SAVE_PATH = "checkpoints/"

    MODEL = "resnet18"
    DSET = "cifar10"

    PAT = 4
    EPOCHS = 100
    LR = 0.001

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    split = [0.7, 0.2, 0.1]
    transform = None

    if DSET == "cifar10":
        dataset = UnlearnCifar10(split=split, transform=transform)
    elif DSET == "cifar100":
        dataset = UnlearnCifar100(split=split, transform=transform)
    elif DSET == "svnh":
        dataset = UnlearnSVNH(split=split, transform=transform)

    classes = dataset.classes

    train_set = Subset(dataset, dataset.TRAIN)
    val_set = Subset(dataset, dataset.VAL)
    test_set = Subset(dataset, dataset.TEST)
    forget_set = Subset(dataset, dataset.FORGET)
    retain_set = Subset(dataset, dataset.RETAIN)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    forget_loader = DataLoader(forget_set, batch_size=32, shuffle=False)
    retain_loader = DataLoader(retain_set, batch_size=32, shuffle=False)

    model, config, transform = get_model(MODEL, len(classes), True)
    model = model.to(DEVICE)

    dataset.set_transform(transform)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("Training model")

    best_acc = 0
    pat = PAT
    for epoch in range(EPOCHS):

        loss = train_loop(model, train_loader, criterion, optimizer, DEVICE)

        val_loss, val_top1, val_top5 = test_loop(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch: {epoch}, Loss: {round(loss,2)}, Val Loss: {round(val_loss,2)}, Val Top1: {round(val_top1,2)} Val Top5: {round(val_top5,2)} PAT: {pat}"
        )

        if val_top1 > best_acc:
            best_acc = val_top1
            pat = PAT

            model_savefile = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            }

            model_savepath = f"{SAVE_PATH}{MODEL}_{DSET}_best.pt"

            torch.save(model_savefile, model_savepath)

        else:
            pat -= 1
            if pat == 0:
                break

    test_loss, test_top1, test_top5 = test_loop(model, test_loader, criterion, DEVICE)

    print(
        f"Test Loss: {round(test_loss,2)}, Test Top1: {round(test_top1,2)} Test Top5: {round(test_top5,2)}"
    )
