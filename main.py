from datasets import UnlearnCifar10, UnlearnCifar100, UnlearnSVNH
from torch.utils.data import DataLoader, Subset
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
import numpy as np
from tqdm import tqdm


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


def compute_topk(labels, outputs, k):

    _, indeces = outputs.topk(k)
    labels_rep = labels.unsqueeze(1).repeat(1, k)
    topk = (labels_rep == indeces).sum().item()

    return topk


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

    split = [0.7, 0.2, 0.1]
    transform = None
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

    PAT = 4
    EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model("resnet18", num_classes=len(classes), pretrained=True)
    model = model.to(DEVICE)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    breakpoint()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

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

        else:
            pat -= 1
            if pat == 0:
                break

    test_loss, test_top1, test_top5 = test_loop(model, test_loader, criterion, DEVICE)

    print(
        f"Test Loss: {round(test_loss,2)}, Test Top1: {round(test_top1,2)} Test Top5: {round(test_top5,2)}"
    )
