from datasets import UnlearnCifar10, UnlearnCifar100, UnlearnSVNH
from torch.utils.data import DataLoader, Subset
import timm
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


def test_loop(model, loader, criterion, device):

    model.eval()

    losses = []

    true_positives = 0

    for idx, data in enumerate(tqdm(loader)):

        image = data["image"]
        label = data["label"]

        image = image.to(device)
        label = label.to(device)

        with torch.no_grad():

            output = model(image)
            loss = criterion(output, label)

            losses.append(loss.item())
            preds = output.argmax(dim=1)
            true_positives += (preds == label).sum().item()

        #     torch.argmax()
        # breakpoint()

    accuracy = true_positives / len(loader.dataset)

    return np.mean(losses), accuracy


if __name__ == "__main__":

    split = [0.7, 0.2, 0.1]
    transform = None
    dataset = UnlearnCifar100(split=split, transform=transform)
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

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    print("Training model")

    best_acc = 0
    pat = PAT
    for epoch in range(EPOCHS):

        loss = train_loop(model, train_loader, criterion, optimizer, DEVICE)

        val_loss, val_acc = test_loop(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch: {epoch}, Loss: {round(loss,2)}, Val Loss: {round(val_loss,2)}, Val Acc: {round(val_acc,2)} PAT: {pat}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            pat = PAT
        else:
            pat -= 1

    test_loss, test_acc = test_loop(model, test_loader, criterion, DEVICE)

    print(f"Test Loss: {round(test_loss,2)}, Test Acc: {round(test_acc,2)}")
