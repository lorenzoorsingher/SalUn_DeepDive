from datasets import UnlearnCifar10, UnlearnCifar100, UnlearnSVNH
from torch.utils.data import DataLoader, Subset


if __name__ == "__main__":

    dataset = UnlearnSVNH()

    train_set = Subset(dataset, dataset.TRAIN)
    val_set = Subset(dataset, dataset.VAL)
    forget_set = Subset(dataset, dataset.FORGET)
    retain_set = Subset(dataset, dataset.RETAIN)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    forget_loader = DataLoader(forget_set, batch_size=32, shuffle=False)
    retain_loader = DataLoader(retain_set, batch_size=32, shuffle=False)

    for data in train_loader:

        image = data["image"]
        label = data["label"]

        breakpoint()
