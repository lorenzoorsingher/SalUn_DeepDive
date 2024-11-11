import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Subset


class UnlearningDataset(Dataset):
    def __init__(
        self,
        unlearning_ratio=0.1,
        val_ratio=0.2,
        transform=None,
    ):

        self.transform = transform

        self.idxs = None
        self.VAL = None
        self.FORGET = None
        self.RETAIN = None

        self.unlearning_ratio = unlearning_ratio
        self.val_ratio = val_ratio

        self.split_unlearning()

        assert set(self.RETAIN).intersection(set(self.FORGET)) == set()
        assert set(self.RETAIN).union(set(self.FORGET)) == set(self.TRAIN)
        assert set(self.TRAIN).intersection(set(self.VAL)) == set()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx]

        if self.transform:
            sample = self.transform(sample)
        else:
            sample = transforms.ToTensor()(sample)

        # breakpoint()

        data = {}
        data["image"] = sample
        data["label"] = label
        return data

    def split_unlearning(self):

        # Split the training set into train and validation
        self.VAL = random.sample(self.TRAIN, int(len(self.TRAIN) * self.val_ratio))
        self.TRAIN = list(set(self.TRAIN) - set(self.VAL))

        # Split the training set into forget and retain
        num_images_to_forget = int(len(self.TRAIN) * self.unlearning_ratio)
        self.FORGET = random.sample(self.TRAIN, num_images_to_forget)
        self.RETAIN = list(set(self.TRAIN) - set(self.FORGET))

        print(
            f"Train samples: {len(self.TRAIN)} - Forget samples: {len(self.FORGET)} - Split Ratio: {self.unlearning_ratio}"
        )


class UnlearnCifar10(UnlearningDataset):
    def __init__(self, unlearning_ratio=0.1, val_ratio=0.2, transform=None):

        self.data = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=None
        )

        self.TRAIN = [i for i in range(len(self.data))]

        super(UnlearnCifar10, self).__init__(
            unlearning_ratio=unlearning_ratio,
            val_ratio=val_ratio,
            transform=transform,
        )


class UnlearnCifar100(UnlearningDataset):
    def __init__(self, unlearning_ratio=0.1, val_ratio=0.2, transform=None):

        self.data = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=None
        )

        self.TRAIN = [i for i in range(len(self.data))]

        super(UnlearnCifar100, self).__init__(
            unlearning_ratio=unlearning_ratio,
            val_ratio=val_ratio,
            transform=transform,
        )


class UnlearnSVNH(UnlearningDataset):
    def __init__(self, unlearning_ratio=0.1, val_ratio=0.2, transform=None):

        self.data = datasets.SVHN(root="./data", download=True, transform=None)

        self.TRAIN = [i for i in range(len(self.data))]

        super(UnlearnSVNH, self).__init__(
            unlearning_ratio=unlearning_ratio,
            val_ratio=val_ratio,
            transform=transform,
        )


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
