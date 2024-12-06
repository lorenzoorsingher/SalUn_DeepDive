import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import json
import pandas as pd


class UnlearningDataset(Dataset):
    def __init__(
        self,
        split=[0.7, 0.2, 0.1],
        transform=None,
        unlearning_ratio=None,
        class_to_forget=None,
        idx_to_forget=None,
    ):

        self.transform = transform
        self.idxs = None
        self.VAL = None
        self.FORGET = None
        self.RETAIN = None

        self.class_to_forget = class_to_forget
        self.unlearning_ratio = unlearning_ratio
        self.idx_to_forget = idx_to_forget

        self.train_split = split[0]
        self.val_split = split[1]
        self.test_split = split[2]

        self.split_unlearning()

        _, labels = zip(*self.data)
        self.classes = set(labels)

        assert set(self.RETAIN).intersection(set(self.FORGET)) == set()
        assert set(self.RETAIN).union(set(self.FORGET)) == set(self.TRAIN)
        assert set(self.TRAIN).intersection(set(self.VAL)) == set()
        assert set(self.TRAIN).intersection(set(self.TEST)) == set()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx]

        if self.transform:
            sample = transforms.ToTensor()(sample)
            sample = self.transform(sample)
        else:
            sample = transforms.ToTensor()(sample)

        data = {}
        data["image"] = sample
        data["label"] = label

        data["idx"] = idx

        return data

    def split_unlearning(self):

        assert [
            self.idx_to_forget,
            self.unlearning_ratio,
            self.class_to_forget,
        ].count(None) == 2

        # Split the training set
        if self.idx_to_forget is not None:
            self.FORGET = self.idx_to_forget

            print(f"[DATASET] Forgetting {len(self.FORGET)} images from json")

        if self.unlearning_ratio is not None:
            self.FORGET = random.sample(
                self.TRAIN, int(len(self.TRAIN) * self.unlearning_ratio)
            )
            print(f"[DATASET] Forgetting {len(self.FORGET)} random images")

        if self.class_to_forget is not None:
            self.FORGET = [
                idx for idx in self.TRAIN if self.data[idx][1] == self.class_to_forget
            ]
            # Remove the class to forget from the test set
            self.TEST = [
                idx for idx in self.TEST if self.data[idx][1] != self.class_to_forget
            ]
            print(
                f"[DATASET] Forgetting {len(self.FORGET)} images from class {self.class_to_forget}"
            )

        self.RETAIN = list(set(self.TRAIN) - set(self.FORGET))
        self.VAL = random.sample(self.RETAIN, int(len(self.RETAIN) * self.val_split))
        self.RETAIN = list(set(self.RETAIN) - set(self.VAL))
        self.TRAIN = list(set(self.TRAIN) - set(self.VAL))

    def get_samples(self, class_to_retreive=0, n_samples=10):
        samples = []
        concatenated_index = pd.concat(
            [pd.Series(index) for index in [self.VAL, self.TRAIN, self.TEST]]
        ).index
        for idx in concatenated_index:
            if self.data[idx][1] == class_to_retreive:
                samples.append(self.data[idx])
            if len(samples) == n_samples:
                break
        return samples

    def set_transform(self, transform):
        self.transform = transform


class UnlearnCifar10(UnlearningDataset):
    ## Dictionary to map CIFAR-10 labels to class names
    # 0: 'Airplane',
    # 1: 'Automobile',
    # 2: 'Bird',
    # 3: 'Cat',
    # 4: 'Deer',
    # 5: 'Dog',
    # 6: 'Frog',
    # 7: 'Horse',
    # 8: 'Ship',
    # 9: 'Truck'

    def __init__(
        self,
        split=[0.7, 0.2, 0.1],
        transform=None,
        unlearning_ratio=None,
        class_to_forget=None,
        idx_to_forget=None,
    ):

        train = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=None
        )

        test = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=None
        )

        self.TRAIN = [i for i in range(len(train))]

        self.TEST = [i for i in range(len(train), len(train) + len(test))]

        self.data = torch.utils.data.ConcatDataset([train, test])

        super(UnlearnCifar10, self).__init__(
            unlearning_ratio=unlearning_ratio,
            split=split,
            transform=transform,
            class_to_forget=class_to_forget,
            idx_to_forget=idx_to_forget,
        )


class UnlearnCifar100(UnlearningDataset):
    def __init__(
        self,
        split=[0.7, 0.2, 0.1],
        transform=None,
        unlearning_ratio=None,
        class_to_forget=None,
        idx_to_forget=None,
    ):

        train = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=None
        )

        test = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=None
        )

        self.TRAIN = [i for i in range(len(train))]

        self.TEST = [i for i in range(len(train), len(train) + len(test))]

        self.data = torch.utils.data.ConcatDataset([train, test])

        super(UnlearnCifar100, self).__init__(
            unlearning_ratio=unlearning_ratio,
            split=split,
            transform=transform,
            class_to_forget=class_to_forget,
            idx_to_forget=idx_to_forget,
        )


class UnlearnSVNH(UnlearningDataset):
    def __init__(
        self,
        split=[0.7, 0.2, 0.1],
        transform=None,
        unlearning_ratio=None,
        class_to_forget=None,
        idx_to_forget=None,
    ):

        test_split = split[2]
        self.data = datasets.SVHN(root="./data", download=True, transform=None)

        TRAIN = [i for i in range(len(self.data))]

        self.TEST = [i for i in range(int(len(self.data) * test_split))]

        self.TRAIN = list(set(TRAIN) - set(self.TEST))

        super(UnlearnSVNH, self).__init__(
            unlearning_ratio=unlearning_ratio,
            split=split,
            transform=transform,
            class_to_forget=class_to_forget,
            idx_to_forget=idx_to_forget,
        )


def get_dataloaders(
    dataname,
    transform,
    unlr=0,
    itf=None,
    cf=None,
    split=[0.7, 0.2, 0.1],
    batch_s=32,
    num_workers=8,
    pin_memory=True,
):

    if dataname == "cifar10":
        dataclass = UnlearnCifar10
    elif dataname == "cifar100":
        dataclass = UnlearnCifar100
    elif dataname == "svnh":
        dataclass = UnlearnSVNH

    if itf is not None:
        with open(itf, "r") as f:
            itf = json.load(f)

    dataset = dataclass(
        split=split,
        transform=transform,
        class_to_forget=cf,
        idx_to_forget=itf,
        unlearning_ratio=unlr,
    )

    train_set = Subset(dataset, dataset.TRAIN)
    val_set = Subset(dataset, dataset.VAL)
    test_set = Subset(dataset, dataset.TEST)
    forget_set = Subset(dataset, dataset.FORGET)
    retain_set = Subset(dataset, dataset.RETAIN)

    shadow_subset = list(np.random.choice(dataset.TRAIN, len(dataset.FORGET)))
    shadow_set = Subset(dataset, shadow_subset)

    train_l = DataLoader(
        train_set,
        batch_size=batch_s,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_l = DataLoader(
        val_set,
        batch_size=batch_s,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_l = DataLoader(
        test_set,
        batch_size=batch_s,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    forget_l = DataLoader(
        forget_set,
        batch_size=batch_s,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    retain_l = DataLoader(
        retain_set,
        batch_size=batch_s,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    shadow_l = DataLoader(
        shadow_set,
        batch_size=batch_s,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_l, val_l, test_l, forget_l, retain_l, shadow_l


if __name__ == "__main__":

    transform = None

    DSET = "cifar10"

    ITF = None  # "checkpoints/resnet18_cifar10_pretrained_forget.json"
    UNLR = None  # 0.2  # None  # 0.2
    CF = 0  # None  # 0

    (
        train_loader,
        val_loader,
        test_loader,
        forget_loader,
        retain_loader,
        _,
    ) = get_dataloaders(DSET, None, unlr=UNLR, itf=ITF, cf=CF)
    ds = train_loader.dataset.dataset

    # ds.data[10][0].show()

    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    for data in forget_loader:

        image = data["image"]
        label = data["label"]

        img = transforms.ToPILImage()(image[0])

        # breakpoint()
        plt.imshow(img)
        plt.title(f"Label: {classes[label[0]]}")
        plt.show()

    breakpoint()
# if __name__ == "__main__":

#     # dataset = UnlearnSVNH()
#     # dataset = UnlearnCifar10()

#     # train val test
#     split = [0.7, 0.2, 0.1]
#     dataset = UnlearnSVNH(split=split)

#     train_set = Subset(dataset, dataset.TRAIN)
#     val_set = Subset(dataset, dataset.VAL)
#     test_set = Subset(dataset, dataset.TEST)
#     forget_set = Subset(dataset, dataset.FORGET)
#     retain_set = Subset(dataset, dataset.RETAIN)

#     train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
#     test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
#     forget_loader = DataLoader(forget_set, batch_size=32, shuffle=False)
#     retain_loader = DataLoader(retain_set, batch_size=32, shuffle=False)

#     for data in test_loader:

#         image = data["image"]
#         label = data["label"]

#         img = transforms.ToPILImage()(image[0])

#         # breakpoint()
#         plt.imshow(img)
#         plt.title(f"Label: {label[0]}")
#         plt.show()
#         # breakpoint()
