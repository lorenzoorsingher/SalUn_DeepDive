import random
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class UnlearningDataset(Dataset):
    def __init__(
        self,
        unlearning_ratio=0.1,
        split=[0.7, 0.2, 0.1],
        transform=None,
        class_to_forget=None,
    ):

        self.transform = transform
        self.class_to_forget = class_to_forget
        self.idxs = None
        self.VAL = None
        self.FORGET = None
        self.RETAIN = None

        self.unlearning_ratio = unlearning_ratio

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
        return data

    def split_unlearning(self):

        # Split the training set into train and validation
        self.VAL = random.sample(self.TRAIN, int(len(self.TRAIN) * self.val_split))
        self.TRAIN = list(set(self.TRAIN) - set(self.VAL))

        # Split the training set into forget and retain
        if(self.class_to_forget is not None):
            self.FORGET = [idx for idx in self.TRAIN if self.data[idx][1] == self.class_to_forget]
            self.RETAIN = list(set(self.TRAIN) - set(self.FORGET))
        else:
            num_images_to_forget = int(len(self.TRAIN) * self.unlearning_ratio)
            self.FORGET = random.sample(self.TRAIN, num_images_to_forget)
            self.RETAIN = list(set(self.TRAIN) - set(self.FORGET))

        print(
            f"Train samples: {len(self.TRAIN)} - Forget samples: {len(self.FORGET)} - Unlearn Ratio: {self.unlearning_ratio}"
        )

    def set_transform(self, transform):
        self.transform = transform
    def get_forget_dataLoader(self):
        forget_idxs = self.FORGET
        forget_dataset = [self.data[i] for i in forget_idxs]
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        forget_dataset = [(transform(image), label) for image, label in forget_dataset]
        
        forget_dataloader = DataLoader(forget_dataset, batch_size=32, shuffle=True)
        return forget_dataloader


class UnlearnCifar10(UnlearningDataset):
    ## Dictionary to map CIFAR-10 labels to class names
    #0: 'Airplane',
    #1: 'Automobile',
    #2: 'Bird',
    #3: 'Cat',
    #4: 'Deer',
    #5: 'Dog',
    #6: 'Frog',
    #7: 'Horse',
    #8: 'Ship',
    #9: 'Truck'

    def __init__(self, unlearning_ratio=0.1, split=[0.7, 0.2, 0.1], transform=None, class_to_forget=None):

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
            class_to_forget=class_to_forget
        )


class UnlearnCifar100(UnlearningDataset):
    def __init__(self, unlearning_ratio=0.1, split=[0.7, 0.2, 0.1], transform=None):

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
        )


class UnlearnSVNH(UnlearningDataset):
    def __init__(self, unlearning_ratio=0.1, split=[0.7, 0.2, 0.1], transform=None):

        test_split = split[2]
        self.data = datasets.SVHN(root="./data", download=True, transform=None)

        TRAIN = [i for i in range(len(self.data))]

        self.TEST = [i for i in range(int(len(self.data) * test_split))]

        self.TRAIN = list(set(TRAIN) - set(self.TEST))

        super(UnlearnSVNH, self).__init__(
            unlearning_ratio=unlearning_ratio,
            split=split,
            transform=transform,
        )


if __name__ == "__main__":

    # dataset = UnlearnSVNH()
    # dataset = UnlearnCifar10()

    # train val test
    split = [0.7, 0.2, 0.1]
    dataset = UnlearnSVNH(split=split)

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

    for data in test_loader:

        image = data["image"]
        label = data["label"]

        img = transforms.ToPILImage()(image[0])

        # breakpoint()
        plt.imshow(img)
        plt.title(f"Label: {label[0]}")
        plt.show()
        # breakpoint()
