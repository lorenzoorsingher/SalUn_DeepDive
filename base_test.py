import json
import os
import random
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm
import wandb
import scipy.stats as st

from utils import (
    get_args,
    load_checkpoint,
    gen_run_name,
    compute_topk,
    get_model,
    set_seed,
)
from datasets import get_dataloaders
from unlearn import compute_mask
import numpy as np


def compute_basic_mia(retain_losses, forget_losses, val_losses, test_losses):
    train_loss = (
        torch.cat((retain_losses, val_losses), dim=0).unsqueeze(1).cpu().numpy()
    )
    train_target = torch.cat(
        (torch.ones(retain_losses.size(0)), torch.zeros(val_losses.size(0))), dim=0
    ).numpy()
    test_loss = (
        torch.cat((forget_losses, test_losses), dim=0).unsqueeze(1).cpu().numpy()
    )
    test_target = (
        torch.cat((torch.ones(forget_losses.size(0)), torch.zeros(test_losses.size(0))))
        .cpu()
        .numpy()
    )

    best_auc = 0
    best_acc = 0
    for n_est in [20, 50, 100]:
        for criterion in ["gini", "entropy"]:
            mia_model = RandomForestClassifier(
                n_estimators=n_est, criterion=criterion, n_jobs=8, random_state=0
            )
            mia_model.fit(train_loss, train_target)

            y_hat = mia_model.predict_proba(test_loss)[:, 1]
            auc = roc_auc_score(test_target, y_hat) * 100
            # breakpoint()
            y_hat = mia_model.predict(forget_losses.unsqueeze(1).cpu().numpy()).mean()

            # breakpoint()
            acc = (1 - y_hat) * 100

            if acc > best_acc:
                best_acc = acc
                best_auc = auc
    # breakpoint()
    return best_auc, best_acc


def eval_unlearning(model, loaders, names, criterion, DEVICE):

    model.eval()
    tot_acc = 0
    accs = {}
    losses = {}
    for loader, name in zip(loaders, names):

        losses[name] = []
        for data in tqdm(loader):

            image = data["image"]
            target = data["label"]
            image = image.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            output = model(image)
            loss = criterion(output, target)

            losses[name].append(loss.mean().item())

            acc = compute_topk(target, output, 1)

            tot_acc += acc

        tot_acc /= len(loader.dataset)
        accs[name] = tot_acc
    return accs, losses


if __name__ == "__main__":

    args, args_dict = get_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    experiments = [
        (
            "checkpoints/resnet18_cifar10_pretrained_best.pt",
            "checkpoints/resnet18_cifar10_pretrained_forget.json",
        ),
        (
            "checkpoints/resnet18_cifar10_pretrained_forget.pt",
            "checkpoints/resnet18_cifar10_pretrained_forget.json",
        ),
    ]

    for exp in experiments:

        # Set seed
        set_seed(0)

        CHKP, ITF = exp

        model, config, transform, opt = load_checkpoint(CHKP)

        DSET = config["dataset"]
        MODEL = config["model"]

        (
            train_loader,
            val_loader,
            test_loader,
            forget_loader,
            retain_loader,
            _,
        ) = get_dataloaders(DSET, transform, unlr=None, itf=ITF, cf=None)

        model = model.to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        # -------------------------------------------------------------

        print("[MAIN] Evaluating model")
        accs, losses = eval_unlearning(
            model,
            [test_loader, forget_loader, retain_loader, val_loader],
            ["test", "forget", "retain", "val"],
            criterion,
            DEVICE,
        )
        accs["forget"] = 1 - accs["forget"]

        print("[MAIN] Computing MIA")
        mia_auc, mia_acc = compute_basic_mia(
            torch.tensor(losses["retain"]),
            torch.tensor(losses["forget"]),
            torch.tensor(losses["val"]),
            torch.tensor(losses["test"]),
        )

        for key, value in accs.items():
            print(f"{key}: {round(value,3)}")
        print(f"MIA AUC: {round(mia_auc,3)}, MIA ACC: {round(mia_acc,3)}")
