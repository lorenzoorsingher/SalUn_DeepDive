import json
import os
from dotenv import load_dotenv
import torch
from tqdm import tqdm
import wandb

from utils import get_args, load_checkpoint, gen_run_name
from datasets import get_dataloaders
from unlearn import compute_mask, eval_unlearning


def rand_label(model, image, target, idx, criterion, loader):
    # Assign random labels to forget data
    ds = loader.dataset.dataset
    forget_tensor = torch.tensor(ds.FORGET).to(DEVICE)
    which_is_in = (idx.unsqueeze(1) == forget_tensor).any(dim=1)
    rand_targets = torch.randint(1, len(ds.classes), target.shape).to(DEVICE)
    rand_targets = (target + rand_targets) % len(ds.classes)
    target[which_is_in] = rand_targets[which_is_in]

    output = model(image)
    loss = criterion(output, target)
    loss = loss.mean()

    return loss


def grad_ascent(model, image, target, idx, criterion, loader):
    output = model(image)
    loss = criterion(output, target)

    ds = loader.dataset.dataset
    forget_tensor = torch.tensor(ds.FORGET).to(DEVICE)
    which_is_in = (idx.unsqueeze(1) == forget_tensor).any(dim=1)
    loss[which_is_in] *= -1
    loss = loss.mean()

    return loss


def retrain(model, image, target, idx, criterion, loader):
    output = model(image)
    loss = criterion(output, target)
    loss = loss.mean()

    return loss


if __name__ == "__main__":

    args, args_dict = get_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    LOAD = args.load

    default = {
        "checkpoint": "checkpoints/resnet18_cifar10_best.pt",
        "class_to_forget": None,
        "unlearning_rate": None,
        "load_mask": False,
        "use_mask": True,
        "mask_thr": 0.5,
        "lr": 0.01,
        "epochs": 10,
        "method": "rl",
    }

    if LOAD == "":
        print("[LOADER] Loading parameters from command line")
        experiments = [args_dict]

    elif LOAD == "exp":
        print("[LOADER] Loading parameters from experiments set")
        experiments = [{}]
    else:
        print("[LOADER] Loading parameters from json file")
        experiments = json.load(open(LOAD, "r"))

    LOG = not args.no_log
    if LOG:
        load_dotenv()
        WANDB_SECRET = os.getenv("WANDB_SECRET")
        wandb.login(key=WANDB_SECRET)

    ###################################################################

    for nexp, exp in enumerate(experiments):

        settings = {**default, **exp}

        print(f"[EXP {nexp+1} of {len(experiments)}] Running settings: {settings}")

        CF = settings["class_to_forget"]
        CHKP = settings["checkpoint"]
        USE_MASK = settings["use_mask"]
        MASK_THR = settings["mask_thr"]
        LR = settings["lr"]
        UNLR = settings["unlearning_rate"]
        EPOCHS = settings["epochs"]
        METHOD = settings["method"]

        # LOAD_MASK = config["load_mask"]
        # MASK_PATH = f"checkpoints/mask_resnet18_cifar10_{CLASS_TO_FORGET}.pt"

        model, config, transform, opt = load_checkpoint(CHKP)

        DSET = config["dataset"]
        MODEL = config["model"]

        (
            train_loader,
            val_loader,
            test_loader,
            forget_loader,
            retain_loader,
        ) = get_dataloaders(DSET, transform, unlr=UNLR, cf=CF)

        model = model.to(DEVICE)

        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        if USE_MASK:
            mask = compute_mask(
                model,
                forget_loader,
                unlearn_lr=LR,
                saliency_threshold=MASK_THR,
                device=DEVICE,
            )

        if LOG:
            run_name = METHOD + "_" + gen_run_name()
            config = {**config, **settings}
            wandb.init(project="TrendsAndApps", name=run_name, config=config)

        print("[MAIN] Unlearning model")

        best_test_acc = 0
        best_test = {}
        best_forget_acc = 100
        best_forget = {}

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

                if METHOD == "rl":
                    loss = rand_label(
                        model, image, target, idx, criterion, train_loader
                    )
                elif METHOD == "ga":
                    loss = grad_ascent(
                        model, image, target, idx, criterion, train_loader
                    )
                elif METHOD == "retrain":
                    loss = retrain(model, image, target, idx, criterion, retain_loader)
                loss.backward()

                if USE_MASK:
                    for name, param in model.named_parameters():
                        if name in mask:
                            param.grad *= mask[name]

                optimizer.step()
                optimizer.zero_grad()

            accs = eval_unlearning(
                model,
                [test_loader, forget_loader, retain_loader],
                ["test", "forget", "retain"],
                DEVICE,
            )

            test_acc = accs["test"]
            forget_acc = accs["forget"]
            retain_acc = accs["retain"]

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test = accs
            if forget_acc < best_forget_acc:
                best_forget_acc = forget_acc
                best_forget = accs

            print(
                f"Test: {round(test_acc,2)}, Forget: {round(forget_acc,2)}, Retain: {round(retain_acc,2)}"
            )

            if LOG:
                wandb.log(
                    {
                        "test": test_acc,
                        "forget": forget_acc,
                        "retain": retain_acc,
                    }
                )

        print(f"Best test: {best_test}")
        print(f"Best forget: {best_forget}")
        wandb.finish()
