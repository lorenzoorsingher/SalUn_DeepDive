import json
import os
import random
import torch
import wandb
import numpy as np

from tqdm import tqdm
from dotenv import load_dotenv

from methods import rand_label, grad_ascent, grad_ascent_small, retrain
from utils import (
    get_args,
    load_checkpoint,
    gen_run_name,
    get_model,
    set_seed,
    get_avg_std,
)
from eval import compute_basic_mia, eval_unlearning
from datasets import get_dataloaders
from unlearn import compute_mask


def train(model, loader, method, criterion, optimizer, device, mask=None):

    model.train()

    for data in tqdm(loader):

        image = data["image"]
        target = data["label"]
        idx = data["idx"]

        image = image.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True)
        idx = idx.to(DEVICE, non_blocking=True)

        loss = method(model, image, target, idx, criterion, loader, device)
        loss.backward()

        if USE_MASK:
            for name, param in model.named_parameters():
                if name in mask:
                    param.grad *= mask[name]

        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":

    args, args_dict = get_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    LOAD = args.load

    default = {
        "checkpoint": "checkpoints/resnet18_cifar10_pretrained_best.pt",
        "class_to_forget": None,
        "unlearning_rate": None,
        "idxs_to_forget": None,
        "load_mask": False,
        "use_mask": True,
        "mask_thr": 0.5,
        "lr": 0.1,
        "epochs": 10,
        "method": "rl",
        "tag": None,
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

    # Create output folder
    run_folder = f"output/{gen_run_name()}"
    os.makedirs(run_folder, exist_ok=True)

    ###################################################################
    NEXP = args.nexp
    for nexp, exp in enumerate(experiments):

        all_results = []

        runid = gen_run_name()

        # Set seed
        set_seed(0)

        for expidx in range(NEXP):

            settings = {**default, **exp}

            print(f"[EXP {nexp+1} of {len(experiments)}] Running settings: {settings}")
            print(f"[RUN {expidx+1} of {NEXP}]")
            CF = settings["class_to_forget"]
            CHKP = settings["checkpoint"]
            USE_MASK = settings["use_mask"]
            MASK_THR = settings["mask_thr"]
            LR = settings["lr"]
            UNLR = settings["unlearning_rate"]
            ITF = settings["idxs_to_forget"]
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
                _,
            ) = get_dataloaders(DSET, transform, unlr=UNLR, itf=ITF, cf=CF)

            if METHOD == "retrain":
                classes = train_loader.dataset.dataset.classes
                model, _, _ = get_model(MODEL, len(classes), False)

            model = model.to(DEVICE)
            optimizer = torch.optim.SGD(model.parameters(), lr=LR)
            criterion = torch.nn.CrossEntropyLoss(reduction="none")

            if LOG:
                config = {**config, **settings}
                config["runid"] = runid
                config["runidx"] = expidx
                run_name = METHOD + "_" + gen_run_name(config) + f"_{expidx}"
                wandb.init(project="TrendsAndApps", name=run_name, config=config)

            mask = None
            if USE_MASK:
                mask = compute_mask(
                    model,
                    forget_loader,
                    unlearn_lr=LR,
                    saliency_threshold=MASK_THR,
                    device=DEVICE,
                )

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

            # -------------------------------------------------------------

            if LOG:
                wandb.log(
                    {
                        "base_test": accs["test"],
                        "base_forget": accs["forget"],
                        "base_retain": accs["retain"],
                        "base_val": accs["val"],
                        "base_mia_auc": mia_auc,
                        "base_mia_acc": mia_acc,
                    }
                )
            print("[MAIN] Unlearning model")

            if METHOD == "rl":
                method = rand_label
                loader = train_loader
            if METHOD == "rl_split":
                method = rand_label
                loader = forget_loader
            elif METHOD == "ga":
                method = grad_ascent
                loader = train_loader
            elif METHOD == "ga_small":
                method = grad_ascent_small
                loader = forget_loader
            elif METHOD == "retrain":
                method = retrain
                loader = retain_loader

            for epoch in range(EPOCHS):

                print(f"Epoch {epoch}")

                train(model, loader, method, criterion, optimizer, DEVICE, mask)

                if METHOD == "rl_split":
                    print("[MAIN] Fine tuning")
                    train(
                        model,
                        retain_loader,
                        retrain,
                        criterion,
                        optimizer,
                        DEVICE,
                        mask,
                    )

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
                accs["mia_auc"] = mia_auc
                accs["mia_acc"] = mia_acc
                test_acc = accs["test"]
                forget_acc = accs["forget"]
                retain_acc = accs["retain"]
                val_acc = accs["val"]

                for key, value in accs.items():
                    print(f"{key}: {round(value,3)}")
                print(f"MIA AUC: {round(mia_auc,3)}, MIA ACC: {round(mia_acc,3)}")

                if LOG:
                    wandb.log(
                        {
                            "test": test_acc,
                            "forget": forget_acc,
                            "retain": retain_acc,
                            "val": val_acc,
                            "mia_auc": mia_auc,
                            "mia_acc": mia_acc,
                        }
                    )

                # -------------------------------------------------------------
            all_results.append(accs)

            if expidx == NEXP - 1:
                print("[MAIN] Computing average results")

                final_results = get_avg_std(all_results)

                model_savefile = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                }

                model_savepath = f"{run_folder}/{runid}_{CF}.pt"

                torch.save(model_savefile, model_savepath)

                if LOG:
                    wandb.log(final_results)

            if LOG:
                wandb.finish()

        # Compute average results
