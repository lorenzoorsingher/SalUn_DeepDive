import json
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm
import wandb
import scipy.stats as st

from utils import get_args, load_checkpoint, gen_run_name, compute_topk, get_model
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


def model_losses(losses):
    losses_conf = np.exp(-losses)
    losses_scaled = np.log(losses_conf / (1 - losses_conf))

    mu = losses_scaled.mean()
    sigma = losses_scaled.std()

    return losses_scaled, mu, sigma


def compute_e_mia(retain_losses, forget_losses, val_losses, test_losses):

    retain_scaled, retain_mu, retain_sigma = model_losses(retain_losses)
    forget_scaled, forget_mu, forget_sigma = model_losses(forget_losses)
    val_scaled, val_mu, val_sigma = model_losses(val_losses)
    test_scaled, test_mu, test_sigma = model_losses(test_losses)

    nonm_z = (forget_scaled - test_mu) / test_sigma
    mem_z = (forget_scaled - retain_mu) / retain_sigma

    nonm_z = nonm_z.abs()
    mem_z = mem_z.abs()

    nonm_prob = 1 - (st.norm.cdf(nonm_z) - st.norm.cdf(-nonm_z))
    mem_prob = 1 - (st.norm.cdf(mem_z) - st.norm.cdf(-mem_z))

    nonm_prob = nonm_prob.mean().item()
    mem_prob = mem_prob.mean().item()

    # SCORE ------------------
    score = nonm_prob / (nonm_prob + mem_prob + 1e-6)

    # LR SCORE ------------------
    lr_score = nonm_prob / (mem_prob + 1e-6)

    # ACC ------------------
    nonm_prob = 1 - (st.norm.cdf(nonm_z) - st.norm.cdf(-nonm_z))
    mem_prob = 1 - (st.norm.cdf(mem_z) - st.norm.cdf(-mem_z))
    acc = (nonm_prob > mem_prob).sum() / len(forget_losses)

    return score, nonm_prob, mem_prob, lr_score, acc


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

    LOAD = args.load

    default = {
        "checkpoint": "checkpoints/resnet18_cifar10_best.pt",
        "class_to_forget": None,
        "unlearning_rate": None,
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

    ###################################################################

    for nexp, exp in enumerate(experiments):

        settings = {**default, **exp}

        print(f"[EXP {nexp+1} of {len(experiments)}] Running settings: {settings}")

        CF = settings["class_to_forget"]
        UNLR = settings["unlearning_rate"]
        CHKP = settings["checkpoint"]
        LR = settings["lr"]
        EPOCHS = settings["epochs"]

        # LOAD_MASK = config["load_mask"]
        # MASK_PATH = f"checkpoints/mask_resnet18_cifar10_{CLASS_TO_FORGET}.pt"

        model, config, transform, opt = load_checkpoint(CHKP)

        DSET = config["dataset"]
        MODEL = config["model"]

        model = model.to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        if LOG:
            config = {**config, **settings}
            run_name = "MIA_" + gen_run_name(config)
            wandb.init(project="TrendsAndApps", name=run_name, config=config)

        # -------------------------------------------------------------

        # -------------------------------------------------------------

        print("[MAIN] Unlearning model")

        for epoch in range(EPOCHS):

            print(f"Epoch {epoch}")

            (
                train_loader,
                val_loader,
                test_loader,
                forget_loader,
                retain_loader,
                shadow_loader,
            ) = get_dataloaders(DSET, transform, unlr=UNLR, cf=CF)

            (
                _,
                _,
                _,
                c2_loader,
                _,
                _,
            ) = get_dataloaders(DSET, transform, unlr=UNLR, cf=CF + 1)

            # -------------------------------------------------------------

            print("[MAIN] Evaluating model")
            _, losses = eval_unlearning(
                model,
                [test_loader, forget_loader, retain_loader, val_loader, c2_loader],
                ["test", "forget", "retain", "val", "c2"],
                criterion,
                DEVICE,
            )

            print("[MAIN] Computing MIA")
            mia_score, nonm, mem, mia_lr, emia_acc = compute_e_mia(
                torch.tensor(losses["retain"]),
                torch.tensor(losses["forget"]),
                torch.tensor(losses["val"]),
                torch.tensor(losses["test"]),
            )
            mia_auc, mia_acc = compute_basic_mia(
                torch.tensor(losses["retain"]),
                torch.tensor(losses["forget"]),
                torch.tensor(losses["val"]),
                torch.tensor(losses["test"]),
            )

            print(f"MIA AUC: {round(mia_auc,2)}, MIA ACC: {round(mia_acc,2)}")
            print(
                f"MIA Score: {round(mia_score,2)} EMIA ACC: {round(emia_acc,2)}  MIA LR: {round(mia_lr,2)}"
            )

            print("[MAIN] Computing MIA2")
            mia_score, nonm, mem, mia_lr, emia_acc = compute_e_mia(
                torch.tensor(losses["retain"]),
                torch.tensor(losses["val"]),
                torch.tensor(losses["forget"]),
                torch.tensor(losses["test"]),
            )
            mia_auc, mia_acc = compute_basic_mia(
                torch.tensor(losses["retain"]),
                torch.tensor(losses["val"]),
                torch.tensor(losses["forget"]),
                torch.tensor(losses["test"]),
            )

            print(f"MIA AUC: {round(mia_auc,2)}, MIA ACC: {round(mia_acc,2)}")
            print(
                f"MIA Score: {round(mia_score,2)} EMIA ACC: {round(emia_acc,2)}  MIA LR: {round(mia_lr,2)}"
            )

            if LOG:
                wandb.log(
                    {
                        "mia_auc": mia_auc,
                        "mia_acc": mia_acc,
                        "mia_score": mia_score,
                        "nonm_mem": nonm,
                        "mia_mem": mem,
                        "mia_lr": mia_lr,
                    }
                )

            # -------------------------------------------------------------

        wandb.finish()
