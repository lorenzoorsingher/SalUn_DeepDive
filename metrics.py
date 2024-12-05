from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm

from utils import compute_topk


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

    with torch.no_grad():
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
