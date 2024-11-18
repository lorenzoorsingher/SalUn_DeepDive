import datetime
import timm
import torch
import argparse

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def get_model(model_name: str, num_classes: int, pretrained: bool = True):
    model = timm.create_model(model_name, num_classes=num_classes, pretrained=True)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return model, config, transform


def load_checkpoint(path):
    model_savefile = torch.load(path, weights_only=False)
    state_dict = model_savefile["model"]
    config = model_savefile["config"]
    opt = model_savefile["optimizer"]

    model_name = config["model"]
    num_classes = config["nclasses"]

    model = timm.create_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)

    model_config = resolve_data_config({}, model=model)
    transform = create_transform(**model_config)

    return model, config, transform, opt


def compute_topk(labels, outputs, k):

    _, indeces = outputs.topk(k)
    labels_rep = labels.unsqueeze(1).repeat(1, k)
    topk = (labels_rep == indeces).sum().item()

    return topk


def gen_run_name():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    return run_name


def get_args():
    """
    Function to get the arguments from the command line

    Returns:
    - args (dict): arguments
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="""Get the params""",
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        help="Path to the checkpoint",
        default="checkpoints/resnet18_cifar10_best.pt",
        metavar="",
    )

    parser.add_argument(
        "-CF",
        "--class-to-forget",
        type=int,
        help="Index of the class to forget",
        default=0,
        metavar="",
    )

    parser.add_argument(
        "-E",
        "--epochs",
        type=int,
        help="Number of epochs",
        default=10,
        metavar="",
    )

    parser.add_argument(
        "-UL",
        "--unlearning-rate",
        type=float,
        help="Percentage of dataset to be unlearned",
        default=None,
        metavar="",
    )

    parser.add_argument(
        "-LM",
        "--load-mask",
        type=str,
        help="Path to the mask",
        default="",
        metavar="",
    )

    parser.add_argument(
        "-L",
        "--load",
        type=str,
        help="Path to the json file or 'exp' to use the experiments set",
        default="",
        metavar="",
    )

    parser.add_argument(
        "-M",
        "--method",
        type=str,
        help="Method of unlearning [rl, ga]",
        default="rl",
        metavar="",
    )

    parser.add_argument(
        "-UM",
        "--use-mask",
        action="store_true",
        help="Set to use SalUn mask",
        default=False,
    )

    parser.add_argument(
        "-NL",
        "--no-log",
        action="store_true",
        help="Set to not log the results",
        default=False,
    )

    parser.add_argument(
        "-LR",
        "--lr",
        type=float,
        help="Learning rate",
        default=0.1,
        metavar="",
    )

    parser.add_argument(
        "-MT",
        "--mask-thr",
        type=float,
        help="Threshold for the mask",
        default=0.5,
        metavar="",
    )

    args = parser.parse_args()
    args_dict = vars(args)
    return args, args_dict
