import timm
import torch

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
