import timm
import torch

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def get_model(model_name: str, num_classes: int, pretrained: bool = True):
    model = timm.create_model("resnet18", num_classes=num_classes, pretrained=True)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return model, config, transform


def compute_topk(labels, outputs, k):

    _, indeces = outputs.topk(k)
    labels_rep = labels.unsqueeze(1).repeat(1, k)
    topk = (labels_rep == indeces).sum().item()

    return topk


def load_checkpoint(path):
    model_savefile = torch.load(path, weights_only=False)
    model = model_savefile["model"]
    config = model_savefile["config"]
    transform = create_transform(**config)
    opt = model_savefile["optimizer"]
    return model, config, transform, opt
