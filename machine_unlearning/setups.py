import torch
from .utils import resolve_classes, resolve_model, resolve_loss

def train_cfg(dataset):
    return {
        "model": resolve_model(dataset),
        "dataset": dataset,
        "setup": "train",
        "loss": resolve_loss(dataset),
        "num_classes": resolve_classes(dataset),
        "optimizer": "sgd",
        "optimizer_lr": 0.1,
        "optimizer_momentum": 0.9,
        "optimizer_weight_decay": 5e-4,
        "lr_scheduler": "cosine",
        "epochs": 200
    }

def forget_cfg(dataset):
    return {
        "model": resolve_model(dataset),
        "dataset": dataset,
        "setup": "forget",
        "loss": lambda x, y: -(torch.nn.CrossEntropyLoss()(x, y)),
        "num_classes": resolve_classes(dataset),
        "optimizer": "sgd",
        "optimizer_lr": 1e-3,
        "optimizer_momentum": 0.0,
        "optimizer_weight_decay": 0,
        "lr_scheduler": "none",
        "epochs": 40
    }    

def retain_cfg(dataset):
    return {
        "model": resolve_model(dataset),
        "dataset": dataset,
        "setup": "retain",
        "loss": torch.nn.CrossEntropyLoss(),
        "num_classes": resolve_classes(dataset),
        "optimizer": "sgd",
        "optimizer_lr": 1e-3,
        "optimizer_momentum": 0.9,
        "optimizer_weight_decay": 5e-4,
        "lr_scheduler": "cosine",
        "epochs": 40
    }

def get_cfg(setup, dataset):
    return eval(f"{setup}_cfg")(dataset)