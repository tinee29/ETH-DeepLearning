#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, EarlyStopping
import wandb
import random
from pytorch_lightning.loggers import WandbLogger

sys.path.append("machine-unlearning")

from machine_unlearning.datasets import get_datasets
from machine_unlearning.datasets import get_forget_retain_sets
from machine_unlearning.models.resnet import ResNet18
from machine_unlearning.setups import get_cfg
from machine_unlearning.utils import resolve_model, resolve_classes, resolve_loss
from machine_unlearning.experiments import Experiment


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

DATASET = "cifar10"
path = "/kaggle/input/utkface-new/UTKFace"

train_dataset, val_dataset = get_datasets(DATASET, path, test_size=0.1, augment=True)

with open('forget_set.txt', 'r') as fp:
    forget_indices = list(map(int, fp.read().split(',')))

forget_set, train_dataset = get_forget_retain_sets(DATASET, train_dataset, forget_indices=forget_indices)

BATCH_SIZE = 128
SETUP = "train"

cfg = get_cfg(SETUP, DATASET)

class PrintCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Training loss: {trainer.callback_metrics['train_loss']}, Accuracy: {trainer.callback_metrics['train_accuracy']}")
        
    def on_validation_end(self, trainer, pl_module):
        print(f"Validation loss: {trainer.callback_metrics['val_loss']}, Accuracy: {trainer.callback_metrics['val_accuracy']}")

#os.environ["WANDB_API_KEY"] = "38ae9d0469697c6ccfcd59c7c7d5252f0b2318f4"
#get_ipython().system('wandb login')

for seed in range(64):
    # Set seeds and device
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Custom cfg
    cfg = {
        "model": resolve_model(DATASET),
        "dataset": DATASET,
        "setup": "train",
        "loss": resolve_loss(DATASET),
        "num_classes": resolve_classes(DATASET),
        "only_train_fc": False,
        "optimizer": "sgd",
        "optimizer_lr": 0.1,
        "optimizer_momentum": 0.9,
        "optimizer_weight_decay": 5e-4,
        "lr_scheduler": "multistep",
        "milestones": [35, 70, 90],
        "epochs": 100,
        "data": eval(f"{SETUP}_loader"),
        "val_data": val_loader
    }

    model = Experiment(cfg)


    wandb_logger = WandbLogger(
        log_model=False, 
        project="machine-unlearning", 
        name=f"{seed}_{cfg['epochs']}epochs_{cfg['setup']}setup_{DATASET}"
    )
    wandb_logger.experiment.config.update(cfg)
    trainer = pl.Trainer(accelerator="gpu", max_epochs=cfg["epochs"], logger=wandb_logger, callbacks=[PrintCallback()])#, EarlyStopping(monitor="val_loss", mode="min")])

    trainer.fit(model, cfg["data"], cfg["val_data"])
    
    wandb.finish()

    torch.save(model.state_dict(), f'models/resnet18_seed{seed}.pt')


