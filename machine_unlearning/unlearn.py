import torch
import random
import wandb
import copy
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from pytorch_lightning.loggers import WandbLogger
from copy import deepcopy

from .setups import get_cfg
from .datasets import get_datasets, get_forget_retain_sets
from .experiments import Experiment
from .utils import PrintCallback


def evaluate(cfg, debug=False, no_fit=False, evaluation=True):
    trainer = pl.Trainer(accelerator="gpu", max_epochs=cfg["epochs"], callbacks=[PrintCallback()] if debug else [])

    model = Experiment(cfg)  
    if not no_fit:  
        if debug:
            trainer.fit(model, cfg["data"], cfg["test_data"])
        else:
            trainer.fit(model, cfg["data"])

    retain, forget, test = None, None, None
    if evaluation:
        retain = trainer.validate(model, cfg["retain_data"])[0]
        forget = trainer.validate(model, cfg["forget_data"])[0]
        test = trainer.validate(model, cfg["test_data"])[0]

    return model, retain, forget, test

def finetune(model, dataset, forget_indices, lr=0.001, epochs=5, milestones=None, seed=0, debug=False):    
    cfg = get_cfg("train", dataset)
    cfg["model"] = model.model
    cfg["optimizer_lr"] = lr
    cfg["lr_scheduler"] = "multistep" if milestones is not None else "none"
    cfg["milestones"] = milestones
    cfg["epochs"] = epochs
    cfg["seed"] = seed

    train_dataset, test_dataset = get_datasets(dataset, augment=True)
    forget_set, retain_set = get_forget_retain_sets(dataset, train_dataset, forget_indices=forget_indices)
    
    retain_loader = DataLoader(retain_set, batch_size=128, shuffle=True, num_workers=4)
    forget_loader = DataLoader(forget_set, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    cfg["data"] = retain_loader 
    cfg["retain_data"] = retain_loader 
    cfg["forget_data"] = forget_loader
    cfg["test_data"] = test_loader      

    model, retain, forget, test = evaluate(cfg, debug=debug)

    return model, retain["val_accuracy"], forget["val_accuracy"], test["val_accuracy"]

def poison(model, dataset, forget_indices, lr=0.0007, epochs=1, seed=0, debug=False):    
    cfg = get_cfg("train", dataset)
    cfg["model"] = model.model
    cfg["optimizer_lr"] = lr
    cfg["lr_scheduler"] = "none"
    cfg["epochs"] = epochs
    cfg["seed"] = seed

    train_dataset, test_dataset = get_datasets(dataset, augment=True)
    forget_set, retain_set = get_forget_retain_sets(dataset, train_dataset, forget_indices=forget_indices)

    poisoned_forget_set = deepcopy(forget_set)
    poisoned_forget_set.dataset.targets = [random.randint(0, cfg["num_classes"] - 1) for _ in range(len(forget_set.dataset.targets))]
    
    poisoned_forget_loader = DataLoader(poisoned_forget_set, batch_size=128, shuffle=True, num_workers=4)
    retain_loader = DataLoader(retain_set, batch_size=128, shuffle=False, num_workers=4)
    forget_loader = DataLoader(forget_set, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    cfg["data"] = poisoned_forget_loader
    cfg["retain_data"] = retain_loader
    cfg["forget_data"] = forget_loader
    cfg["test_data"] = test_loader
    
    model, retain, forget, test = evaluate(cfg, debug=debug)

    return model, retain["val_accuracy"], forget["val_accuracy"], test["val_accuracy"]

def poison_full(model, dataset, forget_indices, lr=0.002, epochs=5, seed=0, debug=False):    
    cfg = get_cfg("train", dataset)
    cfg["model"] = model.model
    cfg["optimizer_lr"] = lr
    cfg["epochs"] = epochs
    cfg["seed"] = seed

    train_dataset, test_dataset = get_datasets(dataset, augment=True)
    forget_set, retain_set = get_forget_retain_sets(dataset, train_dataset, forget_indices=forget_indices)

    poisoned_forget_set = deepcopy(forget_set)
    poisoned_forget_set.dataset.targets = [random.randint(0, cfg["num_classes"] - 1) for _ in range(len(forget_set.dataset.targets))]
    
    train_loader = DataLoader(ConcatDataset([retain_set, poisoned_forget_set]), batch_size=128, shuffle=False, num_workers=4)
    retain_loader = DataLoader(retain_set, batch_size=128, shuffle=False, num_workers=4)
    forget_loader = DataLoader(forget_set, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    cfg["data"] = train_loader
    cfg["retain_data"] = retain_loader
    cfg["forget_data"] = forget_loader
    cfg["test_data"] = test_loader
    
    model, retain, forget, test = evaluate(cfg, debug=debug)
    
    return model, retain["val_accuracy"], forget["val_accuracy"], test["val_accuracy"]


def two_stage_simple(model, dataset, forget_indices, forget_lr=0.01, retain_lr=0.001, forget_epochs=1, retain_epochs=5, seed=0, debug=False):    
    cfg = get_cfg("train", dataset)
    cfg["model"] = model.model
    cfg["optimizer_lr"] = forget_lr
    cfg["lr_scheduler"] = "none"
    cfg["epochs"] = forget_epochs
    cfg["seed"] = seed

    train_dataset, test_dataset = get_datasets(dataset, augment=True)
    forget_set, retain_set = get_forget_retain_sets(dataset, train_dataset, forget_indices=forget_indices)

    poisoned_forget_set = deepcopy(forget_set)
    poisoned_forget_set.dataset.targets = [random.randint(0, cfg["num_classes"] - 1) for _ in range(len(forget_set.dataset.targets))]
    
    poisoned_forget_loader = DataLoader(poisoned_forget_set, batch_size=128, shuffle=True, num_workers=4)
    forget_loader = DataLoader(forget_set, batch_size=128, shuffle=True, num_workers=4)
    retain_loader = DataLoader(retain_set, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    cfg["data"] = poisoned_forget_loader
    cfg["retain_data"] = retain_loader
    cfg["forget_data"] = forget_loader
    cfg["test_data"] = test_loader
    
    # Forget Stage
    model, _, _, _ = evaluate(cfg, evaluation=False, debug=debug)
    
    # Retain Stage
    cfg["data"] = retain_loader
    cfg["optimizer_lr"] = retain_lr
    cfg["epochs"] = retain_epochs   

    # Only evaluate 
    model, retain, forget, test = evaluate(cfg, evaluation=True, debug=debug)

    return model, retain["val_accuracy"], forget["val_accuracy"], test["val_accuracy"]


def prune_last_layer(model, dataset, forget_indices, forget_lr=0.01, forget_epochs = 20, retain_lr=0.001, retain_epochs=5, quantile=0.3, seed=0, debug=False):
    # 1) Initialization stage
    cfg = get_cfg("train", dataset)
    cfg["lr_scheduler"] = "none"
    cfg["seed"] = seed

    train_dataset, test_dataset = get_datasets(dataset, augment=True)
    forget_set, retain_set = get_forget_retain_sets(dataset, train_dataset, forget_indices=forget_indices)

    forget_loader = DataLoader(forget_set, batch_size=128, shuffle=True, num_workers=4)
    retain_loader = DataLoader(retain_set, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    cfg["retain_data"] = retain_loader
    cfg["forget_data"] = forget_loader
    cfg["test_data"] = test_loader

    # 1a) Overfit a model on the forget set
    cfg["model"] = copy.deepcopy(model)
    cfg["data"] = forget_loader
    cfg["optimizer_lr"] = forget_lr
    cfg["epochs"] = forget_epochs

    forget_model , _, _, _ = evaluate(cfg, no_fit=False, evaluation=False, debug=debug)
    # 1b) Reinitialize weights on the last layer depending on value below threshold
    w_full = model.model.state_dict()
    w_forget = forget_model.model.model.state_dict()

    # Apply change to MLP (last layer weights)
    iterator = reversed(w_full)
    next(iterator)
    k = next(iterator)
    diff_forget = torch.abs(w_full[k] - w_forget[k])
    thresh_forget = torch.quantile(diff_forget.reshape(-1), quantile)
    tmp = torch.empty(w_full[k].shape)
    torch.nn.init.kaiming_normal_(tmp, nonlinearity='relu')
    # Reinitialize only the weights similar to the forget but different from the retain_sub
    cond = diff_forget < thresh_forget
    w_full[k][torch.where(cond)] = tmp[torch.where(cond)]  
    
    model.model.load_state_dict(w_full) 
    # 2) Retain stage
    cfg["model"] = model.model
    cfg["data"] = retain_loader
    cfg["optimizer_lr"] = retain_lr
    cfg["epochs"] = retain_epochs   
    
    # Train on retain set
    model, retain, forget, test = evaluate(cfg, evaluation=True, debug=debug)

    return model, retain["val_accuracy"], forget["val_accuracy"], test["val_accuracy"]

def selective_pruning_complex(model, dataset, forget_indices, overfit_lr=0.01, overfit_epochs=20, retain_lr=0.001, retain_epochs=5, retain_percentage=0.02, quantile = 0.3, seed=0, debug=False):
    # 1) Initialization stage
    cfg = get_cfg("train", dataset)
    cfg["lr_scheduler"] = "none"
    cfg["seed"] = seed

    train_dataset, test_dataset = get_datasets(dataset, augment=True)
    forget_set, retain_set = get_forget_retain_sets(dataset, train_dataset, forget_indices=forget_indices)

    forget_loader = DataLoader(forget_set, batch_size=128, shuffle=True, num_workers=4)
    retain_loader = DataLoader(retain_set, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    cfg["retain_data"] = retain_loader
    cfg["forget_data"] = forget_loader
    cfg["test_data"] = test_loader

    # 1a) Overfit a model on the forget set
    cfg["model"] = copy.deepcopy(model)
    cfg["data"] = forget_loader
    cfg["optimizer_lr"] = overfit_lr
    cfg["epochs"] = overfit_epochs

    forget_model , _, _, _ = evaluate(cfg, no_fit=False, evaluation=False, debug=debug)

    # 1b) Overfit a model on retain_percentage of the retain set
    retain_sub_indices = random.sample(range(0, len(retain_set)), int(retain_percentage*len(retain_set)))
    retain_sub_set = torch.utils.data.Subset(retain_set, retain_sub_indices)
    retain_sub_loader = DataLoader(retain_sub_set, batch_size=128, shuffle=True, num_workers=4)

    cfg["model"] = copy.deepcopy(model).model
    cfg["data"] = retain_sub_loader
    cfg["optimizer_lr"] = overfit_lr//3
    cfg["epochs"] = overfit_epochs

    retain_sub_model , _, _, _ = evaluate(cfg, no_fit=False, evaluation=False, debug=debug)

    # 1c) Reinitialize weights of the original model based on the two models
    w_full = model.model.state_dict()
    w_forget = forget_model.model.model.state_dict()
    w_retain_sub = retain_sub_model.model.state_dict()

    # Apply change to original model
    for k in w_full.keys():
        if len(w_full[k].shape) < 2:
            continue

        diff_forget = torch.abs(w_full[k] - w_forget[k])
        diff_retain_sub = torch.abs(w_full[k] - w_retain_sub[k])
        thresh_forget = torch.quantile(diff_forget.reshape(-1), quantile)
        thresh_retain_sub = torch.quantile(diff_retain_sub.reshape(-1), quantile)  

        tmp = torch.empty(w_full[k].shape)
        torch.nn.init.kaiming_normal_(tmp, nonlinearity='relu')
        # Reinitialize only the weights similar to the forget and to the retain_sub since hard to forget
        cond = torch.logical_and(diff_forget < thresh_forget, diff_retain_sub < thresh_retain_sub)
        w_full[k][torch.where(cond)] = tmp[torch.where(cond)]    
    
    model.model.load_state_dict(w_full)
    # 2) Retain stage
    cfg["model"] = model.model
    cfg["data"] = retain_loader
    cfg["optimizer_lr"] = retain_lr
    cfg["epochs"] = retain_epochs   
    
    # Train on retain set
    model, retain, forget, test = evaluate(cfg, evaluation=True, debug=debug)

    return model, retain["val_accuracy"], forget["val_accuracy"], test["val_accuracy"]


def activation_pruning(model, dataset, forget_indices, retain_lr=0.001, retain_epochs=5, retain_percentage=0.02, toll = 0.8, seed=0, reset_mlp=True, debug=False):
    # 1) Initialization stage
    cfg = get_cfg("train", dataset)
    cfg["lr_scheduler"] = "none"
    cfg["seed"] = seed
    
    train_dataset, test_dataset = get_datasets(dataset, augment=True)
    forget_set, retain_set = get_forget_retain_sets(dataset, train_dataset, forget_indices=forget_indices)

    forget_loader = DataLoader(forget_set, batch_size=128, shuffle=True, num_workers=1)
    retain_loader = DataLoader(retain_set, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    cfg["retain_data"] = retain_loader
    cfg["forget_data"] = forget_loader
    cfg["test_data"] = test_loader

    # Add feature extractor that adds all activation values of kernels and mean them over all the samples
    def get_activation(name, embeds):
        def hook(model, input, output):
            if "conv" not in name:
                return 
            # Init to first output
            if name not in embeds:
                embeds[name] = torch.sum(output.data, 0)
            else:
                embeds[name] += torch.sum(output.data, 0)
        return hook
    
    # 1a) Get features activation for all the forget set
    # Add feature extractor to all layers
    forget_embeds = {}
    forget_hooks = []
    for name, module in model.model.named_modules():
        forget_hooks.append(module.register_forward_hook(get_activation(f'{name}.weight', forget_embeds)))

    # Iterate over forget
    with torch.no_grad():
        for images, labels in forget_loader:
            out = model.model(images)
    # Remove hooks
    for hook in forget_hooks:
        hook.remove()
     
    # Calculate mean
    for i, key_layers in enumerate(forget_embeds.keys()):
        forget_embeds[key_layers] = forget_embeds[key_layers]/len(forget_set)
        
        
    # 1b) Get features activation for the retain set
    retain_embeds = {}
    retain_hooks = []
    for name, module in model.model.named_modules():
        retain_hooks.append(module.register_forward_hook(get_activation(f'{name}.weight', retain_embeds)))
    # Iterate over percentage retain set
    retain_sub_indices = random.sample(range(0, len(retain_set)), int(retain_percentage*len(retain_set)))
    retain_sub_set = torch.utils.data.Subset(retain_set, retain_sub_indices)
    retain_sub_loader = DataLoader(retain_sub_set, batch_size=128, shuffle=True, num_workers=4)
    with torch.no_grad():
        for images, labels in retain_sub_loader:
            out = model.model(images)
            
    # Remove hooks
    for hook in retain_hooks:
        hook.remove()
     
    # Calculate mean
    for i, key_layers in enumerate(retain_embeds.keys()):
        retain_embeds[key_layers] = retain_embeds[key_layers]/len(retain_set)        
        
        
    # 1c) Reinitialize weights of the original model based on the activation
    w_full = model.model.state_dict()

    # Apply change to original model
    for k in w_full.keys():
        # Consider only the weights

        if "weight" not in k or "conv" not in k:
            continue
            
        comparison = forget_embeds[k] > retain_embeds[k]
        # Add all the dimension to get number of activations greater in forget_embeds per kernel
        comparison = torch.sum(comparison, (1, 2))
        comparison = comparison > toll*forget_embeds[k].shape[1]*forget_embeds[k].shape[2]
        comparison = comparison.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(w_full[k].shape)         

        tmp = torch.empty(w_full[k].shape)
        torch.nn.init.kaiming_normal_(tmp, nonlinearity='relu')
        # Reinitialize only the weights similar to the forget but different from the retain_sub
        w_full[k][torch.where(comparison)] = tmp[torch.where(comparison)]
    
    # Reinitialize last mlp layer if specified
    if reset_mlp:
        w = model.model.state_dict()
        iterator = reversed(w)
        next(iterator)
        k = next(iterator)
        tmp = torch.empty(w_full[k].shape)
        torch.nn.init.kaiming_normal_(tmp, nonlinearity='relu')
        # Reinitialize only the weights similar to the forget but different from the retain_sub
        w_full[k] = tmp

    model.model.load_state_dict(w_full)
    
    # 2) Retain stage
    cfg["model"] = model.model
    cfg["data"] = retain_loader
    cfg["optimizer_lr"] = retain_lr
    cfg["epochs"] = retain_epochs   
    
    # Train on retain set
    model, retain, forget, test = unlearn.evaluate(cfg, evaluation=True, debug=debug)

    return model, retain["val_accuracy"], forget["val_accuracy"], test["val_accuracy"]


