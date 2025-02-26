#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")


# In[ ]:


GITHUB_TOKEN = "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"
USER = "mkraehenmann"
CLONE_URL = f"https://{USER}:{GITHUB_TOKEN}@github.com/{USER}/machine-unlearning.git"
get_ipython().system(f"git clone {CLONE_URL}")

import sys
sys.path.append("machine-unlearning")

os.environ["WANDB_API_KEY"] = "38ae9d0469697c6ccfcd59c7c7d5252f0b2318f4"


# # Load Datasets

# In[ ]:


from machine_unlearning.datasets import get_datasets

DATASET = "cifar10"

train_dataset, val_dataset = get_datasets(DATASET, "", test_size=0.1, augment=True)


# In[ ]:


from machine_unlearning.datasets import get_forget_retain_sets

with open('forget_set.txt', 'r') as fp:
    forget_indices = list(map(int, fp.read().split(',')))

forget_set, retain_set = get_forget_retain_sets(DATASET, train_dataset, forget_indices=forget_indices, debug=False)


# In[ ]:


from torch.utils.data import DataLoader

BATCH_SIZE = 100

forget_loader = DataLoader(forget_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# # Pytorch Lightning Model

# In[ ]:


from machine_unlearning.experiments import Experiment
from machine_unlearning.models.resnet import ResNet18

model = Experiment({"model": ResNet18(), "loss": torch.nn.CrossEntropyLoss(), "dataset": "cifar10"})


# # Eval

# In[ ]:


MODE = "unlearn"
model_path = "/kaggle/input/trained-resnet18/models/models"
num_models = 80
logits = torch.empty((len(forget_loader), BATCH_SIZE, num_models + 1, 10))


# In[ ]:


def one_hot(index, batch_sz):
    oh = torch.zeros((batch_sz, 10))
    oh[range(batch_sz), index] = 1.
    return oh


# In[ ]:


from machine_unlearning import unlearn

metrics = open("metrics.csv", 'w')
metrics.write("seed,retain_accuracy,forget_accuracy,test_accuracy\n")

#pt_paths = list(map(lambda x: os.path.join(os.path.abspath(model_path), x),os.listdir(model_path)))
for i in tqdm(range(num_models)):    
    if MODE == "retain":
        model.load_state_dict(torch.load(pt))
        
    elif MODE == "unlearn":
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)
        np.random.seed(i)
        
        # Load fully trained model
        model.load_state_dict(torch.load("machine-unlearning/data/resnet18_full.pt"))
        
        # Change this to change the unlearning approach
        model, retain_acc, forget_acc, test_acc = unlearn.two_stage_simple(model, DATASET, forget_indices, seed=i)
        metrics.write(f"{i},{retain_acc},{forget_acc},{test_acc}\n")
        
    else:
        raise Exception("Not a valid mode")
    
    with torch.no_grad():
        for j, xy in enumerate(forget_loader):
            x, y = xy
            
            # Put target in the first entry for each sample
            if i == 0:
                logits[j, :, i] = one_hot(y, BATCH_SIZE)
            
            out = model(x)
            logits[j, :, i + 1] = out

metrics.close()


# In[ ]:


logits = logits.reshape(-1, num_models + 1, 10).detach().cpu().numpy()


# In[ ]:


##### Sanity checks
print(logits.shape)
print(logits[2, 0])
print(logits[2, 1])
print(forget_set[2][1])
for log, f in zip(logits, forget_set):
    im, lab = f
    assert np.argmax(log[0]) == lab


# In[ ]:


np.save(f"logits_{MODE}.npy", logits)


# In[ ]:




