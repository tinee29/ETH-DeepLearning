import sys
import os
import torch

assert len(sys.argv) == 2, "Wrong number of arguments, usage: python3 change_weights.py <data_path>"

w_full = torch.load(os.path.join(sys.argv[1], "resnet18_full.pt"))
w_forget = torch.load(os.path.join(sys.argv[1], "resnet18_forget.pt"))

for k in w_full.keys():
    if len(w_full[k].shape) < 2:
        continue
    print(k)
    diff = torch.abs(w_full[k] - w_forget[k])
    thresh = torch.quantile(diff.reshape(-1), 0.2)    
    tmp = torch.empty(w_full[k].shape)
    torch.nn.init.kaiming_normal_(tmp, nonlinearity='relu')
    w_full[k][torch.where(diff < thresh)] = tmp[torch.where(diff < thresh)]        

torch.save(w_full, os.path.join(sys.argv[1], "resnet18_change.pt"))
    