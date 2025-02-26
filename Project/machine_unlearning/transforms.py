import torch
import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision.transforms import transforms
from torchvision.transforms import v2

# CIFAR10
cifar10_base_transform = v2.Compose([ 
    v2.ToImageTensor(), 
    v2.ConvertImageDtype(),
    v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
#cifar10_base_transform = v2.Compose([ 
#    v2.ToImage(), 
#    v2.ToDtype(torch.float32, scale=True),
#    v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#])

cifar10_train_transform = v2.Compose([
    v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
    v2.RandomCrop(32, padding=4, padding_mode='reflect'),
    v2.RandomHorizontalFlip(),    
    cifar10_base_transform
])

# IMAGENET/UTKFACE
utkface_base_transform = v2.Compose([ 
    v2.ToImageTensor(), 
    v2.ConvertImageDtype(),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
utkface_train_transform = v2.Compose([
    v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
    v2.RandomCrop(100, padding=4, padding_mode='reflect'),
    v2.RandomHorizontalFlip(),    
    utkface_base_transform
])

# FG-Net
fgnet_base_transform = v2.Compose([ 
    v2.ToImageTensor(), 
    v2.ConvertImageDtype(),
    v2.Resize((100, 100), antialias=True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
fgnet_train_transform = fgnet_base_transform

# MORPH
morph_base_transform = v2.Compose([ 
    v2.ToImageTensor(), 
    v2.ConvertImageDtype(),
    v2.Resize((180, 180), antialias=True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
morph_train_transform = v2.Compose([
    v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
    v2.RandomHorizontalFlip(),    
    morph_base_transform
])

# AgeDB
agedb_base_transform = v2.Compose([ 
    v2.ToImageTensor(), 
    v2.ConvertImageDtype(),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
agedb_train_transform = v2.Compose([
    v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
    v2.RandomHorizontalFlip(),    
    morph_base_transform
])

def get_transforms(name):
    return eval(f"{name}_train_transform"), eval(f"{name}_base_transform")
