RETRAINED_CHECKPOINTS = "checkpoints/regression/"
NUM_RUNS = 80

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from transformers import ResNetForImageClassification, Trainer, TrainingArguments
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import warnings
import os
from tqdm.auto import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import det_curve
import json

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


assert(os.path.exists(RETRAINED_CHECKPOINTS)), "Path to retrained checkpoints does not exist"
assert(os.path.exists('data/AgeDB/')), "Path to AgeDB dataset does not exist"

# helper methods and classes

def get_model(checkpoint):
    """
    Get the model with the specified checkpoint
    """
    
    model = ResNetForImageClassification.from_pretrained(
        checkpoint,
        num_labels=1,
        problem_type="regression",
        ignore_mismatched_sizes=True,
    )

    return model

class AgeDBDataset(Dataset):
    """
    This class is used to load the AgeDB dataset for retain set and subject inclusive validation.

    DO NOT MODIFY TO KEEP RETAIN SET EXACTLY AS IN generate_retrained_models.ipynb
    """

    def __init__(self, folders=[0,1,2,3,4,5,6,7,8], start_ratio=0., end_ratio=8/9):

        self.db = pd.read_csv('data/AgeDB/AgeDB.csv')
        mask = self.db['folder'] == folders[0]
        for folder in folders[1:]:
            mask = mask | (self.db['folder'] == folder)
        self.db = self.db[mask].reset_index(drop=True)

        self.imgs = torch.load('data/AgeDB/AgeDB.pt')[mask]     

        self.labels = np.expand_dims(self.db['age'].to_numpy(), axis=1).astype(np.float32)

        # self.transform = transforms.Compose([
        #     transforms.Lambda(lambda img: img.float().div(255)),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     transforms.Resize((224, 224)),
        # ])

        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.float().div(255)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
            # add random noise
            transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.1),
        ])

        self.start_index = int(start_ratio * self.labels.shape[0])
        self.end_index = int(end_ratio * self.labels.shape[0])

        self.db = self.db.iloc[self.start_index:self.end_index].reset_index(drop=True)
        self.imgs = self.imgs[self.start_index:self.end_index]
        self.labels = self.labels[self.start_index:self.end_index]


    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image = self.imgs[idx]
        image = self.transform(image)
        label = self.labels[idx]
        
        return {"pixel_values": image, "labels": label}
    
class AgeDBForgetDataset(Dataset):
    """
    This class is used to load the AgeDB dataset for forget set.

    DO NOT MODIFY TO KEEP RETAIN SET EXACTLY AS IN generate_retrained_models.ipynb
    """

    def __init__(self, folders=[9]):

        self.db = pd.read_csv('data/AgeDB/AgeDB.csv')
        mask = self.db['folder'] == folders[0]
        for folder in folders[1:]:
            mask = mask | (self.db['folder'] == folder)
        self.db = self.db[mask].reset_index(drop=True)

        self.imgs = torch.load('data/AgeDB/AgeDB.pt')[mask]
        
        self.labels = np.expand_dims(self.db['age'].to_numpy(), axis=1).astype(np.float32)

        # self.transform = transforms.Compose([
        #     transforms.Lambda(lambda img: img.float().div(255)),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     transforms.Resize((224, 224)),
        # ])

        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.float().div(255)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
            # add random noise
            transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.1),
        ])

        names = ['AnnJillian', 'SophiaLoren', 'MuhammadAli', 'JaneWyman', 'MichellePfeiffer', 
                'BillCosby', 'DeborahKerr', 'RobertPowell', 'RobertMitchum']
        mask = self.db['name'].isin(names)
        self.db = self.db[mask].reset_index(drop=True)
        self.imgs = self.imgs[mask]
        self.labels = self.labels[mask]


    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image = self.imgs[idx]
        image = self.transform(image)
        label = self.labels[idx]

        return {"pixel_values": image, "labels": label}

class NoTransformDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.float().div(255)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224)),
        ])

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

def compute_metrics(eval_pred):
    """
    Compute the evaluation metric based on the predictions and labels.
    Used by the Trainer class.

    Args:
        - eval_pred: tuple of predictions and labels as numpy arrays
    
    Returns:
        - res: dict of one evaluation metric name and its value
    """

    predictions, labels = eval_pred

    res = mean_absolute_error(labels.flatten(), predictions.flatten())
    return {"mae": res}

def show_one_image(dataset):
    """
    Show one random image from the dataset
    """

    go_back = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    ])

    i = np.random.randint(0, len(dataset))
    input = dataset[i]
    if hasattr(dataset, 'db'):
        print(f'name: {dataset.db["name"][i]}')
        print(f'age: {dataset.db["age"][i]}')
    print(f'label: {input["labels"]}')
    plt.imshow(go_back(input['pixel_values']).permute(1, 2, 0))
    plt.show()

def predict(model, dataset):
    """
    Get predictions and labels for the dataset

    Args:
        - model: the model to predict with
        - dataset: the dataset to predict on
    
    Returns:
        - preds: numpy array of predictions
        - labels: numpy array of labels
    """

    loader = DataLoader(dataset, batch_size=100, num_workers=4)
    labels, preds = [], []

    model.eval()
    model.to(device)

    for inputs in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        labels.append(inputs["labels"].detach().cpu())
        preds.append(outputs.logits.detach().cpu())
    
    labels = torch.cat(labels, dim=0).squeeze().numpy()
    preds = torch.cat(preds, dim=0).squeeze().numpy()

    return preds, labels

def predict_multiple(dataset, checkpoint_names):
    """
    Get predictions and labels for the dataset from multiple models

    Args:
        - dataset: the dataset to predict on
        - checkpoint_names: list of checkpoint names of the models to predict with
    
    Returns:
        - preds: numpy array of predictions, shape (num_samples, num_models)
        - labels: numpy array of labels, shape (num_samples,)
    """

    preds = []
    for name in tqdm(checkpoint_names):
        model = get_model(name)
        tmp_preds, labels = predict(model, dataset)
        preds.append(tmp_preds)

    # get per sample predictions for all models
    preds = np.stack(preds, axis=1)

    return preds, labels

def validation(model, datasets):
    """
    Evaluate the model on the subject inclusive and exclusive validation sets

    Args:
        - model: the model to evaluate
        - datasets: dict of dataset to evaluate on
    
    Returns:
        - res: dict of evaluation metric on each dataset
    """

    res = {}
    for dataset_name, dataset in datasets.items():

        preds, labels = predict(model, dataset)
        
        metric = compute_metrics((preds, labels))
        metric_name = list(metric.keys())[0]
        res[dataset_name + '_' + metric_name] = metric[metric_name]

    return res

def validation_multiple(checkpoint_names, datasets):
    """
    Evaluate multiple models on multiple datasets

    Args:
        - checkpoint_names: list of checkpoint names of the models to evaluate
        - datasets: dict of dataset to evaluate on
    
    Returns:
        - res: dict of evaluation metric on each dataset
    """

    res = {}
    for name in tqdm(checkpoint_names):
        model = get_model(name)
        tmp_res = validation(model, datasets)
        res[name] = tmp_res

    return res

def plot_eval_results(eval_results):
    """
    Plot the evaluation results from the multiple runs.

    Args:
        - eval_results: dict of evaluation results and their corresponding run names
    """

    metrics = list(zip(*[tuple(res.values()) for res in eval_results.values()]))
    metrics = list(map(np.array, metrics))
    metric_names = list(list(eval_results.values())[0].keys())
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    colors = {'retain_mae': 'b', 'SI_val_mae': 'g', 'SE_val_mae': 'r', 'forget_mae': 'c'}

    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        axes[i].hist(metric, label=metric_name, bins=30, color=colors[metric_name])
        axes[i].set_xlabel(metric_name)
        axes[i].set_ylabel('number of runs')
        axes[i].legend()
        axes[i].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.show()

def train(model, train_dataset, epochs, lr=1e-4, eval_datasets=None, output_dir="my_model", save_model=False, do_logging=False, batch_size=None):
    """
    Set up huggingface trainer and train the model

    Args:
        - model: the model to train
        - train_dataset: the dataset to train on
        - epochs: number of epochs to train
        - eval_datasets: dict of datasets to evaluate on
        - output_dir: directory to save the model
        - save_model: whether to save the model
        - do_logging: whether to log the training process
    """

    training_args = TrainingArguments(
        output_dir=output_dir,

        learning_rate=lr,
        # adam_beta1=0.9,
        # adam_beta2=0.999,
        # weight_decay=0.0,
        lr_scheduler_type="constant",
        # warmup_ratio=0.01,

        per_device_train_batch_size=batch_size if batch_size else 100,
        per_device_eval_batch_size=batch_size if batch_size else 100,
        num_train_epochs=epochs,

        save_strategy="epoch" if save_model else "no",
        save_total_limit=1,
        evaluation_strategy="epoch" if eval_datasets else "no",

        logging_strategy="steps" if do_logging else "no",
        logging_steps=10,
        report_to=["wandb"],
        run_name=output_dir,

        dataloader_num_workers=4,
        dataloader_drop_last=True,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=False)

def get_epsilon(fpr, fnr, delta=0.):
    """
    Get the epsilon from fpr and fnr calculated over one sample
    """

    # Use the following to avoid runtime warnings when calculating log of 0 or negative numbers
    # import warnings
    # warnings.filterwarnings("ignore", category=RuntimeWarning)

    # if perfectly separable return infinity
    if ((fpr == 0.) & (fnr == 0.)).any():
        return np.inf
    
    # remove points where fpr or fnr are 0 but not both
    mask = (fpr == 0.) ^ (fnr == 0.)
    fpr = fpr[~mask]
    fnr = fnr[~mask]

    # get epsilons
    eps_1 = np.log(1 - delta - fpr) - np.log(fnr)
    eps_2 = np.log(1 - delta - fnr) - np.log(fpr)

    # concatenate epsilons
    eps = np.concatenate([eps_1, eps_2])

    # return biggest epsilon
    if len(eps) == 0:
        return np.nan
    else:
        return np.nanmax(eps)

def forget_quality(preds_retrain, preds_unlearn, labels, delta=0.):
    """
    Compute the forget quality of the unlearned models compared to retrained models.

    For now it is based on the mean_auc_score of the roc_curve of the per forget_set_sample square loss

    Args:
        - preds_retrain: predictions of retrained models for each sample, has shape (num_samples, num_models)
        - preds_unlearn: predictions of unlearned models for each sample, has shape (num_samples, num_models)
        - labels: the labels of the forget set, has shape (num_samples,)
        - delta: the delta for the epsilon calculation
    
    Returns:
        - forget_quality: float in the range of [0.5, 1.0] where 0.5 is the best
    """

    # get per sample square loss for each retrained and unlearned model
    loss_retrain = np.power(preds_retrain - np.expand_dims(labels, axis=1), 2)
    loss_unlearn = np.power(preds_unlearn - np.expand_dims(labels, axis=1), 2)

    # concatenate losses for each sample, first retrained then unlearned or vice 
    # versa depending on biggest median of the two. SAMPLE ORDER NOT PRESERVED
    mask = np.median(loss_retrain, axis=1) > np.median(loss_unlearn, axis=1)
    losses = np.concatenate([
        np.concatenate([loss_unlearn[mask], loss_retrain[mask]], axis=1),
        np.concatenate([loss_retrain[~mask], loss_unlearn[~mask]], axis=1)
    ], axis=0)

    # labels for roc curve, differentiates between losses of retrained and unlearned models
    y_true = np.concatenate([np.zeros(loss_unlearn.shape), np.ones(loss_retrain.shape), ], axis=1).astype(np.int32)

    # compute fpr and fnr over losses of each sample 
    fpr, fnr = [], []
    for i in range(losses.shape[0]):
        fpr_tmp, fnr_tmp, _ = det_curve(y_true=y_true[i], y_score=losses[i])
        fpr.append(fpr_tmp)
        fnr.append(fnr_tmp)

    # get epsilon for each sample
    epsilons = list(map(get_epsilon, fpr, fnr, [delta]*len(fpr)))

    # aggregate by median and not mean because some epiolons are inf (when sample perfectly separable)
    # epsilons[epsilons == np.inf] = 6.5
    # return epsilons
    return np.nanmedian(epsilons)

def forget_quality_from_checkpoints(forget_set, retrained_checkpoint_names, unlearned_checkpoint_names):
    """
    Compute the forget quality of the unlearned models compared to retrained models.

    For now it is based on the mean_auc_score of the roc_curve of the per forget_set_sample square loss

    Args:
        - forget_set: the forget set dataset
        - retrained_checkpoint_names: list of checkpoint names of the retrained models
        - unlearned_checkpoint_names: list of checkpoint names of the unlearned models
    
    Returns:
        - forget_quality: float in the range of [0.5, 1.0] where 0.5 is the best
    """

    # get per sample predictions of retrained and unlearned models
    preds_retrain, _ = predict_multiple(forget_set, retrained_checkpoint_names)
    preds_unlearn, labels = predict_multiple(forget_set, unlearned_checkpoint_names)

    return forget_quality(preds_retrain, preds_unlearn, labels)

def unlearn(method, num_runs, args={}):
        
    method_name = method.__name__
    output_dir = f'checkpoints/{method_name}/'

    preds = []
    eval_results = {}
    for i in tqdm(range(num_runs)):
        
        args['retain_set'] = retain_set
        args['forget_set'] = forget_set
        args['SI_val'] = SI_val
        args['output_dir'] = output_dir + f'{i+1:03d}'

        model, eval_metric = method(**args)

        # model, eval_metric = method(
        #     retain_set, forget_set, SI_val, output_dir + f'{i+1:03d}',
        # )

        print(eval_metric)

        eval_results['run' + '_' + f'{i+1:03d}'] = eval_metric

        tmp_preds, _ = predict(model, forget_set)
        preds.append(tmp_preds)
    
    for res in eval_results.values():
        for k in res:
            res[k] = float(res[k])
    with open(output_dir + 'eval_results.json', 'w') as f:
        json.dump(eval_results, f)
    
    preds = np.stack(preds, axis=1)
    np.save(f'predictions/preds_{method_name}.npy', preds)

    return eval_results, preds


results = {}


# load datasets
retain_set = AgeDBDataset()
SI_val = AgeDBDataset(start_ratio=8/9, end_ratio=1)
forget_set = AgeDBForgetDataset()

# get checkpoint names
retrained_checkpoint_names = [RETRAINED_CHECKPOINTS + path for path in os.listdir(RETRAINED_CHECKPOINTS) if 'AgeDB' in path]
retrained_checkpoint_names = list(map(lambda p: p + '/' + os.listdir(p)[-1], retrained_checkpoint_names))
to_unlearn_checkpoint_name = RETRAINED_CHECKPOINTS + 'AgeDB_to_unlearn/'
to_unlearn_checkpoint_name += os.listdir(to_unlearn_checkpoint_name)[-1]

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"running on: {device}")


# baseline
preds_retrain , labels = predict_multiple(forget_set, retrained_checkpoint_names[:NUM_RUNS])
preds_retrain_new, _ = predict_multiple(forget_set, retrained_checkpoint_names[:NUM_RUNS])

eval_results_retrain = validation_multiple(retrained_checkpoint_names[:NUM_RUNS], {
    'SI_val': NoTransformDataset(SI_val),
    'forget': NoTransformDataset(forget_set),
})

SI_val_mae, forget_mae = (zip(*[(v['SI_val_mae'],v['forget_mae']) for v in eval_results_retrain.values()]))
results['baseline'] = {
        'epsilon': forget_quality(preds_retrain, preds_retrain_new, labels, delta=0.05),
        'S': np.array(forget_mae).mean(),
        'test': np.array(SI_val_mae).mean(), 
    }


# finetune
def unlearn_finetune(retain_set, forget_set, SI_val, output_dir, lr=0.001, epochs=3):    

    model = get_model(to_unlearn_checkpoint_name)

    args = {
        'model': model,
        'train_dataset': retain_set,
        'epochs': epochs,
        'output_dir': output_dir,
        'lr': lr,
    }

    train(**args)

    eval_metric = validation(model, {
        # 'retain': retain_set,
        'SI_val': SI_val,
        'forget': forget_set,
    })

    return model, eval_metric

n = NUM_RUNS
eval_results_unlearn_finetune, preds_unlearn_finetune = unlearn(unlearn_finetune, n, args={'lr': 0.001, 'epochs': 5})

SI_val_mae, forget_mae = (zip(*[(v['SI_val_mae'],v['forget_mae']) for v in eval_results_unlearn_finetune.values()]))
results['finetune'] = {
        'epsilon': forget_quality(preds_retrain, preds_unlearn_finetune, labels, delta=0.05),
        'S': np.array(forget_mae).mean(),
        'test': np.array(SI_val_mae).mean(),
    }


# label poisoning
class PoisonDataset(Dataset):

    def __init__(self, forget_set):
        self.forget_set = forget_set
        # self.random = np.random.RandomState(0)
        self.max_indx = self.forget_set.__len__()
        

    def __len__(self):
        return self.max_indx
    
    def __getitem__(self, idx):

        item = self.forget_set.__getitem__(idx)
        # random_inx = self.random.randint(0, self.max_indx)
        # item['labels'] = self.forget_set.labels[random_inx]
        noise = np.random.normal(loc=0, scale=30, size=1)[0]
        item['labels'] += noise
        
        return item
    
def unlearn_poison(retain_set, forget_set, SI_val, output_dir, lr=0.0007, epochs=1):    

    model = get_model(to_unlearn_checkpoint_name)

    args = {
        'model': model,
        'train_dataset': PoisonDataset(forget_set),
        'epochs': epochs,
        'output_dir': output_dir,
        'lr': lr,
        'batch_size': 88,
    }

    train(**args)

    eval_metric = validation(model, {
        # 'retain': retain_set,
        'SI_val': SI_val,
        'forget': forget_set,
    })

    return model, eval_metric

n = NUM_RUNS
eval_results_unlearn_poison, preds_unlearn_poison = unlearn(unlearn_poison, n, args={'lr': 0.0007, 'epochs': 2})
SI_val_mae, forget_mae = (zip(*[(v['SI_val_mae'],v['forget_mae']) for v in eval_results_unlearn_poison.values()]))
results['poison'] = {
        'epsilon': forget_quality(preds_retrain, preds_unlearn_poison, labels, delta=0.05),
        'S': np.array(forget_mae).mean(),
        'test': np.array(SI_val_mae).mean()
    }


# hybrid
def unlearn_hybrid(retain_set, forget_set, SI_val, output_dir, lr_poison=0.001, epochs_poison=1, lr_finetune=0.001, epochs_finetune=1):    

    model = get_model(to_unlearn_checkpoint_name)

    args = {
        'model': model,
        'train_dataset': PoisonDataset(forget_set),
        'epochs': epochs_poison,
        'output_dir': output_dir,
        'lr': lr_poison,
        'batch_size': 88,
    }

    train(**args)

    args = {
        'model': model,
        'train_dataset': retain_set,
        'epochs': epochs_finetune,
        'output_dir': output_dir,
        'lr': lr_finetune,
    }

    train(**args)

    eval_metric = validation(model, {
        'retain': retain_set,
        'SI_val': SI_val,
        'forget': forget_set,
    })

    return model, eval_metric

n = NUM_RUNS
eval_results_unlearn_hybrid, preds_unlearn_hybrid = unlearn(unlearn_hybrid, n, args={
    'lr_poison': 0.01, 'epochs_poison': 1, 'lr_finetune': 0.001, 'epochs_finetune': 1
})
SI_val_mae, forget_mae = (zip(*[(v['SI_val_mae'],v['forget_mae']) for v in eval_results_unlearn_hybrid.values()]))
results['hybrid'] ={
        'epsilon': forget_quality(preds_retrain, preds_unlearn_hybrid, labels, delta=0.05),
        'S': np.array(forget_mae).mean(),
        'test': np.array(SI_val_mae).mean(), 
    }


# store results
for res in results.values():
    for k in res:
        res[k] = float(res[k])
with open('results_AgeDB.json', 'w') as f:
    json.dump(results, f)


print(results)