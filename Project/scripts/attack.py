import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import det_curve

def loss(logits):
    """Gets logits and computes the hinge loss

    Args:
        logits (np.array): logits

    Returns:
        z - mz: the hinge loss computed from the logits
    """ 

    S, M, L = logits.shape

    # Labels are stored in the first element of each sample
    labels = np.argmax(logits[:, 0], axis=1)    
    non_labels = np.ones((S, L), dtype=bool)
    non_labels[range(S), labels] = False    
    
    z = np.empty((S, M - 1))   
    mz = np.empty((S, M - 1))
    for i in range(1, M):
        z[:, i - 1] = logits[range(S), i, labels]        
        mz[:, i - 1] = np.max(logits[:, i][non_labels].reshape(-1, L - 1), axis=1)
    
    return z - mz

def likelihood(points, mean, std):
    """Gets points and computes the likelihood of them being sampled from a certain gaussian distribution

    Args:
        points (np.array): points 
        mean (float): the mean of the gaussian
        std (float): the standard deviation of the gaussian

    Returns:
        lh: array containing the likelihood of the points being in the distribution
    """
    S, M = points.shape
    lh = np.empty((S, M))

    for i in range(S):
        for j in range(M):
            lh[i, j] = norm.pdf(points[i, j], mean[i], std[i])
    
    return lh

def per_example_stat(p_r, p_u, compare):
    """Gets probabilities and computes either per example FPR or per example FNR based on the given function

    Args:
        p_r (np.array): likelihoods of points being from the retain distribution 
        p_u (np.array): likelihoods of points being from the unlearn distribution 
        compare (function): function that determines what is considered in the stat

    Returns:
        stat: the computed stat
    """

    S, M = p_r.shape

    stat = np.zeros(S)
    for i in range(S):
        for j in range(M):
            if compare(p_r[i, j], p_u[i, j]):
                stat[i] += 1
        stat[i] /= M

    return stat

def get_epsilon(fpr, fnr, delta=0.):
    EPS = 1e-8
    # if perfectly separable return infinity
    if ((fpr == 0.) & (fnr == 0.)).any():
        return np.inf
    
    # remove points where fpr or fnr are 0 but not both
    mask = (fpr == 0.) ^ (fnr == 0.)
    fpr = fpr[~mask]
    fnr = fnr[~mask]

    # get epsilons
    eps_1 = np.log(1 - delta - fpr + EPS) - np.log(fnr + EPS)
    eps_2 = np.log(1 - delta - fnr + EPS) - np.log(fpr + EPS)

    # concatenate epsilons
    eps = np.concatenate([eps_1, eps_2])
    
    if eps.shape[0] == 0:
        return np.nan

    # return biggest epsilon
    return np.nanmax(eps)

def attack(logits_retain, logits_unlearn, delta=0.0, name="", plot=True, score="gaussian"):

    labels = np.argmax(logits_retain[:, 0], axis=1)
    labels = labels.reshape(-1, 1).repeat(logits_retain.shape[1], axis=1)

    if score == "gaussian":
        # Compute Gaussian Stat (in this case Hinge Loss)
        l_retain = loss(logits_retain)
        l_unlearn = loss(logits_unlearn)

        retain_mean, retain_std = np.mean(l_retain, axis=1), np.std(l_retain, axis=1)
        unlearn_mean, unlearn_std = np.mean(l_unlearn, axis=1), np.std(l_unlearn, axis=1)

        l_retain = likelihood(l_retain, unlearn_mean, unlearn_std) / likelihood(l_retain, retain_mean, retain_std)
        l_unlearn = likelihood(l_unlearn, unlearn_mean, unlearn_std) / likelihood(l_unlearn, retain_mean, retain_std)

    elif score == "loss":
        # Just compute Cross Entropy Loss
        l_retain = torch.nn.functional.cross_entropy(torch.tensor(logits_retain).permute(0, 2, 1), torch.tensor(labels), reduction="none").numpy()
        l_unlearn = torch.nn.functional.cross_entropy(torch.tensor(logits_unlearn).permute(0, 2, 1), torch.tensor(labels), reduction="none").numpy()
    else:
        raise NotImplementedError()

    # Plot some examples
    if plot:
        r, c = 3, 3
        fig, ax = plt.subplots(r, c)
        for i in range(r):
            for j in range(c):
                ax[i, j].hist(l_retain[i * c + j], bins=20, alpha=0.5, label="retain")
                ax[i, j].hist(l_unlearn[i * c + j], bins=20, alpha=0.5, label="unlearn")
        plt.legend()

        plt.show()

    mask = np.median(l_retain, axis=1) > np.median(l_unlearn, axis=1)
    losses = np.concatenate([
        np.concatenate([l_unlearn[mask], l_retain[mask]], axis=1),
        np.concatenate([l_retain[~mask], l_unlearn[~mask]], axis=1)
    ], axis=0)
    y_true = np.concatenate([np.zeros(l_retain.shape), np.ones(l_unlearn.shape)], axis=1).astype(np.int32)

    fpr, fnr = [], []
    for i in range(losses.shape[0]):
        fpr_tmp, fnr_tmp, _ = det_curve(y_true=y_true[i], y_score=losses[i])

        fpr.append(fpr_tmp)
        fnr.append(fnr_tmp)

    eps = list(map(get_epsilon, fpr, fnr, [delta]*len(fpr)))

    return eps


if (len(sys.argv) != 3):
    print("Invalid number of arguments, usage: python3 attack.py <data_path> <approach_name>")
else:
    DATA_PATH = sys.argv[1]
    APPROACH = sys.argv[2]

    logits_retain = np.load(os.path.join(DATA_PATH, "logits_retain.npy"))
    logits_unlearn = np.load(os.path.join(DATA_PATH, APPROACH, "logits_unlearn.npy"))

    # Only Retain
    # logits_retain = np.load(os.path.join(DATA_PATH, "logits_retain.npy"))
    # logits_unlearn = np.hstack((logits_retain[:, 0, :].reshape(-1, 1, 10), logits_retain[:, 41:, :])) 
    # logits_retain = logits_retain[:, :41, :]

    eps = attack(logits_retain, logits_unlearn, delta=0.05, score="gaussian")

    print(f"eps median: {np.nanmedian(eps)}")
    print(f"eps mean: {np.nanmean(eps)}")
    print(f"eps CI (95%): {1.96 * np.nanstd(eps) / (len(eps) ** 0.5)}")

    metrics = pd.read_csv(os.path.join(DATA_PATH, APPROACH, "metrics.csv"))
    print(f"Retain Accuracy: {metrics['retain_accuracy'].mean()}, CI (95%): {1.96 * metrics['retain_accuracy'].std() / (metrics['retain_accuracy'].count() ** 0.5)}")
    print(f"Forget Accuracy: {metrics['forget_accuracy'].mean()}, CI (95%): {1.96 * metrics['forget_accuracy'].std() / (metrics['forget_accuracy'].count() ** 0.5)}")
    print(f"Test Accuracy: {metrics['test_accuracy'].mean()}, CI (95%): {1.96 * metrics['test_accuracy'].std() / (metrics['test_accuracy'].count() ** 0.5)}")