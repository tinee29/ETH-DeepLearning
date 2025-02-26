import os
import sys
import pandas as pd
import numpy as np
from attack import attack
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# usage python3 plot_eps.py in the main directory (machine-unlearning) of the project

DATA_PATH = "./data/"
DELTAS = list(np.arange(0.0, 0.99, 0.01))
# get approaches name
approaches = next(os.walk(DATA_PATH))[1]

assert len(sys.argv) == 1, "Invalid number of arguments, usage: python3 plot_eps.py (in machine-unlearning dir)"

fig, ax = plt.subplots()

for approach in approaches:
    if approach != "baseline":
        logits_retain = np.load(os.path.join(DATA_PATH, "logits_retain.npy"))
        logits_unlearn = np.load(os.path.join(DATA_PATH, approach, "logits_unlearn.npy"))
    else:
        logits_retain = np.load(os.path.join(DATA_PATH, "logits_retain.npy"))
        logits_unlearn = np.hstack((logits_retain[:, 0, :].reshape(-1, 1, 10), logits_retain[:, 41:, :])) 
        logits_retain = logits_retain[:, :41, :]

    mean_deltas = []
    for delta in DELTAS:
    
        eps = attack(logits_retain, logits_unlearn, plot=False, name=approach, delta=delta, score="gaussian")
        mean_deltas.append(np.nanmedian(eps))

    plt.xlabel('Delta')
    plt.ylabel("Epsilon")
    plt.axhline(y=0, color='black', linestyle='--')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.1))  # x-axis ticks at multiples of 2
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.5))

    plt.plot(DELTAS, mean_deltas, label=approach.replace("_", " ").title())
    plt.legend(loc="upper right")
    print("PLOTTING APPROACH " + str(approach) + " PLOTTING DELTA " + str(mean_deltas))
plt.show()

