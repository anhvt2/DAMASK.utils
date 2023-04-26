#!/usr/bin/env python
# coding: utf-8

# In[54]:


# import statements
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

from cycler import cycler
default_cycler = (cycler(color=['r', 'g', 'b', 'y']) +
                  cycler(linestyle=['-', '--', ':', '-.']))
custom_cycler = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']) +
                 cycler(lw=[1, 2, 3, 4, 5, 6, 7, 8]))
# plt.rc('axes', prop_cycle=default_cycler) # default color cycle
plt.rc('axes', prop_cycle=custom_cycler) # custom color cycle
# In[58]:


# function to plot the fractional costs
def plot_frac_cost(filename, label=None):
    frac_cost = pd.read_csv(filename, sep="\t")
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(frac_cost.shape[0])
    for col in frac_cost.columns:
        if col == "total":
            continue
        ax.bar(range(frac_cost.shape[0]), frac_cost[col], 0.5, bottom=bottom, label=r"$\ell$ = " + col)
        bottom += frac_cost[col].values
    ax.legend(bbox_to_anchor=(1, 1), frameon=False, fontsize=24)
    ax.set_xticks(range(frac_cost.shape[0]))
    ax.set_xticklabels(frac_cost["total"])
    ax.set_xlabel(r"Total Cost [CPU $\times$ hr]", fontsize=24)
    ax.set_ylabel("Percentage (%) of Total Cost", fontsize=24)
    ax.set_title(label, fontsize=24)
    ax.set_ylim(top=100)
    plt.subplots_adjust(left=0.1, right=0.85)
    plt.show()


# In[59]:


plot_frac_cost("mlmc_frac_cost.txt", label=r'Distribution of total cost in MLMC for Magnesium')







