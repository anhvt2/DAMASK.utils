#!/usr/bin/env python
# coding: utf-8

# In[54]:


# import statements
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[58]:


# function to plot the fractional costs
def plot_frac_cost(filename):
    frac_cost = pd.read_csv(filename, sep="\t")
    fix, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(frac_cost.shape[0])
    for col in frac_cost.columns:
        if col == "total":
            continue
        ax.bar(range(frac_cost.shape[0]), frac_cost[col], 0.5, bottom=bottom, label=col)
        bottom += frac_cost[col].values
    ax.legend(bbox_to_anchor=(1, 1), frameon=False)
    ax.set_xticks(range(frac_cost.shape[0]))
    ax.set_xticklabels(frac_cost["total"])
    ax.set_xlabel("total cost")
    ax.set_ylabel("% of total cost")
    plt.show()


# In[59]:


plot_frac_cost("mlmc_frac_cost.txt")


# In[60]:


plot_frac_cost("mimc_frac_cost.txt")


# In[ ]:




