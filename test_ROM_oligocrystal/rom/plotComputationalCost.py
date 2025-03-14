
from natsort import natsorted, ns
import pyvista
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob, os, time
import numpy as np
from distutils.util import strtobool
import logging
import pandas as pd
from matplotlib.colors import Normalize, LogNorm
from scipy.interpolate import interpn
from sklearn.metrics import r2_score
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
cmap = plt.cm.get_cmap('coolwarm')

"""
extractData.py: finished in 1494.23 s.
computeBasis.py: Total time for POD basis: 838.64 s
computeCoefs.py: Elapsed time = 10713.857370376587 s.
extractRomData.py: Elapsed time: 976.0721864700317 s.
nn3d.py: Elapsed time: ?? s.
predictCoefs.py: Finish dumping local POD coefs in 190.77394485473633 s.
reconstructRomSolution.py: Total elapsed time: 5178.497043609619 s.
calculateFomRomError.py: Total elapsed time: 7869.626697301865 s.
"""

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

x = ['Offline\n Training', 'Online\n Prediction']
FomCost = 5 * 96 * 2 * 18 * (279 + 370)

CostOffline     = np.array([1494.23, 838.64, 10713.857370376587, 976.0721864700317, 3825.140154361725 * 6/10 * 36]) / 3600
LabelsOffline   = ['Build snapshot matrix',
                    'Compute POD basis',
                    'Compute POD coefs',
                    'Compile POD datasets',
                    'Train NN',
                    ] 

CostOnline      = np.array([190.77394485473633, 5178.497043609619]) / 3600
LabelsOnline    = ['Predict POD coefs',
                    'Reconstruct ROM solutions']

fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True, figsize=(8, 8), gridspec_kw={'height_ratios': [2, 1]})

# Define y-axis breakpoints
low_ylim = 0
low_ymax = 5   # Upper limit of lower plot
high_ymin = 15  # Lower limit of upper plot
high_ymax = 30  # Top limit

# Offline cost breakdown
print(f'\n')
for i, cost, label, color in zip(range(len(CostOffline)), CostOffline, LabelsOffline, colors[:len(CostOffline)]):
    ax1.bar(x[0], CostOffline[i], bottom=np.sum(CostOffline[:i]), color=color, label=label)
    print(f'{label}: {cost*3600} seconds')

# Online cost breakdown
for i, cost, label, color in zip(range(len(CostOnline)), CostOnline, LabelsOnline, colors[len(CostOffline):len(CostOffline)+len(CostOnline)]):
    ax1.bar(x[1], CostOnline[i], bottom=np.sum(CostOnline[:i]), color=color, label=label)
    print(f'{label}: {cost*3600} seconds')

print(f'\n')

ax1.set_ylim(low_ylim, low_ymax)

# Plot the bars on the top (dominant component: Train NN)
ax2.bar(x[0], CostOffline[-1], bottom=np.sum(CostOffline[:-1]), color="tab:purple", label="Train NN")
ax2.set_ylim(high_ymin, high_ymax)

# Hide spines between the two subplots
ax1.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

# Add diagonal lines for the break effect
d = 0.01  # Size of diagonal lines
kwargs = dict(transform=ax1.transAxes, color='black', clip_on=False)
ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

kwargs = dict(transform=ax2.transAxes, color='black', clip_on=False)
ax2.plot((-d, +d), (-d, +d), **kwargs)
ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)

# Labels and title
# ax1.set_ylabel(r'Time [CPU hr]', fontsize=18)
ax2.set_ylabel(r'Time [CPU hr]', fontsize=18)
ax2.set_title(r'Breakdown of Computational Cost for ROM', fontsize=16)

# Legend
ax1.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.05, 1.0), frameon=False)
ax2.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.05, 1.0), frameon=False)

# plt.legend([LabelsOffline + LabelsOnline], fontsize=24)
# plt.ylabel(r'Time [CPU hr]', fontsize=24)
# # plt.yscale('log')
# plt.title(r'Breakdown of computational cost for ROM', fontsize=24)
# plt.legend(fontsize=18, loc='upper left', bbox_to_anchor=(1.05, 1.0),frameon=False, markerscale=4)

plt.savefig('CostRom.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
plt.close()

print(f'Speedup factor: {FomCost / np.sum(np.hstack((CostOffline, CostOnline)))} x.')
print(f'Total Offline Training: {np.sum(CostOffline)} CPU hr.')
print(f'Total Online Prediction: {np.sum(CostOnline)} CPU hr.')
print(f'Total Online/Offline ROM: {np.sum(CostOffline) + np.sum(CostOnline)} CPU hr.')
