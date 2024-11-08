
from natsort import natsorted, ns
import pyvista
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob, os, time
import numpy as np
import argparse
from distutils.util import strtobool
import logging
import pandas as pd
from matplotlib.colors import Normalize, LogNorm
from scipy.interpolate import interpn
from sklearn.metrics import r2_score
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
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

x = ['Offline Training', 'Online Prediction']

CostOffline     = np.array([1494.23, 838.64, 10713.857370376587, 976.0721864700317, 1000])
LabelsOffline   = ['Build snapshot matrix',
                    'Compute POD basis',
                    'Compute POD coefs',
                    'Compile POD datasets',
                    'Train NN',
                    ] 

CostOnline      = np.array([190.77394485473633, 5178.497043609619])
LabelsOnline    = ['Predict POD coefs',
                    'Reconstruct ROM solutions',
                    ] 
# Offline cost breakdown
for i, cost, label, color in zip(range(len(CostOffline)), CostOffline, LabelsOffline, colors[:len(CostOffline)]):
    plt.bar(x[0], CostOffline[i], bottom=np.sum(CostOffline[:i]), color=color, label=label)

plt.legend(LabelsOffline, fontsize=24)
plt.ylabel(r'time [CPU s]', fontsize=24)
plt.legend(fontsize=18, loc='upper left', bbox_to_anchor=(1.05, 1.0),frameon=False, markerscale=4)

plt.savefig('CostOffline.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
plt.close()

# Online cost breakdown
for i, cost, label, color in zip(range(len(CostOnline)), CostOnline, LabelsOnline, colors[:len(CostOnline)]):
    plt.bar(x[0], CostOnline[i], bottom=np.sum(CostOnline[:i]), color=color, label=label)

plt.legend(LabelsOnline, fontsize=24)
plt.ylabel(r'time [CPU s]', fontsize=24)
plt.legend(fontsize=18, loc='upper left', bbox_to_anchor=(1.05, 1.0),frameon=False, markerscale=4)

plt.savefig('CostOnline.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
plt.close()



