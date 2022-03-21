
# ../dakota -i textbook16d_uq_sc_pyImport.in > dakota.log
# grep -inr ' f1' dakota.log  > tmp.txt
# sed -i  's/ f1//g' tmp.txt

# adopt from ./testCBayes-Damask-Phase_Dislotwin_TWIP-Steel-FeMnC-64x64x64/plotQoI.py


import numpy as np
import glob, os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24 

from scipy.stats import norm # The standard Normal distribution
from scipy.stats import gaussian_kde as gaussian_kde # A standard kernel density estimator
d = np.loadtxt('tmp.txt', delimiter=':')[:,1]
q = gaussian_kde(d)

x = np.linspace(0, 0.075, 1000)
plt.plot(x, q(x), c='tab:blue', marker='o', linestyle='-')
plt.xlim([0, np.max(x)])
plt.ylim(bottom=0)
plt.show()


