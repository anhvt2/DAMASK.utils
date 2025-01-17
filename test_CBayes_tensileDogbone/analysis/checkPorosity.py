
import numpy as np
from scipy.stats import norm # The standard Normal distribution
from scipy.stats import gaussian_kde as GKDE # A standard kernel density estimator
from natsort import natsorted, ns # natural-sort
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys, time, glob

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

testIdxs = [20,90,180,437]
folders = natsorted(glob.glob('*/'))

porosities = []
for folder in folders:
    tmp = np.loadtxt(f'{folder}/porosity.txt')
    porosities += [tmp]

porosities = np.array(porosities) # Global, Local, Target

fig = plt.figure(num=None, figsize=(20, 14), dpi=300, facecolor='w', edgecolor='k')
plt.plot(porosities[:,0], porosities[:,1], 'bo', ms=10)
plt.xlabel(r'global $\phi$', fontsize=24)
plt.ylabel(r'local (gauge) $\phi$', fontsize=24)
plt.title(r'QoI porosity $\phi$ measured globally and locally', fontsize=24)
# plt.axis('equal')
plt.xlim(left=0,right=0.06)
plt.ylim(bottom=0,top=0.06)
# plt.show()
plt.savefig('porosities.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
