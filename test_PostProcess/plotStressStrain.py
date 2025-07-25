
# postResults single_phase_equiaxed_tension.spectralOut --cr f,p
# filterTable < single_phase_equiaxed_tension.txt --white inc,1_f,1_p > stress_strain.log
# python3 plotStressStrain.py --file "stress_strain.log"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys, datetime
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("-file", "--file", default='stress_strain.log', type=str)
parser.add_argument("-optSaveFig", "--optSaveFig", type=bool, default=False)
args = parser.parse_args()
file = args.file

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

d = np.loadtxt(file, skiprows=7)
vareps = d[:,1] # strain
sigma  = d[:,2] # stress

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot((vareps - 1), sigma / 1e6, 'bo-', markersize=5)
plt.xlabel(r'$\varepsilon$ ', fontsize=24)
plt.ylabel(r'$\sigma$ [MPa]', fontsize=24)
plt.title(r'Equivalent $\sigma$-$\varepsilon$ curve', fontsize=24)

ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))

plt.show()

