
# postResults single_phase_equiaxed_tension.spectralOut --cr f,p
# filterTable < single_phase_equiaxed_tension.txt --white inc,1_f,1_p > stress_strain.log
# python3 plotStressStrain.py --file "stress_strain.log"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys, datetime, glob
import argparse

# parser = argparse.ArgumentParser(description='')
# parser.add_argument("-file", "--file", default='stress_strain.log', type=str)
# parser.add_argument("-optSaveFig", "--optSaveFig", type=bool, default=False)
# args = parser.parse_args()
# file = args.file

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

fig = plt.figure()
ax = fig.add_subplot(111)

fileList = glob.glob('stress_strain.sve-*.log')

th_stress = 1e16 # numerical threshold for stress -- physically plausible

for file in fileList:
	d = np.loadtxt(file, skiprows=7)
	vareps = d[:,1] # strain
	sigma  = d[:,2] # stress
	if np.max(sigma) < th_stress:
		# ax.plot((vareps - 1), sigma / 1e6, 'bo-', markersize=2)
		ax.plot((vareps - 1), sigma / 1e6, marker='o', markersize=2)

plt.xlabel(r'$\varepsilon$ ', fontsize=30)
plt.ylabel(r'$\sigma$ [MPa]', fontsize=30)
plt.title(r'32x32x32: level-1', fontsize=30)

ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.4f'))

plt.show()

