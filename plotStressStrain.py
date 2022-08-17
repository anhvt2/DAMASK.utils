
# postResults single_phase_equiaxed_tension.spectralOut --cr f,p
# filterTable < single_phase_equiaxed_tension.txt --white inc,1_f,1_p > stress_strain.log
# python3 plotStressStrain.py --StressStrainFile "stress_strain.log"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys, datetime
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='')
parser.add_argument("-StressStrainFile", "--StressStrainFile", default='stress_strain.log', type=str)
parser.add_argument("-LoadFile", "--LoadFile", default='tension.load', type=str)
parser.add_argument("-optSaveFig", "--optSaveFig", type=bool, default=False)
parser.add_argument("-skiprows", "--skiprows", type=int, default=4)
args = parser.parse_args()
StressStrainFile = args.StressStrainFile
LoadFile = args.LoadFile
skiprows = args.skiprows

def readLoadFile(LoadFile):
	load_data = np.loadtxt(LoadFile, dtype=str)
	n_fields = len(load_data)
	# assume uniaxial:
	for i in range(n_fields):
		if load_data[i] == 'Fdot' or load_data[i] == 'fdot':
			print('Found *Fdot*!')
			Fdot11 = float(load_data[i+1])
		if load_data[i] == 'time':
			print('Found *totalTime*!')
			totalTime = float(load_data[i+1])
		if load_data[i] == 'incs':
			print('Found *totalIncrement*!')
			totalIncrement = float(load_data[i+1])
		if load_data[i] == 'freq':
			print('Found *freq*!')
			freq = float(load_data[i+1])
	return Fdot11, totalTime, totalIncrement

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

# d = np.loadtxt(StressStrainFile, skiprows=4)
d = np.loadtxt(StressStrainFile, skiprows=skiprows)
df = pd.DataFrame(d, columns=['inc','elem','node','ip','grain','1_pos','2_pos','3_pos','1_f','2_f','3_f','4_f','5_f','6_f','7_f','8_f','9_f','1_p','2_p','3_p','4_p','5_p','6_p','7_p','8_p','9_p'])
vareps = [1] + list(df['1_f']) # d[:,1] # strain -- pad original
sigma  = [0] + list(df['1_p']) # d[:,2] # stress -- pad original
_, uniq_idx = np.unique(np.array(vareps), return_index=True)
vareps = np.array(vareps)[uniq_idx]
sigma  = np.array(sigma)[uniq_idx]

fig = plt.figure()
ax = fig.add_subplot(111)


x = (vareps - 1)
y = sigma / 1e6
ax.plot(x, y, c='b', marker='o', linestyle=':', markersize=6)

from scipy.interpolate import interp1d
splineInterp = interp1d(x, y, kind='cubic', fill_value='extrapolate')
ax.plot(x, splineInterp(x), c='r', marker='^', linestyle='-', markersize=6)
plt.legend(['true', 'cubic'])


plt.xlabel(r'$\varepsilon$ [-]', fontsize=30)
plt.ylabel(r'$\sigma$ [MPa]', fontsize=30)

if np.all(sigma > -1e-5):
	plt.ylim(bottom=0)

if np.all((vareps - 1) > -1e-5):
	plt.xlim(left=0)

ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.4f'))

parentFolderName = os.getcwd().split('/')[-4:-1]
plt.title('%s' % parentFolderName, fontsize=24)

plt.show()

