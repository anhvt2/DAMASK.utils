
# postResults single_phase_equiaxed_tension.spectralOut --cr f,p
# filterTable < single_phase_equiaxed_tension.txt --white inc,1_f,1_p > stress_strain.log
# python3 plotStressStrain.py --StressStrainFile "stress_strain.log"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import datetime
import argparse

parser = argparse.ArgumentParser(description='')
# parser.add_argument("-StressStrainFile", "--StressStrainFile", default='stress_strain.log', type=str)
parser.add_argument("-StressStrainFile",
                    "--StressStrainFile", default=None, type=str)
parser.add_argument("-LoadFile", "--LoadFile",
                    default='tension.load', type=str)
parser.add_argument("-optSaveFig", "--optSaveFig", type=bool, default=False)
args = parser.parse_args()

print('Looking for .txt file output from DAMASK postResult.py!')
if len(glob.glob('*.txt')) > 0:
    StressStrainFile = np.loadtxt(glob.glob('*.txt')[0])
    print('Reading %s!' % glob.glob('*.txt')[0])
elif os.path.exists(args.StressStrainFile):
    StressStrainFile = args.StressStrainFile
    print('Reading %s!' % args.StressStrainFile)
else:
    print('No StressStrainFile is found! Error!')

LoadFile = args.LoadFile


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

d = np.loadtxt(StressStrainFile, skiprows=4)
vareps = d[:, 1]  # strain
sigma = d[:, 2]  # stress

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot((vareps - 1) * 1e2, sigma / 1e6, c='b',
        marker='o', linestyle='-', markersize=6)
plt.xlabel(r'$\varepsilon$ [%]', fontsize=30)
plt.ylabel(r'$\sigma$ [MPa]', fontsize=30)

if np.all(sigma > -1e-5):
    plt.ylim(bottom=0)

if np.all((vareps - 1) > -1e-5):
    plt.xlim(left=0)

ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.4f'))

parentFolderName = os.getcwd().split('/')[-4:-1]
plt.title('%s' % parentFolderName, fontsize=24)

plt.show()
