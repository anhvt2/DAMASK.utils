#!/usr/bin/env python3
import argparse


# postResults single_phase_equiaxed_tension.spectralOut --cr f,p
# filterTable < single_phase_equiaxed_tension.txt --white inc,1_f,1_p > stress_strain.log
# python3 plotStressStrain.py --stress_strain_file "stress_strain.log"

import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PARSER = argparse.ArgumentParser(description='')
PARSER.add_argument("-stress_strain_file", "--stress_strain_file", default=None, type=str)

PARSER.add_argument("-LoadFile", "--LoadFile", default='tension.load', type=str)

PARSER.add_argument("-optSaveFig", "--optSaveFig", type=bool, default=False)
ARGS = PARSER.parse_args()

print('Looking for .txt file output from DAMASK postResult.py!')
if len(glob.glob('*.txt')) > 0:
    stress_strain_file = np.loadtxt(glob.glob('*.txt')[0])
    print('Reading %s!' % glob.glob('*.txt')[0])
elif os.path.exists(ARGS.stress_strain_file):
    stress_strain_file = ARGS.stress_strain_file
    print('Reading %s!' % ARGS.stress_strain_file)
else:
    print('No stress_strain_file is found! Error!')


mpl.rc_params['xtick.labelsize'] = 24
mpl.rc_params['ytick.labelsize'] = 24

D = np.loadtxt(stress_strain_file, skiprows=4)
VAREPS = D[:, 1]  # strain
SIGMA = D[:, 2]  # stress

FIG = plt.figure()
AX = FIG.add_subplot(111)
AX.plot((VAREPS - 1) * 1e2, SIGMA / 1e6, c='b', marker='o', linestyle='-', markersize=6)

plt.xlabel(r'$\varepsilon$ [%]', fontsize=30)
plt.ylabel(r'$\sigma$ [MPa]', fontsize=30)

if np.all(SIGMA > -1e-5):
    plt.ylim(bottom=0)

if np.all((VAREPS - 1) > -1e-5):
    plt.xlim(left=0)

AX.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.4f'))

PARENT_FOLDER_NAME = os.getcwd().split('/')[-4:-1]
plt.title('%s' % PARENT_FOLDER_NAME, fontsize=24)

plt.show()
