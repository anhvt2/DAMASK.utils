#!/usr/bin/env python3

# postResults single_phase_equiaxed_tension.spectralOut --cr f,p
# filterTable < single_phase_equiaxed_tension.txt --white inc,1_f,1_p > stress_strain.log
# python3 plotStressStrain.py --stress_strain_file "stress_strain.log"

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

PARSER = argparse.ArgumentParser(description='')
PARSER.add_argument(
    "-stress_strain_file", "--stress_strain_file", default='stress_strain.log', type=str
)

PARSER.add_argument("-LoadFile", "--LoadFile", default='tension.load', type=str)

PARSER.add_argument("-OptSaveFig", "--OptSaveFig", type=bool, default=False)
PARSER.add_argument("-PlotTitle", "--PlotTitle", default='', type=str, required=False)


ARGS = PARSER.parse_args()
STRESS_STRAIN_FILE = ARGS.stress_strain_file
PLOT_TITLE = ARGS.PlotTitle


def _get_meta_info(stress_strain_file):
    """
    return
    (1) number of lines for headers
    (2) list of outputs for pandas dataframe
    """
    with open(stress_strain_file) as file_handler:
        txt_in_stress_strain_file = file_handler.readlines()

    try:
        num_lines_header = int(txt_in_stress_strain_file[0].split('\t')[0])
        fields_list = txt_in_stress_strain_file[num_lines_header].split('\t')
    except:
        num_lines_header = int(txt_in_stress_strain_file[0].split(' ')[0])
        fields_list = txt_in_stress_strain_file[num_lines_header].split(' ')
        fields_list = list(filter(''.__ne__, fields_list))
        print('%s is not natural - i.e. it may have been copied/pasted.' % stress_strain_file)
    else:
        print('Reading results in %s...' % stress_strain_file)
    for i in range(len(fields_list)):
        fields_list[i] = fields_list[i].replace('\n', '')
    print('numLinesHeader = ', num_lines_header)
    print('fieldsList = ', fields_list)
    return (num_lines_header, fields_list)


def _get_true_stress_strain(stress_strain_file):
    (numLinesHeader, fieldsList) = _get_meta_info(stress_strain_file)
    d = np.loadtxt(stress_strain_file, skiprows=numLinesHeader + 1)
    df = pd.DataFrame(d, columns=fieldsList)
    vareps = list(df['Mises(ln(V))'])
    sigma = list(df['Mises(Cauchy)'])
    (_, uniq_idx) = np.unique(np.array(vareps), return_index=True)
    vareps = np.array(vareps)[uniq_idx]
    sigma = np.array(sigma)[uniq_idx]
    x = vareps
    y = sigma / 1000000.0
    return (x, y)


def _get_interp_stress_strain(stress_strain_file):
    (x, y) = _get_true_stress_strain(stress_strain_file)
    interp_x = np.linspace(x.min(), x.max(), num=100)
    spline_interp = PchipInterpolator(x, y, extrapolate=True)
    spline_interp(interp_x)
    return (interp_x, rc_paramsy)


FIG = plt.figure()
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

AX = FIG.add_subplot(111)
X, Y = _get_true_stress_strain(STRESS_STRAIN_FILE)
AX.plot(X, Y, c='b', marker='o', linestyle='--', markersize=6)

INTERP_X, INTERP_Y = _get_interp_stress_strain(STRESS_STRAIN_FILE)
AX.plot(INTERP_X, INTERP_Y, c='r', marker='^', linestyle=':', markersize=6)
plt.legend(['true', 'cubic'], fontsize=24, frameon=False, markerscale=3)


plt.xlabel(r'$\varepsilon$ [-]', fontsize=30)
plt.ylabel(r'$\sigma$ [MPa]', fontsize=30)

if np.all(Y * 1e6 > -1e-5):
    plt.ylim(bottom=0)

if np.all(X > -1e-5):
    plt.xlim(left=0)

AX.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.4f'))

PARENT_FOLDER_NAME = os.getcwd().split('/')[-4:-1]

if PLOT_TITLE == '':
    plt.title('%s' % PARENT_FOLDER_NAME, fontsize=24)
else:
    plt.title(PLOT_TITLE, fontsize=24)

plt.show()
