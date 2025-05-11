#!/usr/bin/env python3

# postResults single_phase_equiaxed_tension.spectralOut --cr f,p
# filterTable < single_phase_equiaxed_tension.txt --white inc,1_f,1_p > stress_strain.log
# python3 plotStressStrain.py --stress_strain_file "stress_strain.log"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import datetime
import argparse
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator

parser = argparse.ArgumentParser(description='')
parser.add_argument("-stress_strain_file", "--stress_strain_file",
                    default='stress_strain.log', type=str)
parser.add_argument("-LoadFile", "--LoadFile",
                    default='tension.load', type=str)
parser.add_argument("-OptSaveFig", "--OptSaveFig", type=bool, default=False)
parser.add_argument("-PlotTitle", "--PlotTitle",
                    default='', type=str, required=False)
# parser.add_argument("-skiprows", "--skiprows", type=int, default=4) # deprecated

args = parser.parse_args()
stress_strain_file = args.stress_strain_file
LoadFile = args.LoadFile
OptSaveFig = args.OptSaveFig
PlotTitle = args.PlotTitle

# skiprows = args.skiprows # deprecated


def getMetaInfo(stress_strain_file):
    """
    return 
    (1) number of lines for headers 
    (2) list of outputs for pandas dataframe
    """
    file_handler = open(stress_strain_file)
    txt_in_stress_strain_file = file_handler.readlines()
    file_handler.close()
    try:
        num_lines_header = int(txt_in_stress_strain_file[0].split('\t')[0])
        fields_list = txt_in_stress_strain_file[num_lines_header].split('\t')
    except:
        num_lines_header = int(txt_in_stress_strain_file[0].split(' ')[0])
        fields_list = txt_in_stress_strain_file[num_lines_header].split(' ')
        fields_list = list(filter(('').__ne__, fields_list))  # remove all ''
        print('%s is not natural - i.e. it may have been copied/pasted.' %
              (stress_strain_file))
    else:
        print('Reading results in %s...' % (stress_strain_file))
    for i in range(len(fields_list)):
        fields_list[i] = fields_list[i].replace('\n', '')
    print('numLinesHeader = ', num_lines_header)
    print('fieldsList = ', fields_list)
    return num_lines_header, fields_list


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


def get_true_stress_strain(stress_strain_file):
    # d = np.loadtxt(stress_strain_file, skiprows=4)
    numLinesHeader, fieldsList = getMetaInfo(stress_strain_file)
    # d = np.loadtxt(stress_strain_file, skiprows=skiprows)
    d = np.loadtxt(stress_strain_file, skiprows=numLinesHeader+1)
    # df = pd.DataFrame(d, columns=['inc','elem','node','ip','grain','1_pos','2_pos','3_pos','1_f','2_f','3_f','4_f','5_f','6_f','7_f','8_f','9_f','1_p','2_p','3_p','4_p','5_p','6_p','7_p','8_p','9_p'])
    df = pd.DataFrame(d, columns=fieldsList)
    # vareps = [1] + list(df['1_f']) # d[:,1]  # strain -- pad original
    # sigma  = [0] + list(df['1_p']) # d[:,2]  # stress -- pad original
    vareps = list(df['Mises(ln(V))'])  # strain -- pad original
    sigma = list(df['Mises(Cauchy)'])  # stress -- pad original
    _, uniq_idx = np.unique(np.array(vareps), return_index=True)
    vareps = np.array(vareps)[uniq_idx]
    sigma = np.array(sigma)[uniq_idx]
    # x = (vareps - 1)
    x = (vareps)
    y = sigma / 1e6
    return x, y


def get_interp_stress_strain(stress_strain_file):
    x, y = get_true_stress_strain(stress_strain_file)
    interp_x = np.linspace(x.min(), x.max(), num=100)
    # splineInterp = interp1d(x, y, kind='cubic', fill_value='extrapolate')
    splineInterp = PchipInterpolator(x, y, extrapolate=True)
    interp_y = splineInterp(interp_x)
    return interp_x,rc_paramsy


fig = plt.figure()
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

ax = fig.add_subplot(111)
x, y = get_true_stress_strain(stress_strain_file)
ax.plot(x, y, c='b', marker='o', linestyle='--', markersize=6)

interp_x, interp_y = get_interp_stress_strain(stress_strain_file)
ax.plot(interp_x, interp_y, c='r', marker='^', linestyle=':', markersize=6)
plt.legend(['true', 'cubic'], fontsize=24, frameon=False, markerscale=3)


plt.xlabel(r'$\varepsilon$ [-]', fontsize=30)
plt.ylabel(r'$\sigma$ [MPa]', fontsize=30)

if np.all(y * 1e6 > -1e-5):
    plt.ylim(bottom=0)

if np.all(x > -1e-5):
    plt.xlim(left=0)

ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.4f'))

parentFolderName = os.getcwd().split('/')[-4:-1]

if PlotTitle == '':
    plt.title('%s' % parentFolderName, fontsize=24)
else:
    plt.title(PlotTitle, fontsize=24)

plt.show()
