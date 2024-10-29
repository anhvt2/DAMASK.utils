
# postResults single_phase_equiaxed_tension.spectralOut --cr f,p
# filterTable < single_phase_equiaxed_tension.txt --white inc,1_f,1_p > stress_strain.log
# python3 plotStressStrain.py --StressStrainFile "stress_strain.log"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys, datetime
import argparse
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

def getMetaInfo(StressStrainFile):
    """
    return 
    (1) number of lines for headers 
    (2) list of outputs for pandas dataframe
    """
    fileHandler = open(StressStrainFile)
    txtInStressStrainFile = fileHandler.readlines()
    fileHandler.close()
    try:
        numLinesHeader = int(txtInStressStrainFile[0].split('\t')[0])
        fieldsList = txtInStressStrainFile[numLinesHeader].split('\t')
    except:
        numLinesHeader = int(txtInStressStrainFile[0].split(' ')[0])
        fieldsList = txtInStressStrainFile[numLinesHeader].split(' ')
        fieldsList = list(filter(('').__ne__, fieldsList)) # remove all ''
        print('%s is not natural - i.e. it may have been copied/pasted.' % (StressStrainFile))
    else:
        print('Reading results in %s...' % (StressStrainFile))
    for i in range(len(fieldsList)):
        fieldsList[i] = fieldsList[i].replace('\n', '')
    print('numLinesHeader = ', numLinesHeader)
    print('fieldsList = ', fieldsList)
    return numLinesHeader, fieldsList

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

def getTrueStressStrain(StressStrainFile):
    # d = np.loadtxt(StressStrainFile, skiprows=4)
    numLinesHeader, fieldsList = getMetaInfo(StressStrainFile)
    # d = np.loadtxt(StressStrainFile, skiprows=skiprows)
    d = np.loadtxt(StressStrainFile, skiprows=numLinesHeader+1)
    # df = pd.DataFrame(d, columns=['inc','elem','node','ip','grain','1_pos','2_pos','3_pos','1_f','2_f','3_f','4_f','5_f','6_f','7_f','8_f','9_f','1_p','2_p','3_p','4_p','5_p','6_p','7_p','8_p','9_p'])
    df = pd.DataFrame(d, columns=fieldsList)
    # vareps = [1] + list(df['1_f']) # d[:,1]  # strain -- pad original
    # sigma  = [0] + list(df['1_p']) # d[:,2]  # stress -- pad original
    vareps = list(df['Mises(ln(V))'])  # strain -- pad original
    sigma  = list(df['Mises(Cauchy)']) # stress -- pad original
    _, uniq_idx = np.unique(np.array(vareps), return_index=True)
    vareps = np.array(vareps)[uniq_idx]
    sigma  = np.array(sigma)[uniq_idx]
    # x = (vareps - 1)
    x = (vareps)
    y = sigma / 1e6
    return x, y

def getInterpStressStrain(StressStrainFile):
    x, y = getTrueStressStrain(StressStrainFile)
    interp_x = np.linspace(x.min(), x.max(), num=100)
    # splineInterp = interp1d(x, y, kind='cubic', fill_value='extrapolate')
    splineInterp = PchipInterpolator(x, y, extrapolate=True)
    interp_y = splineInterp(interp_x)
    return interp_x, interp_y

fig = plt.figure(num=None, figsize=(14, 12), dpi=300, facecolor='w', edgecolor='k')

trainIdx   = np.loadtxt('TrainIdx.dat')
testIdxOOD = np.loadtxt('TestIdxOOD.dat')
testIdxID  = np.loadtxt('TestIdxID.dat')

labels = ['train', 'test (OOD)', 'test (ID)']
colors = ['tab:blue', 'tab:orange', 'tab:green']

ax = fig.add_subplot(111)

for idx, label, color in zip([trainIdx, testIdxOOD, testIdxID], labels, colors):
    for i in idx:
        StressStrainFile = '../damask/%d/postProc/stress_strain.log' % i
        if os.path.exists(StressStrainFile):
            x, y = getTrueStressStrain(StressStrainFile)
            ax.plot(x, y, c=color, marker='o', linestyle='--', markersize=6, label=label)

plt.xlabel(r'$\varepsilon$ [-]', fontsize=30)
plt.ylabel(r'$\sigma$ [MPa]', fontsize=30)

if np.all(y * 1e6 > -1e-5):
    plt.ylim(bottom=0)

if np.all(x > -1e-5):
    plt.xlim(left=0)

ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.4f'))
plt.title(r'Variation of $\varepsilon-\sigma$ by train/test', fontsize=24)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=24, loc='upper left', bbox_to_anchor=(1.05, 1.0),frameon=True, markerscale=3)
plt.savefig('StressStrainTrainTest.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)

