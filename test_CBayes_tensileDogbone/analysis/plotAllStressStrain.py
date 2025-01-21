
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys, glob, datetime
import argparse
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator

def getMetaInfo(StressStrainFile):
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
    # print('numLinesHeader = ', numLinesHeader)
    # print('fieldsList = ', fieldsList)
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
    numLinesHeader, fieldsList = getMetaInfo(StressStrainFile)
    d = np.loadtxt(StressStrainFile, skiprows=numLinesHeader+1)
    df = pd.DataFrame(d, columns=fieldsList)
    vareps = list(df['Mises(ln(V))'])  # strain -- pad original
    sigma  = list(df['Mises(Cauchy)']) # stress -- pad original
    _, uniq_idx = np.unique(np.array(vareps), return_index=True)
    vareps = np.array(vareps)[uniq_idx]
    sigma  = np.array(sigma)[uniq_idx]
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
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
ax = fig.add_subplot(111)

# for mltype, color, alpha in zip(['train','test'], ['tab:orange','tab:green'], [0.50, 0.25]):
#     for folderName in glob.glob(f'{mltype}/*'):
#         StressStrainFile = folderName + '/' + 'stress_strain.log'
#         x, y = getTrueStressStrain(StressStrainFile)
#         interp_x, interp_y = getInterpStressStrain(StressStrainFile)
#         # ax.plot(x, y, marker='o', linestyle='--', markersize=6)
#         ax.plot(interp_x, interp_y, marker='o', color=color, alpha=alpha, linestyle=':', markersize=2)
#         # plt.legend(['true', 'cubic'], fontsize=24, frameon=False, markerscale=3)
#
# ax.plot([], [], marker='o', linestyle='-', markersize=2, color='tab:orange', label='train')
# ax.plot([], [], marker='o', linestyle='-', markersize=2, color='tab:green',  label='test')

for mltype, color, alpha in zip(['train/', 'test/test-run-437-', 'test/test-run-180-', 'test/test-run-90-', 'test/test-run-20-'], ['tab:gray', 'tab:red', 'tab:green', 'tab:orange', 'tab:blue'], [0.75, 0.75, 0.65, 0.55, 0.45]):
    for folderName in glob.glob(f'{mltype}*/'):
        StressStrainFile = folderName + '/' + 'stress_strain.log'
        x, y = getTrueStressStrain(StressStrainFile)
        interp_x, interp_y = getInterpStressStrain(StressStrainFile)
        # ax.plot(x, y, marker='o', linestyle='--', markersize=6)
        ax.plot(interp_x, interp_y, marker='o', color=color, alpha=alpha, linestyle=':', markersize=2)
        # plt.legend(['true', 'cubic'], fontsize=24, frameon=False, markerscale=3)

ax.plot([], [], marker='o', linestyle='-', markersize=2, color='tab:gray', label='train')
ax.plot([], [], marker='s', linestyle='-', markersize=2, color='tab:blue',  label=r'test: $\overline{\phi} = 0.11%$')
ax.plot([], [], marker='^', linestyle='-', markersize=2, color='tab:orange',  label=r'test: $\overline{\phi} = 0.84%$')
ax.plot([], [], marker='v', linestyle='-', markersize=2, color='tab:green',  label=r'test: $\overline{\phi} = 1.63%$')
ax.plot([], [], marker='D', linestyle='-', markersize=2, color='tab:red',  label=r'test: $\overline{\phi} = 4.14%$')

plt.legend(loc='best', fontsize=24, frameon=False, markerscale=8)
plt.xlabel(r'$\varepsilon$ [-]', fontsize=30)
plt.ylabel(r'$\sigma$ [MPa]', fontsize=30)
if np.all(y * 1e6 > -1e-5):
    plt.ylim(bottom=0, top=100)
if np.all(x > -1e-5):
    plt.xlim(left=0)
ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
plt.title(r'$\varepsilon-\sigma$', fontsize=24)
plt.savefig(f'StressStrainAll.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)

