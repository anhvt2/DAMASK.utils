
# plot density flow of $\pi^{obs}(\phi)$ and $\pi^{obs}(\sigma)$ as a function of strain $\varepsilon$

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os, sys, glob, datetime
import pandas as pd
from scipy.stats import gaussian_kde
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
    # else:
    #     print('Reading results in %s...' % (StressStrainFile))
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
    interp_x = np.linspace(0.0, 0.025, num=1000)
    # interp_x = np.linspace(x.min(), x.max(), num=100)
    # splineInterp = interp1d(x, y, kind='cubic', fill_value='extrapolate')
    splineInterp = PchipInterpolator(x, y, extrapolate=True)
    interp_y = splineInterp(interp_x)
    return interp_x, interp_y


for mltype, label in zip(['test/test-run-437-', 'test/test-run-180-', 'test/test-run-90-', 'test/test-run-20-'], ['D', 'C', 'B', 'A']):
    # initialize and collect data
    sigmaData = []
    for folderName in glob.glob(f'{mltype}*/'):
        StressStrainFile = folderName + '/' + 'stress_strain.log'
        x, y = getTrueStressStrain(StressStrainFile)
        interp_x, interp_y = getInterpStressStrain(StressStrainFile)
        sigmaData += [list(interp_y)]
    vareps_s = interp_x.copy()
    # calculate KDE
    sigma_s  = np.linspace(0, 100, num=500)
    sigmaData = np.array(sigmaData).T # (n_linspace strain, n_obs stress)
    # print(sigmaData.shape)  # debug
    density_matrix = np.zeros((len(sigma_s), len(vareps_s)))
    for i, vareps in enumerate(vareps_s):
        kde = gaussian_kde(sigmaData[i,:])
        density_matrix[:,i] = kde(sigma_s)
        density_matrix[:,i] /= np.trapz(density_matrix[:,i], sigma_s)
    # plot
    fig = plt.figure(num=None, figsize=(14, 12), dpi=300, facecolor='w', edgecolor='k')
    mpl.rcParams['xtick.labelsize'] = 24
    mpl.rcParams['ytick.labelsize'] = 24
    vmin, vmax = 0, np.max(density_matrix)
    img = plt.imshow(density_matrix, aspect='auto', origin='lower',
           extent=[vareps_s.min(), vareps_s.max(), sigma_s.min(), sigma_s.max()], cmap='Reds', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(img)
    cbar.set_label("density", fontsize=24)
    cbar.ax.tick_params(labelsize=24)
    plt.xlabel(r"$\varepsilon$ [-]", fontsize=24)
    plt.ylabel(r"$\sigma$", fontsize=24)
    plt.title(r'Cluster %s: Evolution of pdf($\sigma$) density over $\varepsilon$' % label, fontsize=24)
    plt.savefig(f'StressFlow_{label}.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
    plt.clf()
    plt.close()

