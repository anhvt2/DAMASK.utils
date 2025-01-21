
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

