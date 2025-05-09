#!/usr/bin/env python3

"""
    This script:
        (1) reads a snapshot (element-wise, not node-wise), for example, 'simple2d_tension_inc30.txt'
            from DAMASK, which has some headers and column labels
        (2) dumps *.npy (2d or 3d) of 'Mises(Cauchy)' and 'Mises(ln(V))' to 2 .npy files

    Adopted from read2panda.py
"""

import numpy as np
import pandas as pd
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-f", "--filename", help='.vtr file',
                    type=str, default='', required=True)
args = parser.parse_args()
filename = args.filename
# filename = 'main_tension_inc01.txt' # debug

fileHandler = open(filename)
txt = fileHandler.readlines()
fileHandler.close()

# Pre-process
numHeaderRows = int(txt[0].split('\t')[0])
oldHeader = txt[numHeaderRows].replace('\n', '').split('\t')
data = np.loadtxt(filename, skiprows=numHeaderRows+1)
df = pd.DataFrame(data, columns=oldHeader)

# Remove duplicate columns: https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
df = df.loc[:, ~df.columns.duplicated()].copy()
newHeader = list(df)

# Get 3d coordinates: follow geom2npy.py principles
Nx = int(df['1_pos'].max() - 0.5) + 1
Ny = int(df['2_pos'].max() - 0.5) + 1
Nz = int(df['3_pos'].max() - 0.5) + 1

x = np.array(df['1_pos'] - 0.5).astype(int).reshape(Nz, Ny, Nx).T
y = np.array(df['2_pos'] - 0.5).astype(int).reshape(Nz, Ny, Nx).T
z = np.array(df['3_pos'] - 0.5).astype(int).reshape(Nz, Ny, Nx).T

MisesCauchy = np.array(df['Mises(Cauchy)']).reshape(Nz, Ny, Nx).T
MisesLnV = np.array(df['Mises(ln(V))']).reshape(Nz, Ny, Nx).T

# Dump to .npy
np.save('MisesCauchy.npy', MisesCauchy)
np.save('MisesLnV.npy', MisesLnV)
