#!/usr/bin/env python3

"""
This script:
    (1) reads a snapshot (element-wise, not node-wise), for example, 'simple2d_tension_inc30.txt'
        from DAMASK, which has some headers and column labels
    (2) dumps *.npy (2d or 3d) of 'Mises(Cauchy)' and 'Mises(ln(V))' to 2 .npy files

Adopted from read2panda.py
"""

import argparse

import numpy as np
import pandas as pd

PARSER = argparse.ArgumentParser()

PARSER.add_argument("-f", "--filename", help='.vtr file', type=str, default='', required=True)

ARGS = PARSER.parse_args()
FILENAME = ARGS.filename

with open(FILENAME) as fileHandler:
    txt = fileHandler.readlines()


# Pre-process
NUM_HEADER_ROWS = int(txt[0].split('\t')[0])
OLD_HEADER = txt[NUM_HEADER_ROWS].replace('\n', '').split('\t')
DATA = np.loadtxt(FILENAME, skiprows=NUM_HEADER_ROWS + 1)
DF = pd.DataFrame(DATA, columns=OLD_HEADER)

# Remove duplicate columns: https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
DF = DF.loc[:, ~DF.columns.duplicated()].copy()

# Get 3d coordinates: follow geom2npy.py principles
NX = int(DF['1_pos'].max() - 0.5) + 1
NY = int(DF['2_pos'].max() - 0.5) + 1
NZ = int(DF['3_pos'].max() - 0.5) + 1

np.array(DF['1_pos'] - 0.5).astype(int).reshape(NZ, NY, NX).T
np.array(DF['2_pos'] - 0.5).astype(int).reshape(NZ, NY, NX).T
np.array(DF['3_pos'] - 0.5).astype(int).reshape(NZ, NY, NX).T

MISES_CAUCHY = np.array(DF['Mises(Cauchy)']).reshape(NZ, NY, NX).T
MISES_LN_V = np.array(DF['Mises(ln(V))']).reshape(NZ, NY, NX).T

# Dump to .npy
np.save('MisesCauchy.npy', MISES_CAUCHY)
np.save('MisesLnV.npy', MISES_LN_V)
