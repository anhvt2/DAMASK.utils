#!/usr/bin/env python3

"""
This script
    (1) Reads an output from DAMASK and convert it to a panda frame
    (2) (Optionally) Remove duplicate columns
    (3) Write to .csv with a relatively close output (w/ changed headers)

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
NEW_HEADER = list(DF)

DAMASK_COMMAND_HISTORY = txt[:NUM_HEADER_ROWS]
DAMASK_COMMAND_HISTORY += ['\t'.join(NEW_HEADER) + '\n']
  # add the last line

with open(FILENAME[:-4] + '_header.txt', 'w') as f:
    for i in range(len(DAMASK_COMMAND_HISTORY)):
        f.write(DAMASK_COMMAND_HISTORY[i])


DF.to_csv(FILENAME[:-4] + '_data.txt', sep='\t', header=False)
  # do not rewrite header
