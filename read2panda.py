
"""
    This script 
        (1) Reads an output from DAMASK and convert it to a panda frame
        (2) (Optionally) Remove duplicate columns
        (3) Write to .csv with a relatively close output (w/ changed headers)

"""

import numpy as np
import pandas as pd
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-f", "--filename", help='.vtr file', type=str, default='', required=True)
args = parser.parse_args()
filename = args.filename
# filename = 'main_tension_inc01.txt' # debug

fileHandler = open(filename)
txt = fileHandler.readlines()
fileHandler.close()

### Pre-process
numHeaderRows = int(txt[0].split('\t')[0])
oldHeader = txt[numHeaderRows].replace('\n', '').split('\t')
data = np.loadtxt(filename, skiprows=numHeaderRows+1)
df = pd.DataFrame(data, columns=oldHeader)

# Remove duplicate columns: https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
df = df.loc[:,~df.columns.duplicated()].copy()
newHeader = list(df)

DamaskCommandHistory = txt[:numHeaderRows]
DamaskCommandHistory += ['\t'.join(newHeader) + '\n'] # add the last line

f = open(filename[:-4] + '_header.txt', 'w')
for i in range(len(DamaskCommandHistory)):
    f.write(DamaskCommandHistory[i])

f.close()

df.to_csv(filename[:-4] + '_data.txt', sep='\t', header=False) # do not rewrite header
# np.savetxt(filename[:-4] + '_data.txt', df.to_numpy(), fmt='%.16e', delimiter='\t')

