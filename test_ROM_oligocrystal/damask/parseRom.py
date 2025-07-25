
"""
    This script 
        (1) Reads an output (default: ['Mises(Cauchy)','Mises(ln(V))'] ) from DAMASK 
        (2) Reads ROM construction in .npy format
        (3) Dumps an output similar to DAMASK but as a ROM solution
    Example:
        python3 parseRom.py --fileName 'main_tension_inc19.txt' --romNpy <npyFileName>
"""

import time
import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
startTime = time.time()

parser.add_argument("-f", "--fileName", 
    help='<geom>-<load>-inc*.txt', 
    type=str, default='', required=True)

args = parser.parse_args()
fileName = args.fileName # fileName = 'main_tension_inc19.txt' # debug
npyFileName = args.npyFileName # npyFileName = 'main_tension_inc19.npy' # debug

fileHandler = open(fileName)
txt = fileHandler.readlines()
fileHandler.close()

# Set fields of interest to dump .npy
FoI = ['Mises(Cauchy)','Mises(ln(V))']

# Pre-process
numHeaderRows = int(txt[0].split('\t')[0])
oldHeader = txt[numHeaderRows].replace('\n', '').split('\t')
data = np.loadtxt(fileName, skiprows=numHeaderRows+1)
df = pd.DataFrame(data, columns=oldHeader)

# Remove duplicate columns: https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
df = df.loc[:,~df.columns.duplicated()].copy()
newHeader = list(df)

DamaskCommandHistory = txt[:numHeaderRows]
DamaskCommandHistory += ['\t'.join(newHeader) + '\n'] # add the last line

# Reformat columns
for fieldName in ['inc', 'elem', 'node', 'ip', 'grain']:
    df[fieldName] = df[fieldName].astype(int)

DamaskCommandHistory = txt[:numHeaderRows]
DamaskCommandHistory += ['\t'.join(newHeader) + '\n'] # add the last line

# Read ROM data
romNpy = np.load(npyFileName)

# Update dataframe
df2 = df.copy
RandomArray = np.random.rand(576000, 2)
df2[FoI] = RandomArray

# Write header
f = open(fileName[:-4] + '_header.txt', 'w')
for i in range(len(DamaskCommandHistory)):
    f.write(DamaskCommandHistory[i])

f.close()
print('Dump header to %s' % (fileName[:-4] + '_header.txt'))

# Reformat columns
for fieldName in ['inc', 'elem', 'node', 'ip', 'grain']:
    df[fieldName] = df[fieldName].astype(int)

# Write to .csv: Do not rewrite header (headers are confirmed to be the same with DamaskCommandHistory[-1])
df.to_csv(fileName[:-4] + '_data.txt', sep='\t', header=False, index=False)
print('Dump data to %s' % (fileName[:-4] + '_header.txt'))

# Diagnostics
stopTime = time.time()
elapsedTime = stopTime - startTime
print(f'elapsedTime (seconds) = {elapsedTime:<.2f}')

