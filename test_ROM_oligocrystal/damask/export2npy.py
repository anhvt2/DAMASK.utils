
"""
    This script 
        (1) Reads an output (default: ['Mises(Cauchy)','Mises(ln(V))'] ) from DAMASK and save it to a numpy array
        (2) Remove duplicate columns
    Example:
        python3 export2npy.py --fileName 'main_tension_inc19.txt'
"""

import time
import numpy as np
import pandas as pd
import glob, os
from natsort import natsorted, ns
# import argparse
# parser = argparse.ArgumentParser()

# parser.add_argument("-f", "--fileName", 
#     help='<geom>-<load>-inc*.txt', 
#     type=str, default='', required=True)

# args = parser.parse_args()
# fileName = args.fileName # fileName = 'main_tension_inc19.txt' # debug

startTime = time.time()
fileNameList = natsorted(glob.glob('main_tension_inc??.txt'))
logger = open('export2npy.py.log', 'w')

for fileName in fileNameList:
    outFileName = fileName[:-4] + '.npy'
    if not os.path.exists(outFileName):
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
        # Save DAMASK header
        DamaskCommandHistory = txt[:numHeaderRows]
        DamaskCommandHistory += ['\t'.join(newHeader) + '\n'] # add the last line
        # Reformat columns
        for fieldName in ['inc', 'elem', 'node', 'ip', 'grain']:
            df[fieldName] = df[fieldName].astype(int)
        # # Write header
        # f = open(fileName[:-4] + '_header.txt', 'w')
        # for i in range(len(DamaskCommandHistory)):
        #     f.write(DamaskCommandHistory[i])
        # f.close()
        # print('Dump header to %s' % (fileName[:-4] + '_header.txt'))
        # # Write to .csv: Do not rewrite header (headers are confirmed to be the same with DamaskCommandHistory[-1])
        # df.to_csv(fileName[:-4] + '_data.txt', sep='\t', header=False, index=False)
        # print('Dump data to %s' % (fileName[:-4] + '_header.txt'))
        # Save FoI to .npy
        np.save(outFileName, df[FoI].to_numpy())
        print('Save data to %s' % (fileName[:-4] + '.npy'))
    else:
        print('%s already exists. Skipping %s' % (outFileName, outFileName))

# Diagnostics
stopTime = time.time()
elapsedTime = stopTime - startTime
logger.write(f'elapsedTime (seconds) = {elapsedTime:<.2f}\n')
logger.close()

