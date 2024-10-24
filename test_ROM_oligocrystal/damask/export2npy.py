
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
import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser()

parser.add_argument("-f", "--fileNameList", 
    help='specific fileName to convert to numpy', 
    type=str, default='', required=False)

parser.add_argument("-overwrite", "--overwrite",
    default=False,
    type=lambda x:bool(strtobool(x)),
    nargs='?', const=True, required=False)

args = parser.parse_args()
fileNameList = args.fileNameList # fileName = 'main_tension_inc19.txt' # debug
isOverwrite = args.overwrite

startTime = time.time()

# If fileNameList is empty, then exporting every file
if fileNameList == '':
    fileNameList = natsorted(glob.glob('main_tension_inc??.txt'))
    print('export2npy.py: Going to export to .npy from ...')
    print(fileNameList)
    print('\n\n')
else:
    fileNameList = [fileNameList] # avoid iterating through strings

logger = open('export2npy.py.log', 'w')

for fileName in fileNameList:
    outFileName = fileName[:-4] + '.npy'
    if not os.path.exists(outFileName) or isOverwrite:
        try:
            fileHandler = open(fileName)
            txt = fileHandler.readlines()
            fileHandler.close()
            # Set fields of interest to dump .npy
            FoI = ['Mises(Cauchy)','Mises(ln(V))']
            # Pre-process
            numHeaderRows = int(txt[0].split('\t')[0])
            oldHeader = txt[numHeaderRows].replace('\n', '').split('\t')
            try:
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
                # Check if FoI is contained in the dataframe
                # https://stackoverflow.com/questions/2582911/how-to-check-if-a-list-is-contained-inside-another-list-without-a-loop
                # https://stackoverflow.com/questions/23549231/check-if-a-value-exists-in-pandas-dataframe-index
                if set(FoI) <= set(df.columns):
                # Save FoI to .npy
                    np.save(outFileName, df[FoI].to_numpy())
                    print('Save data to %s' % (fileName[:-4] + '.npy'))
                else:
                    print('Dataframe in %s does not contain the field of interest.' % fileName)
            except:
                print('Cannot load DAMASK output file in %s into numpy.' % fileName)
        except:
            print('Cannot readlines() in %s' % fileName)
    else:
        print('%s already exists. Skipping %s' % (outFileName, fileName))

# Diagnostics
stopTime = time.time()
elapsedTime = stopTime - startTime
logger.write(f'elapsedTime (seconds) = {elapsedTime:<.2f}\n')
logger.close()

