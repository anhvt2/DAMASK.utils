
import numpy as np
import os, glob
from natsort import natsorted, ns
import time
import logging
# import pandas as pd

level    = logging.INFO
format   = '  %(message)s'
logFileName = 'extractData.py.log'
os.system('rm -fv %s' % logFileName)
handlers = [logging.FileHandler(logFileName), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)

"""
This script
    (1) constructs the global basis,
    (2) compute the mean vector,
for projection-based ROM.
"""
t_start = time.time()
TrainIdx = np.loadtxt('TrainIdx.dat')
FoI = ['Mises(Cauchy)','Mises(ln(V))'] # from export2npy.py

# Count snapshot
numSnapShot = 0
for i in TrainIdx:
    folderName = str(i+1) # from randomizeLoad.py
    logging.info(f'Processing {int(i):<d}')
    fileNameList = natsorted(glob.glob('../damask/%d/postProc/main_tension_inc??.npy' % i))
    numSnapShot += len(fileNameList)
    # for fileName in fileNameList:
    #     d = np.load(fileName) # d = np.load('../damask/1/postProc/main_tension_inc08.npy')

logging.info(f'numSnapShot = {numSnapShot:<d}') # numSnapShot = 11259

# Allocate snapshot matrix
d_MisesCauchy = np.zeros([576000, numSnapShot])
d_MisesLnV    = np.zeros([576000, numSnapShot])

# Create snapshot matrices
j = 0
for i in TrainIdx:
    folderName = str(i+1) # from randomizeLoad.py
    logging.info(f'Processing folder damask/{int(i+1):<d}')
    fileNameList = natsorted(glob.glob('../damask/%d/postProc/main_tension_inc??.npy' % i))
    for fileName in fileNameList:
        try:
            logging.info(f'Processing damask/{int(i):<d}/postProc/{fileName.split("/")[-1]}')
            d = np.load(fileName) # d = np.load('../damask/1/postProc/main_tension_inc08.npy')
            # Copy columns of snapshot
            d_MisesCauchy[:,j] = d[:,0]
            d_MisesLnV[:,j]    = d[:,1]
            j += 1
        except:
            logging.info(f'Cannot load damask/{int(i):<d}/postProc/{fileName.split("/")[-1]}')

logging.info("extractData.py: extracted data in {:5.2f} seconds.\n".format(time.time() - t_start))

# Save the original data
np.save('d_MisesCauchy.npy', d_MisesCauchy)
np.save('d_MisesLnV.npy'   , d_MisesLnV)

# Save the logarithm data -- avoid np.log10(0) by adding some 'numerical' noise (smallest non-zero value in the dataset)
def findSmallestNonzero(d):
    t = np.sort(d.ravel())
    idx = len(t) - (t>0).sum()
    return t[idx]

np.save('log10d_MisesCauchy.npy', np.log10(d_MisesCauchy+findSmallestNonzero(d_MisesCauchy)))
np.save('log10d_MisesLnV.npy'   , np.log10(d_MisesLnV+findSmallestNonzero(d_MisesLnV)))
# np.save('log10d_MisesCauchy.npy', np.log10(d_MisesCauchy+11.3364777529))
# np.save('log10d_MisesLnV.npy'   , np.log10(d_MisesLnV+2.18415723949e-09))

elapsed = time.time() - t_start
logging.info("extractData.py: finished in {:5.2f} seconds.\n".format(elapsed))


