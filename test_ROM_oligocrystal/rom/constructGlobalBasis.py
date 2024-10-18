
import numpy as np
import glob
from natsort import natsorted, ns
# import pandas as pd

"""
This script
    (1) constructs the global basis,
    (2) compute the mean vector,
for projection-based ROM.
"""
TrainIdx = np.loadtxt('TrainIdx.dat')
FoI = ['Mises(Cauchy)','Mises(ln(V))'] # from export2npy.py

# Count snapshot
numSnapShot = 0
for i in TrainIdx:
    folderName = str(i+1) # from randomizeLoad.py
    print(f'Processing {int(i):<d}')
    fileNameList = natsorted(glob.glob('../damask/%d/postProc/main_tension_inc??.npy' % i))
    numSnapShot += len(fileNameList)
    # for fileName in fileNameList:
    #     d = np.load(fileName) # d = np.load('../damask/1/postProc/main_tension_inc08.npy')

print(f'numSnapShot = {numSnapShot:<d}') # numSnapShot = 11259

# Allocate snapshot matrix
d_MisesCauchy = np.zeros([576000, numSnapShot])
d_MisesLnV    = np.zeros([576000, numSnapShot])

# Create snapshot matrices
j = 0
for i in TrainIdx:
    folderName = str(i+1) # from randomizeLoad.py
    print(f'Processing damask/{int(i):<d}')
    fileNameList = natsorted(glob.glob('../damask/%d/postProc/main_tension_inc??.npy' % i))
    for fileName in fileNameList:
        try:
            print(f'Processing damask/{int(i):<d}/postProc/{fileName.split("/")[-1]}')
            d = np.load(fileName) # d = np.load('../damask/1/postProc/main_tension_inc08.npy')
            # Copy columns of snapshot
            d_MisesCauchy[:,j] = d[:,0]
            d_MisesLnV[:,j]    = d[:,1]
            j += 1
        except:
            print(f'Cannot load damask/{int(i):<d}/postProc/{fileName.split("/")[-1]}')

np.save('d_MisesCauchy.npy', d_MisesCauchy)
np.save('d_MisesLnV.npy', d_MisesLnV)

