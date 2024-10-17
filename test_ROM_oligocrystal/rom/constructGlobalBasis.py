
import numpy as np
# import pandas as pd

"""
This script
    (1) constructs the global basis,
    (2) compute the mean vector,
for projection-based ROM.
"""
TrainIdx = np.loadtxt('TrainIdx.dat')
FoI = ['Mises(Cauchy)','Mises(ln(V))'] # from export2npy.py

numSnapShot = 0

# data = np.zeros([576000, 1]) # initialize
# data = np.c_[data, np.zeros(len(data))]

for i in TrainIdx:
    folderName = str(i+1) # from randomizeLoad.py
    print(f'Processing {i}')
    fileNameList = natsorted(glob.glob('../damask/%d/postProc/main_tension_inc??.npy' % i)):
    numSnapShot += len(fileNameList)
    # for fileName in fileNameList:
    #     d = np.load(fileName) # d = np.load('../damask/1/postProc/main_tension_inc08.npy')

