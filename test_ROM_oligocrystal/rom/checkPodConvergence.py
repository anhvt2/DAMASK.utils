
"""
This script reconstruct 
    pred_main_tension_inc??.npy
using
    podCoefs_main_tension_inc??.npy
and
    podBasis_MisesCauchy.npy (28G)
    podBasis_MisesLnV.npy (28G)
to compare with
    main_tension_inc??.npy

See more at predictCoefs.py and computeCoefs.py
"""

import glob, os, time
import numpy as np
import logging
# import pandas as pd

level    = logging.INFO
format   = '  %(message)s'
logFileName = 'checkPodConvergence.py.log'
os.system('rm -fv %s' % logFileName)
handlers = [logging.FileHandler(logFileName), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)

t_start = time.time()

x_test       = np.loadtxt('inputRom_Test.dat',  delimiter=',', skiprows=1)
DamaskIdxs   = x_test[:,5].astype(int)
PostProcIdxs = x_test[:,6].astype(int)
NumCases = len(DamaskIdxs)

t_local = time.time()
basis_MisesCauchy = np.load('podBasis_MisesCauchy.npy')
basis_MisesLnV    = np.load('podBasis_MisesLnV.npy')
mean_MisesCauchy  = np.load('mean_MisesCauchy.npy')
mean_MisesLnV     = np.load('mean_MisesLnV.npy')
SolidIdx = np.loadtxt('SolidIdx.dat', dtype=int)
logging.info(f'checkPodConvergence.py: Load POD basis in {time.time() - t_local:<.2f} seconds.')

for NumFtrs in [1,2,4,8,16,32,64,128,256]:
    for i in range(NumCases):
        tmpSol = np.zeros([576000,2])
        predPodCoefs = np.load('../damask/%d/postProc/podCoefs_main_tension_inc%s.npy' % (DamaskIdxs[i], str(PostProcIdxs[i]).zfill(2))) # shape: (5540, 2)
        # Exclude POD coefs after NumFtrs to check POD convergence
        predPodCoefs[NumFtrs:,:] = 0
        # Option: whole domain
        # tmpSol[:,0] = np.dot(basis_MisesCauchy, predPodCoefs[:,0]) + mean_MisesCauchy
        # tmpSol[:,1] = np.dot(basis_MisesLnV,    predPodCoefs[:,1]) + mean_MisesLnV
        # Option: solid domain only
        tmpSol[SolidIdx,0] = np.dot(basis_MisesCauchy, predPodCoefs[:,0]) + mean_MisesCauchy
        tmpSol[SolidIdx,1] = np.dot(basis_MisesLnV,    predPodCoefs[:,1]) + mean_MisesLnV
        outFileName = '../damask/%d/postProc/pred_main_tension_inc%s_NumFtrs_%d' % (DamaskIdxs[i], str(PostProcIdxs[i]).zfill(2), NumFtrs)
        np.save(outFileName, tmpSol)
        logging.info(f'Processing {i+1:<d}/{NumCases} folders: dumped {outFileName}.npy')

logging.info(f'checkPodConvergence.py: Total elapsed time: {time.time() - t_start} seconds.')
os.system('htop')

