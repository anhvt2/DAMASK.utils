
import numpy as np
import os, glob
import time
import logging
from natsort import natsorted, ns

t_start = time.time()
level    = logging.INFO
format   = '  %(message)s'
logFileName = 'computeCoefs.py.log'
os.system('rm -fv %s' % logFileName)
handlers = [logging.FileHandler(logFileName), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)

# Load POD bases
basis_MisesCauchy = np.load('podBasis_MisesCauchy.npy')
logging.info(f'computeCoefs.py: Loading POD basis: Elapsed {time.time() - t_start} seconds.')
basis_MisesLnV = np.load('podBasis_MisesLnV.npy')
logging.info(f'computeCoefs.py: Loading POD basis: Elapsed {time.time() - t_start} seconds.')
# Load POD mean
mean_MisesCauchy = np.load('mean_MisesCauchy.npy')
mean_MisesLnV    = np.load('mean_MisesLnV.npy')

# os.chdir(romPath) # debug
romPath = os.getcwd() + '/'
damaskPath  = romPath + '../damask'
os.chdir(damaskPath)

for i in range(1,1001):
    print(f'Computing POD coefs in damask/{int(i):<d}/')
    os.chdir(damaskPath + '/%d/postProc/' % i)
    for fileName in natsorted(glob.glob('main_tension_inc??.npy')):
        try:
            tmpData = np.load(fileName)
            # print(tmpData.shape) # debug
            if tmpData.shape == (576000,2): # safeguard -- check shape
                try:
                    # Subtract the mean
                    fluct_MisesCauchy = np.atleast_2d(tmpData[:,0] - mean_MisesCauchy).T
                    fluct_MisesLnV    = np.atleast_2d(tmpData[:,1] - mean_MisesLnV).T
                    # Project into POD space
                    podCoefs_MisesCauchy = np.dot(basis_MisesCauchy.T, fluct_MisesCauchy)
                    podCoefs_MisesLnV    = np.dot(basis_MisesLnV.T,    fluct_MisesLnV)
                    # # Save POD coefs
                    podCoefs = np.hstack((podCoefs_MisesCauchy, podCoefs_MisesLnV))
                    print(f'Finish calculating POD coefs in damask/{int(i):<d}/postProc/{fileName}')
                    np.save('podCoefs_' + fileName.split[:-4], podCoefs)
                except:
                    print(f'Fail to calculate POD coefs in damask/{int(i):<d}/postProc/{fileName}')
        except:
            print(f'Cannot load damask/{int(i):<d}/postProc/{fileName}')
    os.chdir(damaskPath)

logging.info(f'computeCoefs.py: Elapsed time = {time.time() - t_start} seconds.')
