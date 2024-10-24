
import numpy as np
import os, glob
import time
import logging

t_start = time.time()
level    = logging.INFO
format   = '  %(message)s'
logFileName = 'computeCoefs.py.log'
os.system('rm -fv %s' % logFileName)
handlers = [logging.FileHandler(logFileName), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)

basis_MisesCauchy = np.load('podBasis_MisesCauchy.npy')
basis_MisesLnV = np.load('podBasis_MisesLnV.npy')

romPath = os.getcwd() + '/'
damaskPath  = romPath + '../damask/'
os.chdir(damaskPath)
for i in range(1,1001):
    print(f'Computing POD coefs in damask/{int(i):<d}/')
    os.chdir(damaskPath + '%s/postProc/')
    for fileName in natsorted(glob.glob('main_tension_inc??.npy' % i)):
        tmpData = np.load(fileName)
        if tmpData.shape == (576000,2): # safeguard -- check shape
            podCoefs_MisesCauchy = np.dot(basis_MisesCauchy, tmpData[:,0])
            podCoefs_MisesLnV    = np.dot(basis_MisesLnV,    tmpData[:,1])
            podCoefs = np.hstack((podCoefs_MisesCauchy, podCoefs_MisesLnV))
            print(f'Finish calculating POD coefs in damask/{int(i):<d}/postProc/{fileName}')
            np.save('podCoefs_' + fileName.split('.')[:-4], podCoefs)

    os.chdir(damaskPath)

logging.info(f'computeCoefs.py: Elapsed time = {time.time() - t_start} seconds.')
