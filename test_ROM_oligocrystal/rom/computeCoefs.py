
import numpy as np
import os, glob
import time
import logging

level    = logging.INFO
format   = '  %(message)s'
logFileName = 'extractRomInput.py.log'
os.system('rm -fv %s' % logFileName)
handlers = [logging.FileHandler(logFileName), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)

basis_MisesCauchy = np.load('podBasis_MisesCauchy.npy')
basis_MisesLnV = np.load('podBasis_MisesLnV.npy')

for 
    if d.shape == (576000,2): # safeguard -- check shape

