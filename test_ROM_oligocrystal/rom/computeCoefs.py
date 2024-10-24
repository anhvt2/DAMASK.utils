
import numpy as np
import os
import time
import logging

level    = logging.INFO
format   = '  %(message)s'
logFileName = 'extractRomInput.py.log'
os.system('rm -fv %s' % logFileName)
handlers = [logging.FileHandler(logFileName), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)

podBasisMisesCauchy = np.load('podBasis_MisesCauchy.npy')
podBasisMisesLnV = np.load('podBasis_MisesLnV.npy')

