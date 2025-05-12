#!/usr/bin/env python3

import glob
import os

import numpy as np
from natsort import natsorted

CURRENT_PATH = os.getcwd()

print('Current path = %s' % CURRENT_PATH)
for folderName in natsorted(glob.glob('*x*x*')):
    dimX, dimY, dimZ = folderName.split('x')
    _, _, dimZ = int(dimX), int(dimY), int(dimZ)
    numProcessors = int(np.floor(dimZ / 4.0))
    if numProcessors <= 0:
        numProcessors += 1
    print('Assigning %d processors to %s' % (numProcessors, folderName))
    np.savetxt(folderName + '/numProcessors.dat', [numProcessors], fmt='%d')
