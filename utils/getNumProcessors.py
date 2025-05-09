#!/usr/bin/env python3

import numpy as np
import os, glob
from natsort import natsorted, ns

currentPath = os.getcwd()

print('Current path = %s' % currentPath)
for folderName in natsorted(glob.glob('*x*x*')):
	# print(folderName)
	dimX, dimY, dimZ = folderName.split('x')
	dimX, dimY, dimZ = int(dimX), int(dimY), int(dimZ)
	numProcessors = int(np.floor(dimZ / 4.))
	if numProcessors <= 0:
		numProcessors += 1
	print('Assigning %d processors to %s' % (numProcessors, folderName))
	np.savetxt(folderName + '/numProcessors.dat', [numProcessors], fmt='%d')

