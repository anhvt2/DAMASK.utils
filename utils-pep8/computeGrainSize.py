#!/usr/bin/env python3

import numpy as np

mu = np.loadtxt('mu.dat')

outFile = open('grainSize.dat', 'w')
outFile.write('%.4e\n' % (np.exp(mu) * 1e-6))
outFile.close()
