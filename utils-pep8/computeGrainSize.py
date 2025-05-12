#!/usr/bin/env python3

import numpy as np

MU = np.loadtxt('mu.dat')

with open('grainSize.dat', 'w') as outFile:
    outFile.write('%.4e\n' % (np.exp(MU) * 1e-06))
