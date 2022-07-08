
"""
	This script converts engineering stress/strain curves to true stress/strain curves

"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--f", help='file name with first header row', type=str, required=True)
args = parser.parse_args()

fileName = args.f

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

x = np.loadtxt(fileName, skiprows=1)

vareps = x[:,0] # strain
sigma  = x[:,1] # stress

# https://www.youtube.com/watch?v=aLYMP7uii3M
true_vareps = np.log(1 + vareps)
true_sigma = sigma * np.exp(true_vareps)
np.savetxt('true_' + fileName.split('.')[0] + '.dat', np.hstack(( np.atleast_2d(true_vareps).T, np.atleast_2d(true_sigma).T )), fmt='%.18e', header='true strain, true stress')
