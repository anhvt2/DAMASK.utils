
"""
	The objective of this script is to assist a warm-restart of Bayesian optimization
	by recalculating input.dat based on the real input matcfg_input.dat
	based on the updated bounds because the global minimum was localized at the boundary
	This is an intermediate effort for extending Bayesian optimization for unbounded domain

	To use: update
		(1) lower_bounds
		(2) upper_bounds
	from parse2MaterialConfig.py. This could be also generalized by assinging these bounds elsewhere for consistency.

"""
import numpy as np
import os, sys, time, glob, datetime

## read input
matcfg_input = np.loadtxt('matcfg_input.dat', delimiter=',')
d = len(matcfg_input) # dimensionality

## get these bounds from parse2MaterialConfig.py -- CHANGE THESE PARAMETERS
lower_bounds = [    1.2,   1,    1e6,     1e6,     1e6]
upper_bounds = [  150  , 100, 1000e6, 10000e6, 10000e6]

lower_bounds = np.array(lower_bounds)
upper_bounds = np.array(upper_bounds)

bayesOpt_input = (matcfg_input - lower_bounds) / (upper_bounds - lower_bounds)

np.savetxt('bayesOpt_input.dat'     , bayesOpt_input, fmt='%.16e', delimiter=',') # debug
# np.savetxt('matcfg_input.dat', matcfg_input, fmt='%.16e', delimiter=',') # debug


