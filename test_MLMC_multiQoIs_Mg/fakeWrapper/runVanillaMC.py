
import numpy as np
import os, glob
import argparse
import time
import datetime
import socket

os.system('python3 cleanseDataset.py')

def computeNumberSamples(vareps, fixed_level=4):
	d = np.loadtxt('MultilevelEstimators-multiQoIs.dat', skiprows=1, delimiter=',')
	levels = d[:,0]
	cost_per_level = [39,    365,    1955,    3305,    12487]
	d_mc = d[np.where(levels == fixed_level)] # default = 4
	max_num_samples = d_mc.shape[0]
	# calculate number of samples to converge
	n = 3 # warm-up samples
	x = d[:n, 1:]
	E = np.mean(x, axis=0)
	V = np.var(x, axis=0)
	while np.max(V) / n > vareps**2:
		if n > max_num_samples:
			print(f"Require more than {max_num_samples:<2d} samples. Out of budget.")
			break
		x = d[:n, 1:]
		E = np.mean(x, axis=0)
		V = np.var(x, axis=0)
		print(f"Take {n:<2d} samples. RMSE = {np.sqrt(np.max(V) / n):<8.8f}. Tolerance: vareps = {vareps:<8.8f}.")
		n += 1
	if n <= max_num_samples:
		return n, n * cost_per_level[-1], np.sqrt(np.max(V) / n)
	else:
		return np.nan, np.nan, np.nan

os.system('rm -fv vanilla_mc_cost.dat')
f = open('vanilla_mc_cost.dat', 'a+')
f.write('# varepsilon, num_samples, computational_cost, rmse\n')
vareps_ub = 3.51e-1 # upper bound - determined by warm-up MLMC limit
vareps_lb = 1.30e-1  # lower bound - determined by MC limit

for log_vareps in np.linspace(np.log10(vareps_ub), np.log10(vareps_lb), num=10):
	vareps = 10**log_vareps
	n, comp_cost, rmse = computeNumberSamples(vareps, fixed_level=4)
	f.write('%.8e, %.1f, %.2f, %.8e\n' % (vareps, n, comp_cost, rmse))

f.close()

# also run at each level
for level in range(5):
	f = open('vanilla_mc_cost_level-%d.dat' % level, 'w')
	f.write('# varepsilon, num_samples, computational_cost, rmse\n')
	for log_vareps in np.linspace(np.log10(vareps_ub), np.log10(vareps_lb), num=10):
		vareps = 10**log_vareps
		n, comp_cost, rmse = computeNumberSamples(vareps, fixed_level=level)
		f.write('%.8e, %.1f, %.2f, %.8e\n' % (vareps, n, comp_cost, rmse))

	f.close()

