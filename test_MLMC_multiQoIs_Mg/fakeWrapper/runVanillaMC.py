
import numpy as np
import os, glob
import argparse
import time
import datetime
import socket

os.system('python3 cleanseDataset.py')

def computeNumberSamples(vareps):
	d = np.loadtxt('MultilevelEstimators-multiQoIs.dat', skiprows=1, delimiter=',')
	levels = d[:,0]
	cost_per_level = [39,    365,    1955,    3305,    12487]
	d_mc = d[np.where(levels == 4)]
	max_num_samples = d_mc.shape[0]
	# calculate number of samples to converge
	n = 3 # warm-up samples
	x = d[:n, 1:]
	E = np.mean(x, axis=0)
	V = np.var(x, axis=0)
	while np.max(V) / n > vareps**2 / 2:
		if n > max_num_samples:
			print(f"Require more than {max_num_samples:<2d} samples. Out of budget.")
			break
		x = d[:n, 1:]
		E = np.mean(x, axis=0)
		V = np.var(x, axis=0)
		print(f"Take {n:<2d} samples. np.max(V) / n = {np.max(V) / n:<8.5f}. Tolerance: vareps**2 / 2 = {vareps**2 / 2:<13.8f}")
		n += 1
	return n, n * cost_per_level[-1], np.sqrt(2 * np.max(V) / n)



os.system('rm -fv vanilla_mc_cost.dat')
f = open('vanilla_mc_cost.dat', 'a+')
f.write('# varepsilon, num_samples, computational_cost, final_vareps\n')
for vareps in np.arange(1.0, 0.2, -0.1):
	n, comp_cost, final_vareps = computeNumberSamples(vareps)
	f.write('%.8e, %d, %.2f, %.8e\n' % (vareps, n, comp_cost, final_vareps))

f.close()
