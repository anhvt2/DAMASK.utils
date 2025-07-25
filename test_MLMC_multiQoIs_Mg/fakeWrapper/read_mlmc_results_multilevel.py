
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob, os
from natsort import natsorted, ns

cost_per_level = [39,    365,    1955,    3305,    12487]
cost_per_level = np.array(cost_per_level) / 60 # measured in hours
num_levels = len(cost_per_level)

# os.system('rm -v mlmc_cost')
f = open('mlmc_cost.dat', 'w')
f.write('# varepsilon, n, computational_cost, rmse\n')
f.close()

f = open('mlmc_frac_cost.txt', 'w')
f.write('0	1	2	3	4	total\n')
f.close()

for fileName in natsorted(glob.glob('log.mlmc.vareps-*'), reverse=True):
	str_vareps = fileName.split('vareps')[1]
	vareps = - float(str_vareps)
	print(f"Processing {fileName}")

	f = open(fileName)
	txt = f.readlines()
	f.close()
	num_lines = len(txt) # number of lines
	n = np.zeros([num_levels])
	try: n[0] = int(txt[-10].split('│')[6]) # number of samples used at level 0
	except: n[0] = 0
	try: n[1] = int(txt[-9].split('│')[6])  # number of samples used at level 1
	except: n[1] = 0
	try: n[2] = int(txt[-8].split('│')[6])  # number of samples used at level 2
	except: n[2] = 0
	try: n[3] = int(txt[-7].split('│')[6])  # number of samples used at level 3
	except: n[3] = 0
	try: n[4] = int(txt[-6].split('│')[6])  # number of samples used at level 4
	except: n[4] = 0
	rmse = float(txt[-5].split('≈')[1].replace('\n','')[:-1]) # rmse

	total_cost = np.sum(n * cost_per_level) 
	frac_cost = n * cost_per_level / total_cost * 1e2
	f = open('mlmc_cost.dat', 'a+')
	f.write('%.8e, %d, %d, %d, %d, %d, %.2f, %.8e\n' % (vareps, n[0], n[1], n[2], n[3], n[4], total_cost, rmse))
	f.close()

	f = open('mlmc_frac_cost.txt', 'a+')
	f.write('%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.2f\n' % (frac_cost[0], frac_cost[1], frac_cost[2], frac_cost[3], frac_cost[4], total_cost))
	f.close()

## TODO: rewrite mlmc_frac_cost.txt and mlmc_cost.dat sorted by vareps
d = np.loadtxt('mlmc_cost.dat', delimiter=',', skiprows=1)
d = d[d[:, 0].argsort()]
d = d[::-1,:] # flip top/bottom
np.savetxt('mlmc_cost.dat', d, header="# varepsilon, n, computational_cost, rmse", fmt="%.8e, %d, %d, %d, %d, %d, %.2f, %.8e", comments='')

d = np.loadtxt('mlmc_frac_cost.txt', delimiter='\t', skiprows=1)
d = d[d[:, -1].argsort()]
np.savetxt('mlmc_frac_cost.txt', d, header="0	1	2	3	4	total", fmt="%.8e	%.8e	%.8e	%.8e	%.8e	%.2f", comments='')

