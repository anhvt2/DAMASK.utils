

"""
	How to use: 
		python3 geom_spk2dmsk.py -r 50 -d 'dump.12.out'
		rm -f nohup.out; nohup python3 computeGeomStatistics.py --dump 'dump.12.out' --res 50 2>&1   > log.computeGeomStatistics.py &

	Parameters:
		-r: resolution: 1 pixel to 'r' micrometer
		-d: dump file from SPPARKS

	Description:
		This script computes the statistics for an "appropriate" SPPARKS dump file.
		"Appropriate" means grain ID are distinct and not the same for two far away grains.
"""

import numpy as np
import os, sys, time
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
from scipy import stats
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dump", type=str, required=True)
parser.add_argument("-r", "--resolution", type=int, required=True)
args = parser.parse_args()
dumpFileName = args.dump # 'dump.12.out'
outFileName = 'statistics_' + dumpFileName.replace('.','_') + '.geom'
res = args.resolution

def getDumpMs(dumpFileName):
	"""
		This function return a 3d array 'm' microstructure from reading a SPPARKS dump file, specified by 'dumpFileName'.
	"""
	dumpFile = open(dumpFileName)
	dumptxt = dumpFile.readlines()
	dumpFile.close()
	for i in range(20): # look for header info in first 20 lines
		tmp = dumptxt[i]
		if 'BOX BOUNDS' in tmp:
			Nx = int(dumptxt[i+1].replace('\n','').replace('0 ', '').replace(' ', ''))
			Ny = int(dumptxt[i+2].replace('\n','').replace('0 ', '').replace(' ', ''))
			Nz = int(dumptxt[i+3].replace('\n','').replace('0 ', '').replace(' ', ''))
			tmp_i = i
			break
	header = np.array(dumptxt[i+4].replace('\n','').replace('ITEM: ATOMS ', '').split(' '), dtype=str)
	d = np.loadtxt(dumpFileName, skiprows=9, dtype=int)
	num_grains = len(np.unique(d[:,1]))
	old_grain_ids = np.unique(d[:,1])
	new_grain_ids = range(len(np.unique(d[:,1])))
	grain_sizes   = np.zeros(len(np.unique(d[:,1])))
	m = np.zeros([Nx, Ny, Nz]) # initialize
	for ii in range(len(d)):
		i = int(d[ii,np.where(header=='x')[0][0]]) # 'x'
		j = int(d[ii,np.where(header=='y')[0][0]]) # 'y'
		k = int(d[ii,np.where(header=='z')[0][0]]) # 'z'
		grain_id = int(d[ii,1]) # or d[i,2] -- both are the same
		# option: DO re-enumerating
		lookup_idx = np.where(old_grain_ids == grain_id)[0][0]
		new_grain_id = new_grain_ids[lookup_idx]
		m[i,j,k] = new_grain_id
		# option: DO NOT re-enumerating
		# m[i,j,k] = grain_id # TODO: implement re-enumerate grain_id
		# print(f"finish ({x},{y}, {z})")
		## aggregate grain size statistics -- only works for re-enumerating case
		grain_sizes[new_grain_id] += 1
	complete_header = dumptxt[:tmp_i+5]
	return m, Nx, Ny, Nz, num_grains, grain_sizes, complete_header

t_start = time.time()
m, Nx, Ny, Nz, num_grains, grain_sizes, complete_header = getDumpMs(dumpFileName)

grain_size_kernel = stats.gaussian_kde(grain_sizes)
g = np.linspace(0, np.max(grain_sizes) * 1.1, num=100)
grain_size_kernel(g)
# normalized_factor = np.max(np.histogram(grain_sizes, bins='auto')[0]) / np.max(grain_size_kernel(g)) 
# plt.plot(g, grain_size_kernel(g) * normalized_factor, 'ro', ms=2, label='KDE')
# plt.hist(grain_sizes, bins='auto', label='histogram')
# plt.legend(loc='best', fontsize=24, frameon=False)
# plt.xlabel(r'grain size [pixel$^3$]', fontsize=24)
# plt.ylabel(r'frequency', fontsize=24)
# plt.title('SPPARKS dump statistics by computeGeomStatistics.py', fontsize=24)
# plt.show()
elapsed = time.time() - t_start
f = open('log.computeGeomStatistics.py', 'a')
f.write('###\n')
f.write('dumpFileName = \'%s\'\n' % dumpFileName)
f.write("computeGeomStatistics.py: finished in {:5.2f} seconds.\n".format(elapsed))
f.write('complete_header:\n')
for i in range(len(complete_header)):
	f.write(complete_header[i])
f.write('###\n')
f.close()
