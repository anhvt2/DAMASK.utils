
""" Usage
	This script reads and rescales a table, output from geom_toTable.py command in DAMASK
	Example:
		python3 rescaleGeom.py --table=MgRve_16x16x1.txt
"""


import numpy as np
import os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--table", type=str, required=True)
args = parser.parse_args()
tableFileName = args.table

def getNumHeaders(tableFileName):
	f = open(tableFileName, 'r')
	txt = f.readlines()
	f.close()
	num_headers = int(txt[0][0]) + 1
	return txt, num_headers

txt, num_headers = getNumHeaders(tableFileName)

d = np.loadtxt(tableFileName, skiprows=num_headers)
# d[:,0:3] /= np.min(d, axis=0)[0:3]

def getIncrement(d, axisIndex):
	if len(np.unique(d[:,axisIndex])) > 1:
		delta = np.sort(np.unique(d[:,axisIndex]))[1] - np.sort(np.unique(d[:,axisIndex]))[0]
	else:
		delta = 1
	return delta

delta_x = getIncrement(d, axisIndex=0)
delta_y = getIncrement(d, axisIndex=1)
delta_z = getIncrement(d, axisIndex=2)

d[:,0:3] -= np.min(d[:,0:3], axis=0)
d[:,0:3] /= np.array([delta_x, delta_y, delta_z])

outFileName = 'rescaled_' + tableFileName
os.system('rm -fv %s' % outFileName)
f = open(outFileName, 'a')
for i in range(num_headers):
	f.write(txt[i])

np.savetxt(f, d, fmt='%d		%d		%d		%d', newline="\n")
f.close()

