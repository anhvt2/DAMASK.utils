
""" Usage
	python3 convert2dGeom2Graph.py --geom='MgRve_16x16x1.geom'
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--geom", type=str, required=True)
args = parser.parse_args()
geomFileName = args.geom

def read_geom(geomFileName):
	f = open(geomFileName, 'r')
	txt = f.readlines()
	f.close()
	num_headers = int(txt[0][0]) + 1
	txt = txt[num_headers:]
	tmp = []
	for i in range(len(txt)):
		txt[i] = txt[i].replace('\n', '')
		tmp += txt[i].split()
	d = np.array(tmp, dtype=int)
	return d

# geomFileName = 'MgRve_16x16x1.geom'
x_res = int(geomFileName.split('.')[0].split('_')[1].split('x')[0])
y_res = int(geomFileName.split('.')[0].split('_')[1].split('x')[1])
z_res = int(geomFileName.split('.')[0].split('_')[1].split('x')[2])

d = read_geom(geomFileName)
# from geom to table
d = np.reshape(d, [z_res, y_res, x_res]).T

# from table to geom
geom = d.T.flatten()

