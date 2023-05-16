
""" Usage
	python3 convert2dGeom2Graph.py --geom='MgRve_16x16x1.geom'
	This script uses a lot of Mahattan distance (l_1 norm) on taxicab geometry
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

def locateGrainId(grain_id):
	locations = np.argwhere(d == grain_id)
	# note: have to call out explicitly to verify d[ locations[0], locations[1], locations[2] ]
	return locations

def computeDistance(pts_cloud_1, pts_cloud_2):
	"""
	This function computes Manhattan distance between two point clouds (often queried by two grain_id's) 
		where pts_cloud = locateGrainId(grain_id)
	
	This implpementation is NOT optimal. Consider improving in the future.

	This implementation has NOT considered periodic boundary conditions. Need to add PBC.
	"""
	num_pts_1 = pts_cloud_1.shape[0]
	num_pts_2 = pts_cloud_2.shape[0]
	matrix_distance = np.zeros([num_pts_1, num_pts_2])
	for i in range(num_pts_1):
		for j in range(num_pts_2):
			matrix_distance[i,j] = np.linalg.norm(pts_cloud_1[i] - pts_cloud_2[j], ord=1)
			# matrix_distance[i,j] = np.sum(np.abs(pts_cloud_1[i] - pts_cloud_2[j]))
	return matrix_distance

def isNeighbor(grain_id_1, grain_id_2):
	matrix_distance = computeDistance(locateGrainId(grain_id_1), locateGrainId(2))
	if np.min(matrix_distance) == 1:
		return True
	else:
		return False

geomFileName = 'MgRve_16x16x1.geom'
x_res = int(geomFileName.split('.')[0].split('_')[1].split('x')[0])
y_res = int(geomFileName.split('.')[0].split('_')[1].split('x')[1])
z_res = int(geomFileName.split('.')[0].split('_')[1].split('x')[2])

d = read_geom(geomFileName)
# from geom to table
d = np.reshape(d, [z_res, y_res, x_res]).T

# from table to geom
geom = d.T.flatten()

grain_ids = np.unique(d)

# example -- debug
grain_id_1 = 1 
loc_1 = locateGrainId(grain_id_1)
grain_id_2 = 2
loc_2 = locateGrainId(grain_id_2)
matrix_distance = computeDistance(loc_1, loc_2)
