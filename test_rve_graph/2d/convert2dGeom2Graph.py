
""" Usage
	python3 convert2dGeom2Graph.py --geom='MgRve_16x16x1.geom'
	This script uses a lot of Mahattan distance (l_1 norm) on taxicab geometry
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import networkx as nx

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
	"""
	This implementation does NOT considered periodic boundary conditions. Need PBC? Use isNeighborPBC() instead.
	"""
	matrix_distance = computeDistance(locateGrainId(grain_id_1), locateGrainId(2))
	if np.min(matrix_distance) == 1:
		return True
	else:
		return False

def isNeighborPBC(grain_id_1, grain_id_2, pbc='xyz'):
	"""
	This implementation considers if two grains are neighbors under the PBC (w.r.t x-, y-, z- or all).
	Default: pbc = 'xyz' -- periodic in all directions.
	"""
	loc_grain_id_1 = locateGrainId(grain_id_1)
	loc_grain_id_2 = locateGrainId(grain_id_2)
	matrix_distance = computeDistance(loc_grain_id_1, loc_grain_id_2)
	min_distance = np.min(matrix_distance)
	# fixed grain_id_1, mirror grain_id_2
	if min_distance == 1:
		return True
	else: # only check periodic images if distance > 1
		if 'x' in pbc:
			# print(f"Invoking x direction") # debug
			mirror_grain_id_2_neg_x = np.copy(loc_grain_id_2)
			mirror_grain_id_2_neg_x[:,0] -= x_res
			mirror_grain_id_2_pos_x = np.copy(loc_grain_id_2)
			mirror_grain_id_2_pos_x[:,0] += x_res
			matrix_distance_neg_x = computeDistance(loc_grain_id_1, mirror_grain_id_2_neg_x)
			matrix_distance_pos_x = computeDistance(loc_grain_id_1, mirror_grain_id_2_pos_x)
			# print(matrix_distance_neg_x) # debug
			# print(matrix_distance_pos_x) # debug
			min_distance = np.min([np.min(matrix_distance_neg_x), np.min(matrix_distance_pos_x), min_distance])
		if 'y' in pbc:
			# print(f"Invoking y direction") # debug
			mirror_grain_id_2_neg_y = np.copy(loc_grain_id_2)
			mirror_grain_id_2_neg_y[:,0] -= y_res
			mirror_grain_id_2_pos_y = np.copy(loc_grain_id_2)
			mirror_grain_id_2_pos_y[:,0] += y_res
			matrix_distance_neg_y = computeDistance(loc_grain_id_1, mirror_grain_id_2_neg_y)
			matrix_distance_pos_y = computeDistance(loc_grain_id_1, mirror_grain_id_2_pos_y)
			# print(matrix_distance_neg_y) # debug
			# print(matrix_distance_pos_y) # debug
			min_distance = np.min([np.min(matrix_distance_neg_y), np.min(matrix_distance_pos_y), min_distance])
		if 'z' in pbc:
			# print(f"Invoking z direction") # debug
			mirror_grain_id_2_neg_z = np.copy(loc_grain_id_2)
			mirror_grain_id_2_neg_z[:,0] -= z_res
			mirror_grain_id_2_pos_z = np.copy(loc_grain_id_2)
			mirror_grain_id_2_pos_z[:,0] += z_res
			matrix_distance_neg_z = computeDistance(loc_grain_id_1, mirror_grain_id_2_neg_z)
			matrix_distance_pos_z = computeDistance(loc_grain_id_1, mirror_grain_id_2_pos_z)
			# print(matrix_distance_neg_z) # debug
			# print(matrix_distance_pos_z) # debug
			min_distance = np.min([np.min(matrix_distance_neg_z), np.min(matrix_distance_pos_z), min_distance])
		if min_distance == 1:
			return True
		else:
			return False


def buildAdjacencyMatrix(grain_ids):
	A = np.zeros([num_grains, num_grains])
	for i in range(num_grains):
		for j in range(i+1, num_grains):
			# print(f"i = {i}; j = {j}") # debug
			A[i,j] = isNeighborPBC(i+1, j+1, pbc='xyz') # NOTE: grain_ids indexed 1, A[i,j] indexed at 0
	A = A + A.T # mirror symmetric
	return A

def buildGraph(A):
	# A = buildAdjacencyMatrix(grain_ids) # NOTE: computationally expensive - don't invoke unless absolutely necessary
	G = nx.Graph()
	for i in range(num_grains):
		for j in range(i+1, num_grains):
			if A[i,j] > 0: # or A[i,j] == 1
				G.add_edge(i,j)
	return G

def buildDegreeMatrix(A):
	D = np.zeros([num_grains, num_grains])
	for i in range(num_grains):
		for j in range(i+1, num_grains):
			D[i,i] = np.sum(A[i,:])
	return D

def buildLaplacianMatrix(A):
	D = buildDegreeMatrix(A)
	L = D - A
	return L

# geomFileName = 'MgRve_16x16x1.geom'
x_res = int(geomFileName.split('.')[0].split('_')[1].split('x')[0])
y_res = int(geomFileName.split('.')[0].split('_')[1].split('x')[1])
z_res = int(geomFileName.split('.')[0].split('_')[1].split('x')[2])

d = read_geom(geomFileName)
# from geom to table
d = np.reshape(d, [z_res, y_res, x_res]).T

# from table to geom
geom = d.T.flatten()

grain_ids = np.sort(np.unique(d))
num_grains = len(grain_ids)

# example -- debug
grain_id_1 = 1 
loc_1 = locateGrainId(grain_id_1)
grain_id_2 = 2
loc_2 = locateGrainId(grain_id_2)
matrix_distance = computeDistance(loc_1, loc_2)

A = buildAdjacencyMatrix(grain_ids)
G = buildGraph(A)
D = buildDegreeMatrix(A)
L = buildLaplacianMatrix(A)
# nx.draw(G, with_labels=True, node_color="tab:blue", font_size=22)
labeldict = {}
for i in range(num_grains):
	labeldict[i] = '%d' % (i+1)

# d = nx.degree(G)
# https://stackoverflow.com/questions/16566871/node-size-dependent-on-the-node-degree-on-networkx
# list_degree=list(G.degree()) #this will return a list of tuples each tuple is(node,deg)
# nodes , degree = map(list, zip(*list_degree)) #build a node list and corresponding degree list
plt.figure(figsize=(12,12), dpi=600)
nx.draw(G, with_labels=True, labels=labeldict)
# nx.draw(G, with_labels=True, node_color="tab:blue", alpha=0.75, node_size=[D[i,i] * 100 for i in range(num_grains)], font_size=12)
# nx.draw(G, labels=labeldict, with_labels=True, node_color="tab:blue", alpha=0.75, node_size=[D[i,i] * 100 for i in range(num_grains)], font_size=12)
# nx.draw_networkx_labels(G, pos=nx.spring_layout(G), labels=grain_ids, alpha=0.75, font_size=22)
# print([D[i,i] for i in range(num_grains)])

plt.savefig('graph_' + geomFileName.split('.')[0] + '.png')
plt.show()

