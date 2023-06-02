
import numpy as np
import os, sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dump", type=str, required=True)
parser.add_argument("-r", "--resolution", type=int, required=True)
parser.add_argument('--dogbone_geom', nargs='+', type=int) # geometry input: (L, W, T, l, w, b, R)
geom_tuple = tuple(args.dogbone_geom)
# https://stackoverflow.com/questions/33564246/passing-a-tuple-as-command-line-argument

args = parser.parse_args()

dumpFileName = args.dump # 'dump.12.out'
res = args.resolution

outFileName = 'phase_' + dumpFileName.replace('.','_') + '.npy'


"""
	How to use: 
		python3 geom_cad2phase.py -r 50 -d 'dump.12.out' -L

	Parameters:
		-r: resolution: 1 pixel to 'r' micrometer
		-d: dump file from SPPARKS

		-L: total length of the dogbone (z-dir)
		-W: total width of the dogbone (x-dir)
		-T: total thickness of the dogbone (y-dir)

		-l: inner length
		-w: inner width
		-T: inner thickness (assume thickness does not change)

		-b: dogbone padding thickness at 4 corners
		-R: fillet radius

	Assumptions:
		L = Nz * res
		W = Nx * res
		T = Ny * res

	Note: NOT (but close to) 45 degree from gage section to the dogbone

	Return:
		A numpy 3d array in '.npy' format that encodes phase of the microstructure.
		Normal grain = -1
		Void = num_grains+1

	Description:
		The dogbone is measured in the box (0, Nx), (0, Ny), (0, Nz)
		where Nz > Nx > Ny. In other words, length is in z-direction, 
		width is in x-direction, thickness is in y-direction.

		The origin is at (0,0,0).

		Conceptually, the geometry of the dogbone specimen is parameterized mainly in 2D, 
		followed by an extrusion in y-direction. Therefore, we will sketch the dogbone on 2D, 
		and extend for all y in (0, Ny) in the thickness direction. The dogbone is mainly defined by 2 rectangles:
			* (0, 0); (0, L); (W, L); (W, 0)
			* (W/2 +- w/2, L/2 +- l/2): (W/2 - w/2, L/2 - l/2)
										(W/2 - w/2, L/2 + l/2)
										(W/2 + w/2, L/2 + l/2)
										(W/2 + w/2, L/2 - l/2)
	


"""
# copied from geom_spk2dmsk.py
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
			break
	header = np.array(dumptxt[i+4].replace('\n','').replace('ITEM: ATOMS ', '').split(' '), dtype=str)
	d = np.loadtxt(dumpFileName, skiprows=9, dtype=int)
	num_grains = len(np.unique(d[:,1]))
	m = np.zeros([Nx, Ny, Nz])
	for i in range(len(d)):
		x = int(d[i,np.where(header=='x')[0][0]])
		y = int(d[i,np.where(header=='y')[0][0]])
		z = int(d[i,np.where(header=='z')[0][0]])
		grain_id = int(d[i,1]) # or d[i,2] -- both are the same
		m[x,y,z] = grain_id
		# print(f"finish ({x},{y}, {z})")
	return m, Nx, Ny, Nz, num_grains


m, Nx, Ny, Nz, num_grains = getDumpMs(dumpFileName)

# example from Tim'slides
L = 10000
W = 6000
l = 4000
w = 1000
b = 1000

p = np.ones([Nx, Ny, Nz]) * (-1)

# work in true coordinate system
for i in range(Nx):
	for j in range(Ny):
		for k in range(Nz):
			x, y, z = np.array(i,j,k) * res


np.save(outFileName, p)
