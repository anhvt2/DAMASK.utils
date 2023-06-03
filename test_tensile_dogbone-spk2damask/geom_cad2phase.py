
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
res = args.resolution # 50

outFileName = 'phase_' + dumpFileName.replace('.','_') + '.npy'


"""
	How to use: 
		python3 geom_cad2phase.py -r 50 -d 'dump.12.out' --dogbone-geom L W T l w b R

	Note:
		A lot of this script is adopted from 'draw_dogbone.py' in 2d. 
		It is usually a good idea to visualize the dogbone geometry first before getting hand dirty.

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
w = 1000
T = w
l = 4*w
b = w
R = w

def rad2deg(alpha):
	return alpha*360/(2*math.pi)

def deg2rad(alpha):
	return alpha/360*(2*math.pi)

alpha = np.arctan( (L/2.-l/2.-b) / (W/2.-w/2.) )
print(f"Corner degree = {rad2deg(alpha)}")

# https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
def line(p1, p2):
	# This function computes the coefs (A,B,C) s.t. Ax + By = C for 'p1' and 'p2'
	# or Ax + By -C = 0
	A = (p1[1] - p2[1])
	B = (p2[0] - p1[0])
	C = (p1[0]*p2[1] - p2[0]*p1[1])
	# parameterized in term of B: 
	# 	upper-half plane mean y > (C-Ax)/B and lower-half plane mean y < (C-Ax)/B
	# 	normal vector (A,B) always points upward (B>0 regardless of A) 
	if B > 0:
		return A, B, -C
	else:
		return -A, -B, +C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def projectPt2Line(c, p1, p2):
	# https://math.stackexchange.com/questions/62633/orthogonal-projection-of-a-point-onto-a-line
	# This function	projects the point 'c' (supposed to be the fillet center) 
	# to a line (parameterized by 'p1' and 'p2')
	A,B,C = line(p1, p2) # Ax + By - C = 0
	if B != 0:
		c0 = np.atleast_2d([0, C/B]).T # c0: an arbitrary point on the line Ax + By - C = 0
	else:
		c0 = np.atleast_2d([C/A, 0]).T # c0: an arbitrary point on the line Ax + By - C = 0
	v = np.atleast_2d([-B, A]).T # a column vector normal to the normal vector (A,B)
	proj_c = np.matmul(np.matmul(v,v.T)/np.matmul(v.T, v), c) + np.matmul(np.eye(2) - np.matmul(v,v.T) / np.matmul(v.T,v), c0)
	return proj_c

p = np.ones([Nx, Ny, Nz]) * (-1)

# calculate the position of the fillet center & its projections
fillet_c_nw = intersection(line([0, L-b-R/np.cos(alpha)], [W/2.-w/2., L/2.+l/2.-R/np.cos(alpha)]), line([W/2.-w/2.-R, L/2.+l/2], [W/2.-w/2.-R, L/2.-l/2]))
proj_c_nw_1 = projectPt2Line(np.atleast_2d(fillet_c_nw).T, [0, L-b], [W/2.-w/2., L/2.+l/2.]).ravel()
proj_c_nw_2 = projectPt2Line(np.atleast_2d(fillet_c_nw).T, [W/2.-w/2., L/2.+l/2], [W/2.-w/2., L/2.-l/2]).ravel()
fillet_c_sw = intersection(line([0, b+R/np.cos(alpha)], [W/2.-w/2., L/2.-l/2.+R/np.cos(alpha)]), line([W/2.-w/2.-R, L/2.+l/2], [W/2.-w/2.-R, L/2.-l/2]))
proj_c_sw_1 = projectPt2Line(np.atleast_2d(fillet_c_sw).T, [W/2.-w/2., L/2.+l/2], [W/2.-w/2., L/2.-l/2]).ravel()
proj_c_sw_2 = projectPt2Line(np.atleast_2d(fillet_c_sw).T, [0, b], [W/2.-w/2., L/2.-l/2]).ravel()
fillet_c_ne = intersection(line([W, L-b-R/np.cos(alpha)], [W/2.+w/2., L/2.+l/2.-R/np.cos(alpha)]), line([W/2.+w/2.+R, L/2.+l/2.], [W/2.+w/2.+R, L/2.-l/2.]))
proj_c_ne_1 = projectPt2Line(np.atleast_2d(fillet_c_ne).T, [W, L-b], [W/2.+w/2., L/2.+l/2.]).ravel()
proj_c_ne_2 = projectPt2Line(np.atleast_2d(fillet_c_ne).T, [W/2.+w/2., L/2.+l/2.], [W/2.+w/2., L/2.-l/2.]).ravel()
fillet_c_se = intersection(line([W/2.+w/2.+R, L/2.+l/2.], [W/2.+w/2.+R, L/2.-l/2.]), line([W/2.+w/2., L/2.-l/2.+R/np.cos(alpha)], [W,b+R/np.cos(alpha)]))
proj_c_se_1 = projectPt2Line(np.atleast_2d(fillet_c_se).T, [W/2.+w/2., L/2.+l/2.], [W/2.+w/2., L/2.-l/2.]).ravel()
proj_c_se_2 = projectPt2Line(np.atleast_2d(fillet_c_se).T, [W/2.+w/2., L/2.-l/2.], [W,b]).ravel()

# work in true coordinate system
for i in range(Nx):
	for j in range(Ny):
		for k in range(Nz):
			x, y, z = np.array(i,j,k) * res
			line2d([0, b], [W/2.-w/2., L/2.-l/2.])
			
			line2d([W/2.-w/2., L/2.+l/2.], [0, L-b])



np.save(outFileName, p)
