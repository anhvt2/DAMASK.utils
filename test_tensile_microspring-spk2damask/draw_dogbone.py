

import numpy as np
import os, sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

"""
	This script supports a visualization of a dogbone specimen given these following parameters:

		-L: total length of the dogbone (z-dir)
		-W: total width of the dogbone (x-dir)
		-T: total thickness of the dogbone (y-dir)

		-l: inner length
		-w: inner width
		-T: inner thickness (assume thickness does not change)

		-b: dogbone padding thickness at 4 corners
		-R: fillet radius

		Conceptually, the geometry of the dogbone specimen is parameterized mainly in 2D, 
		followed by an extrusion in y-direction. Therefore, we will sketch the dogbone on 2D, 
		and extend for all y in (0, Ny) in the thickness direction. The dogbone is mainly defined by 2 rectangles:
			* (0, 0); (0, L); (W, L); (W, 0)
			* (W/2 +- w/2, L/2 +- l/2): (W/2 - w/2, L/2 - l/2)
										(W/2 - w/2, L/2 + l/2)
										(W/2 + w/2, L/2 + l/2)
										(W/2 + w/2, L/2 - l/2)

"""

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



# b = L/2. - l/2. - (W/2. - w/2.) * np.round(np.tan(45 * 2*math.pi/360), 8)

# plt.figure()
fig, ax = plt.subplots()

def line2d(p1, p2):
	# This function plots line between 2 points 'p1' and 'p2'
	plt.plot((p1[0], p2[0]), (p1[1], p2[1]), c='k')

def line2dfillet(p1, p2):
	plt.plot((p1[0], p2[0]), (p1[1], p2[1]), c='b', linestyle=':')

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

# v  = np.atleast_2d([3,5]).T
# c  = np.atleast_2d([-4,5]).T
# c0 = np.atleast_2d([0,-5]).T
# proj_c = np.matmul(np.matmul(v,v.T)/np.matmul(v.T, v), c) + np.matmul(np.eye(2) - np.matmul(v,v.T) / np.matmul(v.T,v), c0)
# print(f"{proj_c}") # [[3.35294118], [0.58823529]]


# line2d([0,0], [0, L])
# line2d([0, L], [W, L])
# line2d([W, L], [W,0])
# line2d([W,0], [0,0])

### plot line segments for dogbone
line2d([0,0], [0, b])
line2d([0, b], [W/2.-w/2., L/2.-l/2.])
line2d([W/2.-w/2., L/2.-l/2.], [W/2.-w/2., L/2.+l/2.])
line2d([W/2.-w/2., L/2.+l/2.], [0, L-b])
line2d([0, L-b], [0, L])
line2d([0, L], [W, L])
line2d([W, L], [W, L-b])
line2d([W, L-b], [W/2.+w/2., L/2.+l/2.])
line2d([W/2.+w/2., L/2.+l/2.], [W/2.+w/2., L/2.-l/2.])
line2d([W/2.+w/2., L/2.-l/2.], [W, b])
line2d([W, b], [W,0])
line2d([W,0], [0,0])

### plot fillet intersecting line segment to determine the center of the circle
line2dfillet([0, L-b-R/np.cos(alpha)], [W/2.-w/2., L/2.+l/2.-R/np.cos(alpha)])
line2dfillet([W/2.-w/2.-R, L/2.+l/2], [W/2.-w/2.-R, L/2.-l/2])
line2dfillet([0, b+R/np.cos(alpha)], [W/2.-w/2., L/2.-l/2+R/np.cos(alpha)])
line2dfillet([W, L-b-R/np.cos(alpha)], [W/2.+w/2., L/2.+l/2.-R/np.cos(alpha)])
line2dfillet([W/2.+w/2.+R, L/2.+l/2.], [W/2.+w/2.+R, L/2.-l/2.])
line2dfillet([W/2.+w/2., L/2.-l/2.+R/np.cos(alpha)], [W,b+R/np.cos(alpha)])

### plot (1) fillet circle and (2) projection of fillet circle onto the line segment to determine tangential points
# for every nw/sw/ne/se section, plot two of them
fillet_c_nw = intersection(line([0, L-b-R/np.cos(alpha)], [W/2.-w/2., L/2.+l/2.-R/np.cos(alpha)]), line([W/2.-w/2.-R, L/2.+l/2], [W/2.-w/2.-R, L/2.-l/2]))
print(f"NW fillet center = {fillet_c_nw}")
proj_c_nw_1 = projectPt2Line(np.atleast_2d(fillet_c_nw).T, [0, L-b], [W/2.-w/2., L/2.+l/2.]).ravel()
plt.plot(proj_c_nw_1[0], proj_c_nw_1[1], '*', c='r', markersize=10)
proj_c_nw_2 = projectPt2Line(np.atleast_2d(fillet_c_nw).T, [W/2.-w/2., L/2.+l/2], [W/2.-w/2., L/2.-l/2]).ravel()
plt.plot(proj_c_nw_2[0], proj_c_nw_2[1], '*', c='r', markersize=10)
patch_circle_nw = plt.Circle(fillet_c_nw, R, fill='False', color='tab:blue')
ax.add_artist(patch_circle_nw)

fillet_c_sw = intersection(line([0, b+R/np.cos(alpha)], [W/2.-w/2., L/2.-l/2.+R/np.cos(alpha)]), line([W/2.-w/2.-R, L/2.+l/2], [W/2.-w/2.-R, L/2.-l/2]))
print(f"SW fillet center = {fillet_c_sw}")
proj_c_sw_1 = projectPt2Line(np.atleast_2d(fillet_c_sw).T, [W/2.-w/2., L/2.+l/2], [W/2.-w/2., L/2.-l/2]).ravel()
plt.plot(proj_c_sw_1[0], proj_c_sw_1[1], '*', c='r', markersize=10)
proj_c_sw_2 = projectPt2Line(np.atleast_2d(fillet_c_sw).T, [0, b], [W/2.-w/2., L/2.-l/2]).ravel()
plt.plot(proj_c_sw_2[0], proj_c_sw_2[1], '*', c='r', markersize=10)
patch_circle_sw = plt.Circle(fillet_c_sw, R, fill='False', color='tab:orange')
ax.add_artist(patch_circle_sw)

fillet_c_ne = intersection(line([W, L-b-R/np.cos(alpha)], [W/2.+w/2., L/2.+l/2.-R/np.cos(alpha)]), line([W/2.+w/2.+R, L/2.+l/2.], [W/2.+w/2.+R, L/2.-l/2.]))
print(f"NE fillet center = {fillet_c_ne}")
proj_c_ne_1 = projectPt2Line(np.atleast_2d(fillet_c_ne).T, [W, L-b], [W/2.+w/2., L/2.+l/2.]).ravel()
plt.plot(proj_c_ne_1[0], proj_c_ne_1[1], '*', c='r', markersize=10)
proj_c_ne_2 = projectPt2Line(np.atleast_2d(fillet_c_ne).T, [W/2.+w/2., L/2.+l/2.], [W/2.+w/2., L/2.-l/2.]).ravel()
plt.plot(proj_c_ne_2[0], proj_c_ne_2[1], '*', c='r', markersize=10)
patch_circle_ne = plt.Circle(fillet_c_ne, R, fill='False', color='tab:purple')
ax.add_artist(patch_circle_ne)

fillet_c_se = intersection(line([W/2.+w/2.+R, L/2.+l/2.], [W/2.+w/2.+R, L/2.-l/2.]), line([W/2.+w/2., L/2.-l/2.+R/np.cos(alpha)], [W,b+R/np.cos(alpha)]))
print(f"SE fillet center = {fillet_c_se}")
proj_c_se_1 = projectPt2Line(np.atleast_2d(fillet_c_se).T, [W/2.+w/2., L/2.+l/2.], [W/2.+w/2., L/2.-l/2.]).ravel()
plt.plot(proj_c_se_1[0], proj_c_se_1[1], '*', c='r', markersize=10)
proj_c_se_2 = projectPt2Line(np.atleast_2d(fillet_c_se).T, [W/2.+w/2., L/2.-l/2.], [W,b]).ravel()
plt.plot(proj_c_se_2[0], proj_c_se_2[1], '*', c='r', markersize=10)
patch_circle_se = plt.Circle(fillet_c_se, R, fill='False', color='tab:cyan')
ax.add_artist(patch_circle_se)


plt.axis('equal')
# plt.xlim([-1, L+1])
# plt.ylim([-1, W+1])

### construct p matrix 2d



plt.show()

