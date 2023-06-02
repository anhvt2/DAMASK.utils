

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
	# Ax + By = C for 'p1' and 'p2'
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

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

# line2d([0,0], [0, L])
# line2d([0, L], [W, L])
# line2d([W, L], [W,0])
# line2d([W,0], [0,0])

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

line2dfillet([0, L-b-R/np.cos(alpha)], [W/2.-w/2., L/2.+l/2.-R/np.cos(alpha)])
line2dfillet([W/2.-w/2.-R, L/2.+l/2], [W/2.-w/2.-R, L/2.-l/2])
line2dfillet([0, b+R/np.cos(alpha)], [W/2.-w/2., L/2.-l/2+R/np.cos(alpha)])
line2dfillet([W, L-b-R/np.cos(alpha)], [W/2.+w/2., L/2.+l/2.-R/np.cos(alpha)])
line2dfillet([W/2.+w/2.+R, L/2.+l/2.], [W/2.+w/2.+R, L/2.-l/2.])
line2dfillet([W/2.+w/2., L/2.-l/2.+R/np.cos(alpha)], [W,b+R/np.cos(alpha)])

fillet_c_nw = intersection(line([0, L-b-R/np.cos(alpha)], [W/2.-w/2., L/2.+l/2.-R/np.cos(alpha)]), line([W/2.-w/2.-R, L/2.+l/2], [W/2.-w/2.-R, L/2.-l/2]))
print(f"NW fillet center = {fillet_c_nw}")
patch_circle_nw = plt.Circle(fillet_c_nw, R, fill='False', color='tab:blue')
ax.add_artist(patch_circle_nw)

fillet_c_sw = intersection(line([0, b+R/np.cos(alpha)], [W/2.-w/2., L/2.-l/2.+R/np.cos(alpha)]), line([W/2.-w/2.-R, L/2.+l/2], [W/2.-w/2.-R, L/2.-l/2]))
print(f"SW fillet center = {fillet_c_sw}")
patch_circle_sw = plt.Circle(fillet_c_sw, R, fill='False', color='tab:orange')
ax.add_artist(patch_circle_sw)

fillet_c_ne = intersection(line([W, L-b-R/np.cos(alpha)], [W/2.+w/2., L/2.+l/2.-R/np.cos(alpha)]), line([W/2.+w/2.+R, L/2.+l/2.], [W/2.+w/2.+R, L/2.-l/2.]))
print(f"NE fillet center = {fillet_c_ne}")
patch_circle_ne = plt.Circle(fillet_c_ne, R, fill='False', color='tab:purple')
ax.add_artist(patch_circle_ne)

fillet_c_se = intersection(line([W/2.+w/2.+R, L/2.+l/2.], [W/2.+w/2.+R, L/2.-l/2.]), line([W/2.+w/2., L/2.-l/2.+R/np.cos(alpha)], [W,b+R/np.cos(alpha)]))
print(f"SE fillet center = {fillet_c_se}")
patch_circle_se = plt.Circle(fillet_c_se, R, fill='False', color='tab:cyan')
ax.add_artist(patch_circle_se)


plt.axis('equal')
# plt.xlim([-1, L+1])
# plt.ylim([-1, W+1])
plt.show()