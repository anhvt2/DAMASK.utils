
''' 
    This script reads a .geom file and prints the locations of the gauge in tensile bar for postResults in DAMASK
'''

import numpy as np
import glob

fileName = glob.glob('*.geom')[0]
ms = np.load(fileName.split('.')[0] + '.npy')
Nx, Ny, Nz = ms.shape

# Take a random 2d slice of ms
slice2d = ms[:,0,:] - 1 # y-direction is extrusion direction - not important, so any would work

# Assume void id is '1'

'''
Find the gauge location in 'z' direction by summing along the 'x' direction:
The gauge locations is determined to the region that has the LEAST materials in 'z' direction.
'''
for i in range(Nx):
    slice2dz = np.sum(slice2d,axis=0)
    zLocs = np.where(slice2dz == np.min(slice2dz))

'''
Find the gauge location in 'x' direction by summing along the 'z' direction:
The gauge locations is determined to the region that has the MOST materials in 'x' direction.
'''

for j in range(Nz):
    slice2dx = np.sum(slice2d,axis=1)
    xLocs = np.where(slice2dx == np.max(slice2dx))

print(f'{np.min(xLocs)} <= x <= {np.max(xLocs)}')
print(f'{np.min(zLocs)} <= z <= {np.max(zLocs)}')
