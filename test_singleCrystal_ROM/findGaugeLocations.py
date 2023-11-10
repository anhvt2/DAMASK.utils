
''' 
    This script reads a .geom file and prints the locations of the gauge in tensile bar for postResults in DAMASK
'''

import numpy as np
import glob, os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--geom", type=str, required=True)
args = parser.parse_args()

fileName = args.geom
msFileName = fileName.split('.')[0] + '.npy'
if os.path.exists(msFileName):
    print(f"\nCheckpoint: ms {msFileName} .npy exists. Proceed forward.")
else:
    raise Exception(f"\n{msFileName} does not exist. Please run geom2npy.py to convert .geom to .npy array.") 
ms = np.load(msFileName)
Nx, Ny, Nz = ms.shape

# Take centered 2d slice of ms
slice2d = ms[:, int(np.floor(Ny/2)),:] - 1

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

'''
Find number of air voxels and bound its regions by tracing a ray at center
'''
segment = ms[int(np.floor(Nx/2)),:,int(np.floor(Nz/2))] - 1
yLocs = np.where(segment == np.max(segment))

# Diagnostics
print(f'{np.min(xLocs)} <= x <= {np.max(xLocs)}')
print(f'{np.min(yLocs)} <= y <= {np.max(yLocs)}')
print(f'{np.min(zLocs)} <= z <= {np.max(zLocs)}')

# Write gaugeFilter.txt

f = open('gaugeFilter.txt', 'w')
f.write('%d <= x <= %d and %d <= y <= %d and %d <= z <= %d\n' % 
    (   np.min(xLocs), np.max(xLocs), 
        np.min(yLocs), np.max(yLocs), 
        np.min(zLocs), np.max(zLocs)))
f.close()
