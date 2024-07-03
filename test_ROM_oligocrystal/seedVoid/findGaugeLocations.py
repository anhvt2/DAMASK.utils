
''' 
	This script reads a .geom file and prints the locations of the gauge in tensile bar for postResults in DAMASK.

	Example
	-------
	python3 findGaugeLocations.py --geom spk_dump_12_out.npy
	python3 findGaugeLocations.py --geom padded_spk_dump_12_out.npy
	python3 findGaugeLocations.py --geom padded_spk_dump_12_out.npy --resolution 50

	Parameters
	----------
	--geom: geometry in 3d numpy array (if don't have, can run geom2npy.py)

	Return
	------
	gaugeFilter.txt for filtering x and z directions
'''

import numpy as np
import argparse
import glob, os

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--geom", type=str, required=True)
parser.add_argument("-r", "--resolution", type=int, required=True) # resolution, e.g. 50um/voxel
args = parser.parse_args()

fileName = args.geom
resolution = args.resolution

msFileName = fileName[:-4] + '.npy' # fileName.split('.')[0] + '.npy'
if os.path.exists(msFileName):
    print(f"\nCheckpoint: ms {msFileName} .npy exists. Proceed forward.")
else:
    raise Exception(f"\n{msFileName} does not exist. Please run geom2npy.py to convert .geom to .npy array.") 
ms = np.load(msFileName)
Nx, Ny, Nz = ms.shape


# Take centered 2d slice of ms
slice2d = ms[:, int(np.floor(Ny/2)),:] - 1
slice2d = slice2d.astype(bool) # for polycrystalline - void id = 1

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
segment = segment.astype(bool) # for polycrystalline - void id = 1
yLocs = np.where(segment == np.max(segment))

# Diagnostics
print(f'{np.min(xLocs)} <= x <= {np.max(xLocs)}')
print(f'{np.min(yLocs)} <= y <= {np.max(yLocs)}')
print(f'{np.min(zLocs)} <= z <= {np.max(zLocs)}')

f = open('gaugeFilter.txt', 'w')
f.write('%d*%d <= x <= %d*%d and %d*%d <= y <= %d*%d and %d*%d <= z <= %d*%d\n' % 
    (   np.min(xLocs), resolution, np.max(xLocs), resolution, 
        np.min(yLocs), resolution, np.max(yLocs), resolution, 
        np.min(zLocs), resolution, np.max(zLocs), resolution))
f.close()
