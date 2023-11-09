
''' 
    This script reads a .geom file and prints the locations of the gauge in tensile bar for postResults in DAMASK.

    Example
    -------
    python3 findGaugeLocations.py --geom spk_dump_12_out.npy

    Parameters
    ----------
    --geom: geometry in 3d numpy array (if don't have, can run geom2npy.py)

    Return
    ------
    gaugeFilter.txt for filtering x and z directions
'''

import numpy as np
import argparse
# import glob


parser = argparse.ArgumentParser()
parser.add_argument("-g", "--geom", type=str, required=True)
args = parser.parse_args()

# fileName = glob.glob('*.geom')[0]
fileName = args.geom
ms = np.load(fileName)
Nx, Ny, Nz = ms.shape

# Take a random 2d slice of ms
slice2d = ms[:,0,:] - 1 # y-direction is extrusion direction - not important, so any would work
slice2d = slice2d.astype(bool)

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

f = open('gaugeFilter.txt', 'w')
f.write('%d <= x <= %d and %d <= z <= %d\n' % (np.min(xLocs), np.max(xLocs), np.min(zLocs), np.max(zLocs)))
f.close()
