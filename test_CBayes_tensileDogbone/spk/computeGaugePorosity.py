
import numpy as np
import os, time, glob

"""
This script computes the (true AND global) porosities in the gauge section area of the HTT dogbone specimen

$ cat gaugeFilter.txt 
    51*50 <= x <= 69*50 and 2*50 <= y <= 21*50 and 70*50 <= z <= 130*50
"""

grainInfo = np.loadtxt('grainInfo.dat')
geom = np.load(glob.glob('htt*.npy')[0])

# Global: Count how many void voxels in the WHOLE domain
numSolidVoxels = np.where((geom >= grainInfo[3]))[0].shape[0]
numVoidVoxels = np.where((geom >= grainInfo[1]) & (geom <= grainInfo[2]))[0].shape[0]
globalPoro = numVoidVoxels / numSolidVoxels

# Local: Count how many void voxels in the GAUGE domain
gaugeGeom = geom[51:69+1, 2:21+1, 70:130+1]
numSolidVoxels = np.where((gaugeGeom >= grainInfo[3]))[0].shape[0]
numVoidVoxels = np.where((gaugeGeom >= grainInfo[1]) & (gaugeGeom <= grainInfo[2]))[0].shape[0]
localPoro = numVoidVoxels / numSolidVoxels

print(f'(whole specimen) Global porosity = {globalPoro:<.4f}')
print(f'(gauge section) Local porosity = {localPoro:<.4f}')

f = open('porosity.txt', 'w')
f.write(f'{globalPoro:<.4f}\n')
f.write(f'{localPoro:<.4f}\n')
f.close()

