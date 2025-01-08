
import numpy as np
import os, time, glob

"""
This script computes the (true AND global) porosities in the gauge section area of the HTT dogbone specimen

$ cat gaugeFilter.txt 
    51*50 <= x <= 69*50 and 2*50 <= y <= 21*50 and 70*50 <= z <= 130*50

NOTE:
    Due to some bugs, the air voxels are indexed as 1 and 2, instead of just 1. In order to scope with this bug, increase the threshold to 3. Debug ongoing.
"""
porosity = np.loadtxt('../seedVoid/porosity.txt')
grainInfo = np.loadtxt('grainInfo.dat')
geom = np.load(glob.glob('htt*.npy')[0])

runIdx = int(os.getcwd().split('run-')[-1].split('-')[0]) - 1 # reset start index = 0
targetPoro = porosity[runIdx] / 1e2 # convert from percentage

# Global: Count how many void voxels in the WHOLE domain
numSolidVoxels = np.where((geom >= grainInfo[3]))[0].shape[0]
numVoidVoxels = np.where((geom >= 3) & (geom <= grainInfo[2]))[0].shape[0]
globalPoro = numVoidVoxels / numSolidVoxels

# Local: Count how many void voxels in the GAUGE domain
gaugeGeom = geom[51:69+1, 2:21+1, 70:130+1]
numSolidVoxels = np.where((gaugeGeom >= grainInfo[3]))[0].shape[0]
numVoidVoxels = np.where((gaugeGeom >= 3) & (gaugeGeom <= grainInfo[2]))[0].shape[0]
localPoro = numVoidVoxels / numSolidVoxels

print(f'In {os.getcwd().split('/')[-1]}:')
print(f'(whole specimen) Global porosity = {globalPoro:<.4f}')
print(f'(gauge section) Local porosity = {localPoro:<.4f}')
print(f'(whole specimen) Target porosity = {targetPoro:<.4f}')
print('\n')

f = open('porosity.txt', 'w')
f.write(f'{globalPoro:<.4f}\n')
f.write(f'{localPoro:<.4f}\n')
f.write(f'{targetPoro:<.4f}\n')
f.close()

