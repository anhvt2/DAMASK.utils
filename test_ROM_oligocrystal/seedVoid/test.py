import numpy as np
import glob, os
from natsort import natsorted, ns # natural-sort
import random
import pyvista
import argparse
import time
import logging

level    = logging.INFO
format   = '  %(message)s'
logFileName = 'seedVoid.log'
os.system('rm -fv %s' % logFileName)
handlers = [logging.FileHandler(logFileName), logging.StreamHandler()]

logging.basicConfig(level = level, format = format, handlers = handlers)

t_start = time.time()
'''
This script seeds voids (of various morphology, sampled from DREAM.3D) into a tensile dogbone with a pre-defined voidPercentage of porosity.

Example
-------
python3 seedVoid.py \
    --origGeomFileName potts-12_3d.975.geom \
    --voidPercentage 1 \
    --voidDictionary voidEquiaxed.geom \
    --phaseFileName phase_dump_12_out.npy

# dump 'voidSeeded_1pc_spk_dump_12_out.geom'

Parameters
----------
-g or --origGeomFileName: original .geom file without air padded
--voidPercentage: voidPercentage of void in float format. For air: phase[x,y,z] = np.inf; else: -1 (default)
--voidDictionary: dictionary of void morphology
--phase: phase file name in .npy format

Return
------
voidSeeded_ + origGeomFileName: seeded .geom file with voids
'''

def delete(lst, to_delete):
    '''
    Recursively delete an element with content described by  'to_delete' variable
    https://stackoverflow.com/questions/53265275/deleting-a-value-from-a-list-using-recursion/
    Parameter
    ---------
    to_delete: content needs removing
    lst: list
    Return
    ------
    a list without to_delete element
    '''
    return [element for element in lst if element != to_delete]

def geom2npy(fileName):
    '''
    Read a .geom file and return a numpy array with meta-data
    '''
    fileHandler = open(fileName)
    txt = fileHandler.readlines()
    fileHandler.close()
    numSkippingLines = int(txt[0].split(' ')[0])+1 
    # Search for 'size' within header:
    for j in range(numSkippingLines):
        if 'grid' in txt[j]:
            cleanString = delete(txt[j].replace('\n', '').split(' '), '')
            Nx_grid = int(cleanString[2])
            Ny_grid = int(cleanString[4])
            Nz_grid = int(cleanString[6])
        if 'size' in txt[j]:
            cleanString = delete(txt[j].replace('\n', '').split(' '), '')
            Nx_size = float(cleanString[2])
            Ny_size = float(cleanString[4])
            Nz_size = float(cleanString[6])
    #
    geomBlock = txt[numSkippingLines:]
    geom = ''
    for i in range(len(geomBlock)):
        geom += geomBlock[i]
    #
    geom = geom.split(' ')
    geom = list(filter(('').__ne__, geom))
    geom = np.array(geom, dtype=int).reshape(Nz_grid, Ny_grid, Nx_grid).T
    headers = txt[:numSkippingLines] # also return headers
    return Nx_grid, Ny_grid, Nz_grid, Nx_size, Ny_size, Nz_size, geom, headers

origGeomFileName = 'potts-12_3d.975.geom' \
voidPercentage = 1
voidDictionary = 'voidEquiaxed.geom'
phaseFileName = 'phase_dump_12_out.npy'
outFileName = 'voidSeeded_%.3fpc_' % voidPercentage + origGeomFileName 

# Read from origGeomFileName
Nx_grid, Ny_grid, Nz_grid, Nx_size, Ny_size, Nz_size, origGeom, headers = geom2npy(origGeomFileName)
phase = np.load(phaseFileName)

if np.all(np.array(phase.shape) == np.array(origGeom.shape)):
    print(f'\nCheckpoint: phase.shape() == origGeom.shape() passed.\n')
else:
    raise Exception('\nError: phase.shape() != origGeom.shape().\n')

# Read void dictionary from voidDictionary
_, _, _, _, _, _, voidDict, _ = geom2npy(voidDictionary)

def sampleVoid(voidDict):
    '''
    This function reads a voidDict and randomly samples a local void based on its random indices. 
    For example:
    >>> xV
    array([0, 0, 0])
    >>> yV
    array([0, 0, 1])
    >>> zV
    array([0, 1, 0])
    '''
    # Sample void index
    randomIdx = random.choice(np.unique(voidDict))
    # Locate void
    xV, yV, zV = np.where(voidDict == randomIdx)
    # Left-center void location
    xV -= min(xV)
    yV -= min(yV)
    zV -= min(zV)
    return xV, yV, zV

# Calculate MINIMUM number of void voxels to be inserted
'''
The number of void voxels is at least minNumVoidVoxels. Seeding algorithms stop once the number of void crosses the threshold minNumVoidVoxels.
'''
numSolidVoxels = (~np.isinf(phase)).sum()
minNumVoidVoxels = np.floor(numSolidVoxels * voidPercentage / 100).astype(int)
# deprecate: numGrains = np.max(origGeom) - 1
numGrains = len(np.unique(origGeom))

# Insert void voxels to phase
logging.info(f'\n-------------------- COMMAND --------------------\n')
logging.info(f'python3 seedVoid.py \\')
logging.info(f'    --origGeomFileName {origGeomFileName} \\')
logging.info(f'    --voidPercentage {voidPercentage} \\')
logging.info(f'    --phaseFileName {phaseFileName}')
logging.info(f'\n')
logging.info(f'\n-------------------- INFORMATION --------------------\n')
logging.info(f'Box shape = {phase.shape}')
# logging.info(f'Sampling efficiency: {numSolidVoxels / np.prod(phase.shape)}')
# logging.info(f'Total number of voxels = {np.prod(phase.shape)} voxels.')
# logging.info(f'Number of solid voxels = {numSolidVoxels} voxels.')
# logging.info(f'Number of air voxels = {np.prod(phase.shape) - numSolidVoxels} voxels.')
# logging.info(f'Inserting {minNumVoidVoxels} voxels as voids.')
# logging.info(f'Number of grains: {numGrains}.')
# logging.info(f'\n-------------------- NOTE --------------------\n')
# logging.info(f'Indexing grain id:')
# logging.info(f'Grain id for AIR: 1.')
# logging.info(f'Grain id for VOIDS: from 2 to {minNumVoidVoxels+1}.')
# logging.info(f'Grain id for SOLID: from {minNumVoidVoxels+2} to {minNumVoidVoxels+np.max(origGeom)}.')

def sampleLocation(Nx, Ny, Nz):
    """
    This function samples a random location in dogbone tensile specimen.
    """
    x = np.random.randint(low=0, high=Nx)
    y = np.random.randint(low=0, high=Ny)
    z = np.random.randint(low=0, high=Nz)
    return x, y, z

# Initialize
voidLocations = []
totalNumVoidVoxels = 0
geom = np.copy(origGeom) # Make a copy of origGeom and work on this copy

# Insert/seed voids
while totalNumVoidVoxels < minNumVoidVoxels:
    # Sample a solid voxel in the dogbone specimen
    x, y, z = sampleLocation(Nx_grid, Ny_grid, Nz_grid)
    # Sample a void from voidDictionary
    xV, yV, zV = sampleVoid(voidDictionary)
    voidVoxels = len(xV)

    totalNumVoidVoxels += voidVoxels

# # Sample void locations
# for i in range(minNumVoidVoxels):
#     # Sample a solid voxel in the dogbone specimen
#     x, y, z = sampleLocation(Nx_grid, Ny_grid, Nz_grid)
#     while np.isinf(phase[x,y,z]):
#         x, y, z = sampleLocation(Nx_grid, Ny_grid, Nz_grid)
#     voidLocations += [[x,y,z]]

voidLocations = np.array(voidLocations)

# Increase grain id for solid
x, y, z = np.where(origGeom > 1)
'''
Index for grain id:
    ?=1: air
    2 < ? < minNumVoidVoxels+1: voids
    ? > minNumVoidVoxels+2: solid
'''
geom[x,y,z] += (minNumVoidVoxels+1)
# Insert voids
for i in range(minNumVoidVoxels):
    x, y, z = voidLocations[i]
    geom[x,y,z] = (i+2) # i=0 means geom[?,?,?] = 2

# Convert 3d numpy array to 1d flatten array
geom = geom.T.flatten()

# Write output
num_lines = int(np.floor(len(geom)) / 10)
num_elems_last_line = int(len(geom) % 10)

f = open(outFileName, 'w')
f.write('6       header\n')
f.write('# Generated by seedVoid.py\n')
f.write('grid    a %d    b %d    c %d\n' % (Nx_grid, Ny_grid, Nz_grid))
f.write('size    x %.3f    y %.3f    z %.3f\n' % (int(Nx_size), int(Ny_size), int(Nz_size))) 
f.write('origin    x 0.000    y 0.000    z 0.000\n')
f.write('homogenization  1\n')
f.write('microstructures %d\n' % (np.max(geom)))

for j in range(int(num_lines)):
    for k in range(10):
        idx = int(j * 10 + k)
        f.write('%10d' % int(geom[idx]))
    f.write('\n')

if num_elems_last_line > 0:
    for idx in range(-num_elems_last_line,0):
        f.write('%10d' % int(geom[idx]))

f.close()

# Write orientations as part of material.config
ubEulerAngles = np.array([360,180,360]) # upper bound
lbEulerAngles = np.array([0,0,0]) # lower bound
voidOrientations = np.random.rand(minNumVoidVoxels+1, 3) * (ubEulerAngles - lbEulerAngles) + lbEulerAngles # +1 to include air
solidOrientations = np.loadtxt('orientations.dat')
outFileName = 'material.config'
f = open(outFileName, 'w')

f.write('#############################################################################\n')
f.write('# Generated by seedVoid.py\n')
f.write('#############################################################################\n')
f.write('# Add <homogenization>, <crystallite>, and <phase> for a complete definition\n')
f.write('#############################################################################\n')
f.write('<texture>\n')

### NOTE:
# if air_id = 1, then grain ids translate to (i+2)
# if air_id = numGrains, then grain ids translate to (i+1)

air_id = 1
f.write('[grain%d]\n' % (air_id))
f.write('(gauss) phi1 0   Phi 0    phi2 0   scatter 0.0   fraction 1.0 \n')

for i in range(minNumVoidVoxels):
    f.write('[grain%d]\n' % (i+air_id+1)) # assign grain id
    phi1, Phi, phi2 = voidOrientations[i,:]
    f.write('(gauss) phi1 %.3f   Phi %.3f    phi2 %.3f   scatter 0.0   fraction 1.0 \n' % (phi1, Phi, phi2))

for i in range(numGrains+1):
    f.write('[grain%d]\n' % (i+minNumVoidVoxels+air_id+1)) # assign grain id
    phi1, Phi, phi2 = solidOrientations[i,:]
    f.write('(gauss) phi1 %.3f   Phi %.3f    phi2 %.3f   scatter 0.0   fraction 1.0 \n' % (phi1, Phi, phi2))

f.write('\n')
f.write('<microstructure>\n')
f.write('[grain%d]\n' % (air_id))
f.write('crystallite 1\n')
f.write('(constituent)   phase 2 texture %d fraction 1.0\n' % air_id)

for i in range(minNumVoidVoxels):
    f.write('[grain%d]\n' % (i+air_id+1)) # assign grain id
    f.write('crystallite 1\n')
    f.write('(constituent)   phase 2 texture %d fraction 1.0\n' % (i+air_id+1)) # assign grain id

for i in range(numGrains+1):
    f.write('[grain%d]\n' % (i+minNumVoidVoxels+air_id+1)) # assign grain id
    f.write('crystallite 1\n')
    f.write('(constituent)   phase 1 texture %d fraction 1.0\n' % (i+minNumVoidVoxels+air_id+1)) # assign grain id


f.close()

### diagnostics
elapsed = time.time() - t_start
logging.info("geom_spk2dmsk.py: finished in {:5.2f} seconds.\n".format(elapsed))
