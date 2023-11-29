import numpy as np
import glob, os
from natsort import natsorted, ns # natural-sort
import pyvista
import argparse

import logging

level    = logging.INFO
format   = '  %(message)s'
handlers = [logging.FileHandler('seedVoid.log'), logging.StreamHandler()]

logging.basicConfig(level = level, format = format, handlers = handlers)


'''
This script seeds voids (1 pixel) into a tensile dogbone with a pre-defined percentage of porosity.

Example
-------
python3 seedVoid.py \
    --origGeomFileName spk_dump_12_out.geom \
    --percentage 1.5 \
    --phaseFileName phase_dump_12_out.npy

# dump 'voidSeeded_1pc_spk_dump_12_out.geom'

Parameters
----------
-g or --origGeomFileName: original .geom file without air padded
--percent: percentage of void in float format. For air: phase[x,y,z] = np.inf; else: -1 (default)
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

parser = argparse.ArgumentParser(description='')
parser.add_argument("-g" , "--origGeomFileName", help='original geom fileName', type=str, required=True)
parser.add_argument("-p" , "--phasephaseFileName", help='phase fileName', type=str, required=True)
parser.add_argument("-pc", "--percentage", help='percentage of void', type=float, required=True)

# Parse file names
args = parser.parse_args()
origGeomFileName = args.origGeomFileName # e.g. 'singleCrystal_res_50um.geom'
phaseFileName = args.phaseFileName
voidPercentage = args.percentage
outFileName = 'voidSeeded_' + origGeomFileName

Nx_grid, Ny_grid, Nz_grid, Nx_size, Ny_size, Nz_size, origGeom, headers = geom2npy(origGeomFileName)
phase = np.load(phaseFileName)

if np.all(np.array(phase.shape) == np.array(origGeom.shape)):
    print(f'Checkpoint: phase.shape() == origGeom.shape() passed.')
else:
    raise Exception('Error: phase.shape() != origGeom.shape()')

# Calculate number of void voxels to be inserted
numSolidVoxels = (~np.isinf(phase)).sum()
numVoidVoxels = np.floor(numSolidVoxels * percentage / 100).astype(int)

# Insert void voxels to phase
logging.info(f'Box shape = {phase.shape}')
logging.info(f'Sampling efficiency: {numSolidVoxels / np.prod(phase.shape)}')
logging.info(f'Total number of voxels = {np.prod(phase.shape)} voxels.')
logging.info(f'Number of solid voxels = {numSolidVoxels} voxels.')
logging.info(f'Number of air voxels = {np.prod(phase.shape) - numSolidVoxels} voxels.')
logging.info(f'Inserting {numVoidVoxels} voxels as voids.')
logging.info(f'Number of grains: {np.max(origGeom)-1}.')
logging.info(f'\n-------------------- NOTE --------------------\n')
logging.info(f'Indexing grain id:')
logging.info(f'Grain id for AIR: 1.')
logging.info(f'Grain id for VOIDS: from 2 to {numVoidVoxels+1}.')
logging.info(f'Grain id for SOLID: from {numVoidVoxels+2} to {numVoidVoxels+np.max(origGeom)}.')

def sampleLocation(Nx, Ny, Nz):
    """
    This function samples a random location in dogbone tensile specimen.
    """
    x = np.random.randint(low=0, high=Nx)
    y = np.random.randint(low=0, high=Ny)
    z = np.random.randint(low=0, high=Nz)
    return x, y, z

voidLocations = []

# Sample void locations
for i in range(numVoidVoxels):
    # Sample a solid voxel in the dogbone specimen
    x, y, z = sampleLocation(Nx_grid, Ny_grid, Nz_grid)
    while np.isinf(phase[x,y,z]):
        x, y, z = sampleLocation(Nx_grid, Ny_grid, Nz_grid)
    voidLocations += [[x,y,z]]

voidLocations = np.array(voidLocations)

# Increase grain id for solid
x, y, z = np.where(origGeom > 1)
'''
Index for grain id:
    ?=1: air
    2 < ? < numVoidVoxels+1: voids
    ? > numVoidVoxels+2: solid
'''
# Make a copy of origGeom and work on this copy
geom = np.copy(origGeom)
geom[x,y,z] += (numVoidVoxels+1)
# Insert voids
for i in range(numVoidVoxels):
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
# f.write('size    x %d    y %d    z %d\n' % (int(Nx), int(Ny), int(Nz)))
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

# Write orientations part of material.config
orientations = np.loadtxt('orientations.dat')
outFileName = 'material.config'
f = open(outFileName, 'w')

f.write('#############################################################################\n')
f.write('# Generated by seedVoid.py\n')
f.write('#############################################################################\n')
f.write('# Add <homogenization>, <crystallite>, and <phase> for a complete definition\n')
f.write('#############################################################################\n')
f.write('<texture>\n')

### NOTE:
# if void_id = 1, then grain ids translate to (i+2)
# if void_id = num_grains, then grain ids translate to (i+1)

f.write('[grain%d]\n' % (void_id))
f.write('(gauss) phi1 0   Phi 0    phi2 0   scatter 0.0   fraction 1.0 \n')

for i in range(num_grains):
    f.write('[grain%d]\n' % (i+2)) # assign grain id
    phi1, Phi, phi2 = orientations[i,:]
    f.write('(gauss) phi1 %.3f   Phi %.3f    phi2 %.3f   scatter 0.0   fraction 1.0 \n' % (phi1, Phi, phi2))

f.write('\n')
f.write('<microstructure>\n')

f.write('[grain%d]\n' % (void_id))
f.write('crystallite 1\n')
f.write('(constituent)   phase 2 texture %d fraction 1.0\n' % void_id)

for i in range(num_grains):
    f.write('[grain%d]\n' % (i+2)) # assign grain id
    f.write('crystallite 1\n')
    f.write('(constituent)   phase 1 texture %d fraction 1.0\n' % (i+2)) # assign grain id


f.close()

### diagnostics
print(f"Number of unique grains = {num_grains}")
elapsed = time.time() - t_start
print("geom_spk2dmsk.py: finished in {:5.2f} seconds.\n".format(elapsed), end="")
