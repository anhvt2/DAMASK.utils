import numpy as np
import glob, os
from natsort import natsorted, ns # natural-sort
import pyvista
import argparse

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
print(f'Sampling efficiency: {numSolidVoxels / np.prod(phase.shape)}')

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


