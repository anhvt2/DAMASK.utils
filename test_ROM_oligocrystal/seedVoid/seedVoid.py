import numpy as np
import glob, os
from natsort import natsorted, ns # natural-sort
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist,squareform
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
    --voidPercentage 3 \
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

def renumerate(geom, startIndex=0, cluster=False, excludeList=None):
    ''' 
    This function renumerates so that the DEFAULT grain index starts at ZERO (0) and (gradually) increases by 1. 
    Input
    -----
        3d npy array
        startIndex: starting index, usually at 0 or 1
        cluster: 'False' or 'True'
            * True: perform clustering algorithm using DBSCAN then renumerate
            * False: only renumerate
    Output
    ------
        3d npy array
    '''
    grainIdxList = np.sort(np.unique(geom))

    # Remove grains from excludeList from being segmented, e.g. air
    if excludeList is not None:
        grainIdxList = np.setdiff1d(grainIdxList, excludeList)

    renumeratedGeom = np.copy(geom) # make a deep copy
    maxGrainId = np.max(grainIdxList)
    if cluster==True:
        # Perform clustering algorithm to decluster many grains with same grain id: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
        for i in range(len(grainIdxList)):
            grainIdx = grainIdxList[i]
            x, y, z = np.where(geom==grainIdx)
            X = np.hstack((np.atleast_2d(x).T, np.atleast_2d(y).T, np.atleast_2d(z).T))
            clustering = DBSCAN(eps=2, min_samples=1).fit(X)
            # Relabel grainId for every pixels needed relabel: Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.
            clustering.labels_ -= np.min(clustering.labels_) # re-start at 0: no noisy samples
            for j in range(clustering.labels_.shape[0]):
                renumeratedGeom[x[j],y[j],z[j]] = maxGrainId+clustering.labels_[j]+startIndex+1
            # Print diagnostics
            logging.info(f'clustering.labels_ = {clustering.labels_}')
            logging.info(f'np.min(clustering.labels_) = {np.min(clustering.labels_)}')
            logging.info(f'np.max(clustering.labels_) = {np.max(clustering.labels_)}')
            logging.info(f'maxGrainId = {maxGrainId}')
            logging.info(f'renumerate(): Segregating grains from grainId {grainIdx} to [{maxGrainId+np.min(clustering.labels_)+startIndex+1}, {maxGrainId+np.max(clustering.labels_)+startIndex+1}].')
            logging.info(f'\n')
            # Update maxGrainId
            # maxGrainId = np.max(renumeratedGeom)+np.max(clustering.labels_)+startIndex+1 # debug
            # maxGrainId += np.max(clustering.labels_)+startIndex+1 # debug
            maxGrainId = np.max(renumeratedGeom) # debug
        # run vanilla renumerate()
        grainIdxList = np.sort(np.unique(renumeratedGeom))
        for i in range(len(grainIdxList)):
            grainIdx = grainIdxList[i]
            x, y, z = np.where(geom==grainIdx)    
            logging.info(f'renumerate(): Mapping grain id from {grainIdx} to {startIndex+i}.')
            for j in range(len(x)):
                renumeratedGeom[x[j],y[j],z[j]] = i+startIndex
    else:
        # run vanilla renumerate() without clustering grains
        for i in range(len(grainIdxList)):
            grainIdx = grainIdxList[i]
            x, y, z = np.where(geom==grainIdx)    
            logging.info(f'renumerate(): Mapping grain id from {grainIdx} to {startIndex+i}.')
            for j in range(len(x)):
                renumeratedGeom[x[j],y[j],z[j]] = i+startIndex
    return renumeratedGeom

# Set up parser
parser = argparse.ArgumentParser(description='')
parser.add_argument("-g" , "--origGeomFileName", help='original geom fileName', type=str, required=True)
parser.add_argument("-p" , "--phaseFileName", help='phase fileName', type=str, required=True)
parser.add_argument("-v" , "--voidDictionary", help='void dictionary', type=str, required=True)
parser.add_argument("-pc", "--voidPercentage", help='voidPercentage of void', type=float, required=True)

# Parse file names
args = parser.parse_args()
origGeomFileName = args.origGeomFileName # e.g. 'singleCrystal_res_50um.geom'
phaseFileName = args.phaseFileName
voidDictionary = args.voidDictionary
voidPercentage = args.voidPercentage
outFileName = 'voidSeeded_%.3fpc_' % voidPercentage + origGeomFileName 

# Read from origGeomFileName
Nx_grid, Ny_grid, Nz_grid, Nx_size, Ny_size, Nz_size, origGeom, headers = geom2npy(origGeomFileName)
phase = np.load(phaseFileName)

if np.all(np.array(phase.shape) == np.array(origGeom.shape)):
    print(f'\nCheckpoint: phase.shape() == origGeom.shape() passed.\n')
else:
    raise Exception('\nError: phase.shape() != origGeom.shape().\n')

# Read void dictionary from voidDictionary
_, _, _, _, _, _, voidDictGeom, _ = geom2npy(voidDictionary)

def sampleVoid(voidDictGeom):
    '''
    This function reads a voidDict and randomly samples a local void based on its random indices. 
    For example:
    >>> xV
    array([0, 0, 0])
    >>> yV
    array([0, 0, 1])
    >>> zV
    array([0, 1, 0])
    Input
    -----
    voidDict: a .npy array
    Output
    ------
    3 arrays (x,y,z) of locations
    '''
    # Sample void index
    randomIdx = random.choice(np.unique(voidDictGeom))
    # Locate void
    xV, yV, zV = np.where(voidDictGeom == randomIdx)
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

def sampleLocation(Nx, Ny, Nz):
    """
    This function samples a uniformly distributed random location in dogbone tensile specimen.
    """
    x = np.random.randint(low=0, high=Nx)
    y = np.random.randint(low=0, high=Ny)
    z = np.random.randint(low=0, high=Nz)
    return x, y, z

# Initialize
totalNumVoidVoxels = 0
solidIdx = 2 # minimum = 2
geom = renumerate(np.copy(origGeom), cluster=True) + solidIdx # make a deep copy of origGeom, sindex starts at solidIdx
numVoids = 0
if solidIdx < 2:
    raise ValueError('solidIdx must be at least 2.')

# Insert/seed voids
'''
Index for grainIDs:
    ?=1: air
    2 < ? < solidIdx: voids (or more precisely, 2 < ? <= numVoids)
    ? >= solidIdx: solid
'''
while totalNumVoidVoxels < minNumVoidVoxels:
    tmpVoidIdx = 1
    # Sample a solid voxel in the dogbone specimen
    x, y, z = sampleLocation(Nx_grid, Ny_grid, Nz_grid)
    # Sample a void from voidDictionary
    xV, yV, zV = sampleVoid(voidDictGeom)
    voidVoxels = len(xV)
    # Assign void id (if possible -- passed if not)
    boolVoidSeed = False
    for i in range(len(xV)):
        try:
            # Criteria: do not assign void on the top of void
            if (x+xV[i] <= Nx_grid) and (y+yV[i] <= Ny_grid) and (z+zV[i] <= Nz_grid) and geom[x+xV[i], y+yV[i], z+zV[i]] > solidIdx:
                geom[x+xV[i], y+yV[i], z+zV[i]] = tmpVoidIdx
                boolVoidSeed = True
        except:
            # print('Warning: out-of-bounds error. Temporarily skip assigning void.')
            pass
    if boolVoidSeed:
        # Increase void counter
        totalNumVoidVoxels += voidVoxels
        # Bump grainID by 1, leave space for assigning tmpVoidIdx in the future
        geom += 1
        # Track solidIdx
        solidIdx += 1
        # Track number of voids
        numVoids += 1
        # Print diagnostics
        print(f'Finished seeding {totalNumVoidVoxels} out of {minNumVoidVoxels} voxels...')

# Assign AIR (TODO: is there a more efficient implementation?)
for i in range(Nx_grid):
    for j in range(Ny_grid):
        for k in range(Nz_grid):
            if np.isinf(phase[i,j,k]):
                geom[i,j,k] = 1

# Get void and solid indices/statistics
# https://numpy.org/doc/stable/reference/routines.logic.html

indices = np.unique(geom)
voidIdxList  = indices[np.where(np.logical_and(np.greater(indices,1), np.less(indices,solidIdx)))[0]]
solidIdxList = indices[np.where(np.greater_equal(indices,solidIdx))[0]]

logging.info(f'Total number of voxels = {np.prod(phase.shape)} voxels.')
logging.info(f'Number of solid voxels = {numSolidVoxels} voxels.')
logging.info(f'Number of air voxels = {np.prod(phase.shape) - numSolidVoxels} voxels.')
logging.info(f'Inserting AT LEAST {minNumVoidVoxels} voxels as voids.')
# logging.info(f'Estimate {len(np.unique(geom)) - numGrains - 1} clusters of voids.') # deprecate: inaccurate indicator
logging.info(f'Calculate {len(voidIdxList)} clusters of voids.') # do not account for AIR=1
logging.info(f'Number of grains: {numGrains}.')
logging.info(f'\n-------------------- NOTE --------------------\n')
logging.info(f'Indexing grain id:')
logging.info(f'Grain id for AIR: 1.')
# logging.info(f'Grain id for VOIDS: from 2 to {solidIdx-1}.') # deprecated
logging.info(f'Grain id for VOIDS: from 2 to {numVoids+1}.')
logging.info(f'Grain id for SOLID: from {solidIdx} to {np.max(geom)}.')
logging.info(f'Average grain size [m] = {(numSolidVoxels / numGrains * (5e-6)**3)**(1/3)}.') # For DAMASK material.config
logging.info(f'\n--------------- VOID STATISTICS ---------------\n')
logging.info(str(('Void indices are: ', list(voidIdxList))))
logging.info(f'\nVoid Index, Void Size')
for voidIdx in voidIdxList:
    tmpVoid = np.where(geom == voidIdx)
    logging.info(f'{voidIdx},{len(tmpVoid[0])}')


logging.info(f'\n-------------- GRAIN STATISTICS ---------------\n')
logging.info(str(('Grain indices are: ', list(solidIdxList))))
logging.info(f'\nGrain Index, Grain Size')
for solidIdx in solidIdxList:
    tmpGrain = np.where(geom == solidIdx)
    logging.info(f'{solidIdx},{len(tmpGrain[0])}')

# renumerate geom
logging.info(f'-------------- RE-ENUMERATION ---------------')
geom = renumerate(geom, startIndex=1, cluster=True, excludeList=[1])  # clustering: this step does not renumerate but only cluster
geom = renumerate(geom, startIndex=1, cluster=False, excludeList=[1]) # index start at 1: this step renumerate without clustering
updatedNumVoids = len(voidIdxList)
updatedSolidIdx = 1 + updatedNumVoids + 1
updatedNumGrains = np.max(geom) - updatedSolidIdx + 1
logging.info(f'\n\nIndexing grain id:')
logging.info(f'Grain id for AIR: 1.')
logging.info(f'Grain id for VOIDS: from 2 to {updatedNumVoids+1}.')
logging.info(f'Grain id for SOLID: from {updatedSolidIdx} to {np.max(geom)}.')
logging.info(f'Number of grains: {updatedNumGrains}.')

# Dump essential indices to grainInfo.dat for fast check
f = open('grainInfo.dat', 'w')
f.write('1\n')
f.write('2\n')
f.write('%d\n' % (updatedNumVoids+1))
f.write('%d\n' % updatedSolidIdx)
f.write('%d\n' % np.max(geom))
f.close()

# Dump void+air phases to another 3d *.npy array
phasePlusVoids = np.copy(phase)
for i in range(Nx_grid):
    for j in range(Ny_grid):
        for k in range(Nz_grid):
            if geom[i,j,k] < updatedSolidIdx:
                phasePlusVoids[i,j,k] = np.inf

logging.info(f'\nDumping new phase into phase + {phaseFileName}')
np.save('void+' + phaseFileName, phasePlusVoids)

# Save 3d microstructure as *.npy array
np.save(outFileName[:-5] + '.npy', geom)

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
voidOrientations = np.random.rand(updatedNumVoids+1, 3) * (ubEulerAngles - lbEulerAngles) + lbEulerAngles # +1 to include air
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
# if air_id = updatedNumGrains, then grain ids translate to (i+1)

air_id = 1
f.write('[grain%d]\n' % (air_id))
f.write('(gauss) phi1 0   Phi 0    phi2 0   scatter 0.0   fraction 1.0 \n')

for i in range(updatedNumVoids):
    f.write('[grain%d]\n' % (i+air_id+1)) # assign grain id
    phi1, Phi, phi2 = voidOrientations[i,:]
    f.write('(gauss) phi1 %.3f   Phi %.3f    phi2 %.3f   scatter 0.0   fraction 1.0 \n' % (phi1, Phi, phi2))

for i in range(updatedNumGrains):
    f.write('[grain%d]\n' % (i+updatedNumVoids+air_id+1)) # assign grain id
    phi1, Phi, phi2 = solidOrientations[i,:]
    f.write('(gauss) phi1 %.3f   Phi %.3f    phi2 %.3f   scatter 0.0   fraction 1.0 \n' % (phi1, Phi, phi2))

f.write('\n')
f.write('<microstructure>\n')
f.write('[grain%d]\n' % (air_id))
f.write('crystallite 1\n')
f.write('(constituent)   phase 2 texture %d fraction 1.0\n' % air_id)

for i in range(updatedNumVoids):
    f.write('[grain%d]\n' % (i+air_id+1)) # assign grain id
    f.write('crystallite 1\n')
    f.write('(constituent)   phase 2 texture %d fraction 1.0\n' % (i+air_id+1)) # assign grain id

for i in range(updatedNumGrains):
    f.write('[grain%d]\n' % (i+updatedNumVoids+air_id+1)) # assign grain id
    f.write('crystallite 1\n')
    f.write('(constituent)   phase 1 texture %d fraction 1.0\n' % (i+updatedNumVoids+air_id+1)) # assign grain id


f.close()

### diagnostics

elapsed = time.time() - t_start
logging.info("seedVoid.py: finished in {:5.2f} seconds.\n".format(elapsed))
