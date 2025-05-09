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

np.random.seed(8)

level    = logging.INFO
format   = '  %(message)s'
logFileName = 'cleanseGeom.log'
os.system('rm -fv %s' % logFileName)
handlers = [logging.FileHandler(logFileName), logging.StreamHandler()]

logging.basicConfig(level = level, format = format, handlers = handlers)

t_start = time.time()

parser = argparse.ArgumentParser(description='')
parser.add_argument("-g" , "--geom", help='original geom filename', type=str, required=True)

args = parser.parse_args()
geomFileName = args.geom # e.g. geomFileName = 'singleCrystal_res_50um.geom'

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

def geom2npy(geomFileName):
    fileHandler = open(geomFileName)
    txt = fileHandler.readlines()
    fileHandler.close()
    numSkippingLines = int(txt[0].split(' ')[0])+1 
    # Search for 'size' within header:
    for j in range(numSkippingLines):
        if 'grid' in txt[j]:
            cleanString = delete(txt[j].replace('\n', '').split(' '), '')
            Nx = int(cleanString[2])
            Ny = int(cleanString[4])
            Nz = int(cleanString[6])

    geomBlock = txt[numSkippingLines:]
    geom = ''
    for i in range(len(geomBlock)):
        geom += geomBlock[i]

    geom = geom.split(' ')
    geom = list(filter(('').__ne__, geom))
    geom = np.array(geom, dtype=int).reshape(Nz, Ny, Nx).T
    return geom

def renumerate(geom, startIndex=0, cluster=False):
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

# Read to .npy format
geom = geom2npy(geomFileName)
Nx_grid, Ny_grid, Nz_grid = geom.shape
Nx_size, Ny_size, Nz_size = geom.shape

geom = renumerate(geom, startIndex=1, cluster=True) # This doesn't renumerate but only clustering
geom = renumerate(geom, startIndex=1) # This is absolutely necessary for renumerating grainIDs. 
outFileName = 'cleansed_' + geomFileName

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

### diagnostics

elapsed = time.time() - t_start
logging.info("cleanseGeom.py: finished in {:5.2f} seconds.\n".format(elapsed))
