

"""
    Convert a microstructure SPPARKS dump file to another modified SPPARKS dump file. 

    This script is to be used in concert with
        1. `geom_cad2phase.py` to model void.

    In the nutshell, it increases the spin/grain ID += 1 and assign void ID = 1 for empty space. 
    This script is inspired by SPPARKS.utils/spparks_logo/maskSpkVti.py

    Examples
    --------
        python3 geom_spk2spk.py --vti='potts_3d.*.vti' --phase='m_dump_12_out.npy'

    Parameters
    ----------
        --vti: dump file from SPPARKS
        --phase (formatted in .npy): phase for dogbone modeling (could be generalized to internal void as well)
        --void_id (DEPRECATED) (adopted from geom_cad2phase.py): default void_id = np.inf, non void = -1 (see geom_cad2phase.py for more information)


"""

import numpy as np
import os
import sys
import time
import glob
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import pyvista
from natsort import natsorted, ns
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist,squareform


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dumpFileName",     type=str, required=True)
args = parser.parse_args()

dumpFileName = args.dumpFileName
npyFileName = dumpFileName[:-4] + '.npy'

if os.path.exists(dumpFileName):
    print(f'{dumpFileName} exists. Checkpoint passed.')
else:
    print(f"Dump file {dumpFileName} does not exist. Check {dumpFileName}")

def getDumpMs(dumpFileName):
    """
        This function return a 3d array 'm' microstructure from reading a SPPARKS dump file, specified by 'dumpFileName'.
    """
    dumpFile = open(dumpFileName)
    dumptxt = dumpFile.readlines()
    dumpFile.close()
    for i in range(20):  # look for header info in first 20 lines
        tmp = dumptxt[i]
        if 'BOX BOUNDS' in tmp:
            Nx = int(dumptxt[i+1].replace('\n',
                     '').replace('0 ', '').replace(' ', ''))
            Ny = int(dumptxt[i+2].replace('\n',
                     '').replace('0 ', '').replace(' ', ''))
            Nz = int(dumptxt[i+3].replace('\n',
                     '').replace('0 ', '').replace(' ', ''))
            break
    header = np.array(
        dumptxt[i+4].replace('\n', '').replace('ITEM: ATOMS ', '').split(' '), dtype=str)
    d = np.loadtxt(dumpFileName, skiprows=9, dtype=int)
    # Get indices of relevant fields
    # print(header) # debug
    typeIdx = np.where(header=='type')[0]
    xIdx = np.where(header=='x')[0]
    yIdx = np.where(header=='y')[0]
    zIdx = np.where(header=='z')[0]
    # print(typeIdx, xIdx, yIdx, zIdx) # debug
    '''
    Re-enumerate grains: instead of having unique but sparse grain id, e.g. [1,2,4,7], now re-enumerate to [0,1,2,3]
    '''
    numGrains = len(np.unique(d[:, typeIdx]))
    oldGrainIds = np.unique(d[:, typeIdx])
    newGrainIds = range(len(np.unique(d[:, typeIdx])))
    # Create ms in .npy format
    ms = np.zeros([Nx, Ny, Nz])  # initialize
    for ii in range(len(d)):
        i = int(d[ii, xIdx])  # 'x'
        j = int(d[ii, yIdx])  # 'y'
        k = int(d[ii, zIdx])  # 'z'
        grain_id = int(d[ii, typeIdx]) # grain id
        # option: DO re-enumerating
        lookupIdx = np.where(oldGrainIds == grain_id)[0][0]
        new_grain_id = newGrainIds[lookupIdx]
        ms[i, j, k] = new_grain_id
        # option: DO NOT re-enumerating
        # m[i,j,k] = grain_id # TODO: implement re-enumerate grain_id
        # print(f"finish ({x},{y}, {z})")

    return ms, Nx, Ny, Nz, numGrains

def reEnumerate(ms):
    '''
    ASSUMPTION: NON-PERIODIC
    This function
        (1) reads a microstructure, 
        (2) performs connectivity check,
        (3) re-assign grainId to clusters that have same grainId but are faraway
    Note: 
    (1) Need to verify against ParaView with thresholding
    (2) Clustering under periodic boundary condition: https://francescoturci.net/2016/03/16/clustering-and-periodic-boundaries/
    
    Preliminary results:
    DBSCAN works (well), Birch not, HDBSCAN only available in 1.3 (N/A), MeanShift may work if 'bandwidth' is tuned correctly (won't work w/ default param), AgglomerativeClustering works well with correctly estimated n_clusters

    Parameters
    ----------
    ms
    Return
    ------
    ms (updated)
    '''
    grainIdList = np.sort(np.unique(ms))
    maxGrainId = np.max(grainIdList)
    # Segregate grains with the same grainId by clustering
    for grainId in grainIdList:
        # Collect location information
        x,y,z = np.where(ms==grainId)
        # Combine to a 3-column array
        X = np.hstack((np.atleast_2d(x).T, np.atleast_2d(y).T, np.atleast_2d(z).T))
        # # Estimate maximum number of clusters # DEPRECATED for not being used
        # x_estimated_clusters = len(np.where(np.diff(np.sort(x)) > 2)[0])
        # y_estimated_clusters = len(np.where(np.diff(np.sort(y)) > 2)[0])
        # z_estimated_clusters = len(np.where(np.diff(np.sort(z)) > 2)[0])
        # estimated_clusters = np.max([x_estimated_clusters, y_estimated_clusters, z_estimated_clusters]) + 1
        # Perform clustering algorithm
        clustering = DBSCAN(eps=2, min_samples=10).fit(X)
        # print(f"n_clusters = {len(set(clustering.labels_))}")
        # Relabel grainId for every pixels needed relabel
        for j in range(clustering.labels_.shape[0]):
            ms[x[j],y[j],z[j]] = maxGrainId+clustering.labels_[j]+1
        # Update maxGrainId
        maxGrainId = np.max(np.sort(np.unique(ms)))
        print(f'Segregating grains for grainId {grainId.astype(int)}')
    # Reorder
    print(f'Re-enumerating microstructure ... ', end='')
    grainIdList = np.sort(np.unique(ms))
    for i in range(len(grainIdList)):
        grainId = grainIdList[i]
        # Collect location information
        x,y,z = np.where(ms==grainId)
        for j in range(x.shape[0]):
            ms[x[j],y[j],z[j]] = i
    print(f'Done!')
    return ms

t_start = time.time()  # tic

# Process
print(f'Reading dumpFileName {dumpFileName}...', end=' ')
ms, Nx, Ny, Nz, numGrains = getDumpMs(dumpFileName)
print(f'done!')
print(f'Reenumerate microstructure...', end=' ')
reEnum_ms = reEnumerate(ms)
print(f'done!')

# Save outputs
np.save(npyFileName, ms)
np.save('reenum_' + npyFileName, reEnum_ms)

elapsed = time.time() - t_start  # toc
print("geom_spk2npy.py: finished in {:5.2f} seconds.\n".format(elapsed), end="")
