#!/usr/bin/env python3


"""
    Convert a microstructure SPPARKS dump file to another modified SPPARKS dump file. 

    This script is to be used in concert with
        1. `geom_cad2phase.py` to model void.

    In the nutshell, it increases the spin/grain ID += 1 and assign void ID = 1 for empty space. 
    This script is inspired by SPPARKS.utils/spparks_logo/maskSpkVti.py

    Examples
    --------
        python3 geom_spk2spk.py --dumpFileName='dump.additive_dogbone.2802'

    Parameters
    ----------
        --dumpFileName: dump file from SPPARKS
"""

import numpy as np
import os
import sys
import time
import glob
import argparse
from sklearn.cluster import DBSCAN
# from scipy.spatial.distance import pdist,squareform


parser = argparse.ArgumentParser()
parser.add_argument("-dump", "--dump",     type=str, required=True)
parser.add_argument("-renum", "--renumateFlag",
                    type=str, required=False, default=False)
args = parser.parse_args()


dumpFileName = args.dump
renumateFlag = args.renumateFlag
npyFileName = dumpFileName + '.npy'

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
    type_idx = np.where(header == 'i1')[0]  # header=='type' or 'i1'
    x_idx = np.where(header == 'x')[0]
    y_idx = np.where(header == 'y')[0]
    z_idx = np.where(header == 'z')[0]
    # print(typeIdx, xIdx, yIdx, zIdx) # debug
    '''
    Renumerate grains: instead of having unique but sparse grain id, e.g. [1,2,4,7], now renumerate to [0,1,2,3]
    '''
    num_grains = len(np.unique(d[:, type_idx]))
    old_grain_ids = np.unique(d[:, type_idx])
    new_grain_ids = range(len(np.unique(d[:, type_idx])))
    # Create ms in .npy format
    ms = np.zeros([Nx, Ny, Nz])  # initialize
    for ii in range(len(d)):
        i = int(d[ii, x_idx])  # 'x'
        j = int(d[ii, y_idx])  # 'y'
        k = int(d[ii, z_idx])  # 'z'
        grain_id = int(d[ii, type_idx])  # grain id
        # option: DO renumerate
        lookup_idx = np.where(old_grain_ids == grain_id)[0][0]
        new_grain_id = new_grain_ids[lookup_idx]
        ms[i, j, k] = new_grain_id
        # option: DO NOT renumerate
        # m[i,j,k] = grain_id # TODO: implement renumerate grain_id
        # print(f"finish ({x},{y}, {z})")
    return ms, Nx, Ny, Nz, num_grains


def renumerate(ms):
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
    grain_id_list = np.sort(np.unique(ms))
    max_grain_id = np.max(grain_id_list)
    # Segregate grains with the same grainId by clustering
    for grainId in grain_id_list:
        # Collect location information
        x, y, z = np.where(ms == grainId)
        # Combine to a 3-column array
        X = np.hstack(
            (np.atleast_2d(x).T, np.atleast_2d(y).T, np.atleast_2d(z).T))
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
            ms[x[j], y[j], z[j]] = max_grain_id+clustering.labels_[j]+1
        # Update maxGrainId
        max_grain_id = np.max(np.sort(np.unique(ms)))
        print(f'Segregating grains for grainId {grainId.astype(int)}')
    # Reorder
    print(f'Renumerating microstructure ... ', end='')
    grain_id_list = np.sort(np.unique(ms))
    for i in range(len(grain_id_list)):
        grainId = grain_id_list[i]
        # Collect location information
        x, y, z = np.where(ms == grainId)
        for j in range(x.shape[0]):
            ms[x[j], y[j], z[j]] = i
    print(f'Done!')
    return ms


t_start = time.time()  # tic

# Process
print(f'Reading dumpFileName {dumpFileName}...', end=' ')
ms, Nx, Ny, Nz, numGrains = getDumpMs(dumpFileName)
print(f'done!')
print(f'Renumerating microstructure...', end=' ')
if renumateFlag == True:
    reEnum_ms = renumerate(ms)
else:
    reEnum_ms = ms
print(f'done!')

# Save outputs
np.save(npyFileName, ms)
np.save(npyFileName, reEnum_ms)  # deprecate: 'reenum_' + npyFileName

elapsed = time.time() - t_start  # toc
print("dump2npy.py: finished in {:5.2f} seconds.\n".format(elapsed), end="")
