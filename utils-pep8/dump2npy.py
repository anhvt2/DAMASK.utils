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

import argparse
import os
import time

import numpy as np
from sklearn.cluster import DBSCAN

PARSER = argparse.ArgumentParser()
PARSER.add_argument("-dump", "--dump", type=str, required=True)

PARSER.add_argument("-renum", "--renumateFlag", type=str, required=False, default=False)

ARGS = PARSER.parse_args()


DUMP_FILE_NAME = ARGS.dump
RENUMATE_FLAG = ARGS.renumateFlag
NPY_FILE_NAME = DUMP_FILE_NAME + '.npy'

if os.path.exists(DUMP_FILE_NAME):
    print(f'{DUMP_FILE_NAME} exists. Checkpoint passed.')
else:
    print(f"Dump file {DUMP_FILE_NAME} does not exist. Check {DUMP_FILE_NAME}")


def _get_dump_ms(dumpFileName):
    """
    This function return a 3d array 'm' microstructure from reading a SPPARKS dump file, specified by 'dumpFileName'.
    """
    with open(dumpFileName) as dumpFile:
        dumptxt = dumpFile.readlines()
    for i in range(20):
        tmp = dumptxt[i]
        if 'BOX BOUNDS' not in tmp:
            continue
        Nx = int(dumptxt[i + 1].replace('\n', "").replace('0 ', "").replace(' ', ""))
        Ny = int(dumptxt[i + 2].replace('\n', "").replace('0 ', "").replace(' ', ""))
        Nz = int(dumptxt[i + 3].replace('\n', "").replace('0 ', "").replace(' ', ""))
        break
    header = np.array(
        dumptxt[i + 4].replace('\n', "").replace('ITEM: ATOMS ', "").split(' '), dtype=str
    )
    d = np.loadtxt(dumpFileName, skiprows=9, dtype=int)
    type_idx = np.where(header == 'i1')[0]
    x_idx = np.where(header == 'x')[0]
    y_idx = np.where(header == 'y')[0]
    z_idx = np.where(header == 'z')[0]
    num_grains = len(np.unique(d[:, type_idx]))
    old_grain_ids = np.unique(d[:, type_idx])
    new_grain_ids = range(len(np.unique(d[:, type_idx])))
    ms = np.zeros([Nx, Ny, Nz])
    for ii in range(len(d)):
        i = int(d[ii, x_idx])
        j = int(d[ii, y_idx])
        k = int(d[ii, z_idx])
        grain_id = int(d[ii, type_idx])
        lookup_idx = np.where(old_grain_ids == grain_id)[0][0]
        new_grain_id = new_grain_ids[lookup_idx]
        ms[i, j, k] = new_grain_id
    return (ms, Nx, Ny, Nz, num_grains)


def _renumerate(ms):
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
    for grainId in grain_id_list:
        (x, y, z) = np.where(ms == grainId)
        X = np.hstack((np.atleast_2d(x).T, np.atleast_2d(y).T, np.atleast_2d(z).T))
        clustering = DBSCAN(eps=2, min_samples=10).fit(X)
        for j in range(clustering.labels_.shape[0]):
            ms[x[j], y[j], z[j]] = max_grain_id + clustering.labels_[j] + 1
        max_grain_id = np.max(np.sort(np.unique(ms)))
        print(f'Segregating grains for grainId {grainId.astype(int)}')
    print(f'Renumerating microstructure ... ', end="")
    grain_id_list = np.sort(np.unique(ms))
    for i in range(len(grain_id_list)):
        grainId = grain_id_list[i]
        (x, y, z) = np.where(ms == grainId)
        for j in range(x.shape[0]):
            ms[x[j], y[j], z[j]] = i
    print(f'Done!')
    return ms


T_START = time.time()  # tic

# Process
print(f'Reading dumpFileName {DUMP_FILE_NAME}...', end=' ')
MS, _, _, _, _ = _get_dump_ms(DUMP_FILE_NAME)
print(f'done!')
print(f'Renumerating microstructure...', end=' ')
if RENUMATE_FLAG is True:
    reEnum_ms = _renumerate(MS)
else:
    reEnum_ms = MS
print(f'done!')

# Save outputs
np.save(NPY_FILE_NAME, MS)
np.save(NPY_FILE_NAME, reEnum_ms)  # deprecate: 'reenum_' + npyFileName

ELAPSED = time.time() - T_START  # toc
print("dump2npy.py: finished in {:5.2f} seconds.\n".format(ELAPSED), end="")
