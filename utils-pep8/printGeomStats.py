#!/usr/bin/env python3
import argparse

  # natural-sort
import numpy as np

PARSER = argparse.ArgumentParser(description='')
PARSER.add_argument("-g", "--geom", help='original geom filename', type=str, required=True)

ARGS = PARSER.parse_args()
GEOM_FILE_NAME = ARGS.geom  # e.g. geomFileName = 'singleCrystal_res_50um.geom'


def _delete(lst, to_delete):
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


def _geom2_npy(geomFileName):
    with open(geomFileName) as file_handler:
        txt = file_handler.readlines()
    num_skipping_lines = int(txt[0].split(' ')[0]) + 1
    for j in range(num_skipping_lines):
        if 'grid' in txt[j]:
            clean_string = _delete(txt[j].replace('\n', '').split(' '), '')
            Nx = int(clean_string[2])
            Ny = int(clean_string[4])
            Nz = int(clean_string[6])

    geom_block = txt[num_skipping_lines:]
    geom = sum(geom_block)

    geom = geom.split(' ')
    geom = list(filter(''.__ne__, geom))
    geom = np.array(geom, dtype=int).reshape(Nz, Ny, Nx).T
    return geom


GEOM = _geom2_npy(GEOM_FILE_NAME)

GRAIN_LIST = np.unique(GEOM)
NUM_GRAINS = len(GRAIN_LIST)
print('Number of grains = %d\n' % NUM_GRAINS)

for grainId in GRAIN_LIST:
    x, _, _ = np.where(GEOM == grainId)
    print('Grain %s: %d voxel' % (grainId, len(x)))
