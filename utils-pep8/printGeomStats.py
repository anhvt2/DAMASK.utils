import numpy as np
import glob
import os
from natsort import natsorted, ns  # natural-sort
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument(
    "-g", "--geom", help='original geom filename', type=str, required=True)

args = parser.parse_args()
geomFileName = args.geom  # e.g. geomFileName = 'singleCrystal_res_50um.geom'


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


geom = geom2npy(geomFileName)

grainList = np.unique(geom)
numGrains = len(grainList)
print('Number of grains = %d\n' % numGrains)

for grainId in grainList:
    x, y, z = np.where(geom == grainId)
    print('Grain %s: %d voxel' % (grainId, len(x)))
