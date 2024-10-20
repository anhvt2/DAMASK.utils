import numpy as np
import glob, os
from natsort import natsorted, ns # natural-sort
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("-g" , "--geom", help='original geom fileName', type=str, required=True)

args = parser.parse_args()
geomFileName = args.geom # e.g. geomFileName = 'singleCrystal_res_50um.geom'

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
print('Number of grains = %d' % numGrains)

for grainId in grainList:
    x,y,z = np.where()

