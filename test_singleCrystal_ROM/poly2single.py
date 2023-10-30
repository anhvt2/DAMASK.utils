
import numpy as np
import glob, os
from natsort import natsorted, ns # natural-sort

for fileName in natsorted(glob.glob('*.geom')):
    outFileName = 'singleCrystal_' + fileName
    fileHandler = open(fileName)
    txt = fileHandler.readlines()
    fileHandler.close()

    geomBlock = txt[7:]
    geom = ''
    for i in range(len(geomBlock)):
        geom += geomBlock[i]

    geom = geom.split(' ')
    geom = list(filter(('').__ne__, geom))
    geom = np.array(geom, dtype=int)
    for i in range(len(geom)):
        if geom[i] > 1:
            geom[i] = 2

    num_lines = int(np.floor(len(geom)) / 10)
    num_elems_last_line = int(len(geom) % 10)
    num_grains = 1 # single crystal
    # print(txt)

    f = open(outFileName, 'w')
    for i in range(6):
        f.write(txt[i])

    f.write('microstructures %d\n' % (num_grains+1))

    for j in range(int(num_lines)):
        for k in range(10):
            idx = int(j*10 + k)
            f.write('%10d' % int(geom[idx]))
        f.write('\n')

    if num_elems_last_line > 0:
        for idx in range(-num_elems_last_line,0):
            f.write('%10d' % int(geom[idx]))

    f.close()

