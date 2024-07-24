
"""
This script stitches *.vti to a *.gif with sequentially different sequences. It only selects the frames that are different (no matter how small the difference is) to make a movie.

How to use:
    ls -1v *.vti > vtiFileList.txt
    python3 ../../animateGif.py --vtiFileListPointer='vtiFileList.txt' --phaseFileName='phase_dump_10_out.npy'
"""

import numpy as np
import pyvista
import matplotlib.pyplot as plt
import glob, os
import argparse
import gc
from natsort import natsorted, ns # natural-sort
pyvista.start_xvfb() # for running on headless server

parser = argparse.ArgumentParser()

parser.add_argument("-fl", "--vtiFileListPointer", help='provide the (synchronous) file for *.vti', type=str, default='', required=True) # 'ls -1v *.vti > vtiFileList.txt'
parser.add_argument("-p", "--phaseFileName", help='provide masked phase', type=str, default='', required=True)

args = parser.parse_args()
vtiFileListPointer = args.vtiFileListPointer # 'ls -1v *.vti > vtiFileList.txt'
phaseFileName      = args.phaseFileName # 'phase_dump_10_out.npy'

def vti2npy(fileName):
    reader = pyvista.get_reader(fileName)
    msMesh = reader.read()
    ms = msMesh.get_array('Spin')
    x, y, z = int(msMesh.bounds[1]), int(msMesh.bounds[3]), int(msMesh.bounds[5])
    ms = ms.reshape(z,y,x).T
    return np.array(ms)

def maskMs(phase, ms):
    '''
    This function masks a microstructure (ms) using phase.
    The resulting microstructure has grainID += 1 due to masking.
    '''
    maskedPhase = np.array(~np.isinf(phase), dtype=int)
    maskedMs = np.multiply(maskedPhase, ms)
    maskedMs += 1
    return maskedMs

def highlightMs(currentMs, initialMs):
    '''
    This function 
        (1) compares a current ms with the initial ms,
        (2) and masks every element as 1 if the same, retain the same if different
    '''
    dimensions = currentMs.shape
    mask = np.array(currentMs != initialMs, dtype=int)
    highlightedCurrentMs = np.multiply(currentMs, mask)
    return highlightedCurrentMs

vtiFileList = np.loadtxt(vtiFileListPointer, dtype=str)
phase = np.load(phaseFileName)
initialMs = maskMs(phase, vti2npy(vtiFileList[0]))

msMesh = pyvista.get_reader(vtiFileList[0]).read()
x_ex, y_ex, z_ex = msMesh.x, msMesh.y, msMesh.z

### 
# grid = pyvista.StructuredGrid(x_ex, y_ex, z_ex)
grid = msMesh.copy(deep=True)

pl = pyvista.Plotter()
pl.add_mesh(
    grid, # msMesh.threshold(value=1+1e-6, scalars='Spin'),
    scalars="Spin",
    lighting=False,
    show_edges=True,
)

j = 0
pl.open_gif('wiggle.gif')
for i in range(1, len(vtiFileList)):
    currentMs = maskMs(phase, vti2npy(vtiFileList[i]))
    previousMs = maskMs(phase, vti2npy(vtiFileList[i-1]))
    if np.any(currentMs != previousMs):
        highlightedCurrentMs = highlightMs(currentMs, initialMs)
        # nextMs = maskMs(phase, np.load(vtiFileList[i+1]))
        grid.points = highlightedCurrentMs.ravel()
        grid['Spin'] = highlightedCurrentMs.ravel()
        pl.write_frame()
        print(f'done {vtiFileList[i]}')
        j += 1

pl.close()
