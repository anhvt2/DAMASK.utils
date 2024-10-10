
"""
Example how to use:
    python3 vizAM.py --npyFolderName='npy' --phaseFileName='void+phase_dump_12_out.npy'

then
bash highlight.sh
# python3 ../../../npy2png.py --threshold=1

This script 
    (1) converts a series of microstructures formatted in *.npy (must be)
    (2) along with a chosen phase (for example, dogbone)
    (3) to a series of images (that could be converted to video) for illustration purpose

* Work for any voxelized-STL file.

How?
Step-1: mask according to phase (void default id = np.inf)
Step-2: only show difference compared to initial microstructure

Other random thoughts:
- Convert *.vti to *.npy
- Dump masked *.npy based on (1) phase and (2) difference with the initial condition *.npy
- Convert masked *.npy to *.geom
- Convert *.geom to *.vtr
OR
- ~~Convert masked *.npy directly to *.vtr # https://tutorial.pyvista.org/tutorial/02_mesh/solutions/c_create-uniform-grid.html~~ (done, see `npy2png.py`)

Show us with threshold, hide the rest.
- Always show the final us (with opacity=0 for consistent grain ID colormap)
- Only show the diff b/w the current us and the initial us, but NOT include masked phase

"""

import numpy as np
import pyvista
import matplotlib.pyplot as plt
import glob, os
import argparse
import gc
from natsort import natsorted, ns # natural-sort
parser = argparse.ArgumentParser()

parser.add_argument("-n", "--npyFolderName", help='provide folders that supply all *.npy', type=str, default='', required=True)
parser.add_argument("-p", "--phaseFileName", help='provide masked phase', type=str, default='', required=True)
args = parser.parse_args()
npyFolderName = args.npyFolderName # npyFolderName = 'npy' # debug
phaseFileName = args.phaseFileName # phaseFileName = 'void+phase_dump_12_out.npy' # debug

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


npyFolderList = natsorted(glob.glob(npyFolderName + '/*.npy'))
phase = np.load(phaseFileName)
initialMs = maskMs(phase, np.load(npyFolderList[0]).astype(int))
lastMs    = maskMs(phase, np.load(npyFolderList[-1]).astype(int))

# Create a domain of random size that is less likely to be affected -- usually the last piece of AM path
rS = (np.array(initialMs.shape) * 0.05).astype(int) # random size

f = open('vizAM.py.log', 'w')
f.write('imgIdx,dumpIdx\n')

# Re-index with j (instead of i) for continuous time frame
j = 0
for i in range(len(npyFolderList)):
    # Strategy: shift first, then mask (not the other way around)
    initialMs = np.load(npyFolderList[0]).astype(int)
    currentMs = np.load(npyFolderList[i]).astype(int)
    prevMs = np.load(npyFolderList[i-1]).astype(int)
    # Safeguard: automatically detect uniform shift & adjust 
    # currentMs / prevMs
    calibDom = currentMs[-rS[0]:,-rS[1]:,-rS[2]:] - prevMs[-rS[0]:,-rS[1]:,-rS[2]:] # last domain to be AM'ed
    if np.max(calibDom) == np.min(calibDom):
        grainIdxShift = np.max(calibDom)
        prevMs += grainIdxShift # adjust grain ID shift
    # currentMs / initialMs
    calibDom = currentMs[-rS[0]:,-rS[1]:,-rS[2]:] - initialMs[-rS[0]:,-rS[1]:,-rS[2]:] # last domain to be AM'ed
    if np.max(calibDom) == np.min(calibDom):
        grainIdxShift = np.max(calibDom)
        initialMs += grainIdxShift # adjust grain ID shift
    # Mask with phase
    initialMs = maskMs(phase, initialMs)
    currentMs = maskMs(phase, currentMs)
    prevMs = maskMs(phase, prevMs)
    # Highlight the different
    if np.any(currentMs != prevMs):
        highlightedCurrentMs = highlightMs(currentMs, initialMs)
        # nextMs = maskMs(phase, np.load(npyFolderList[i+1]))
        np.save('highlighted_ms_%d.npy' % j, highlightedCurrentMs)
        print(f'done {npyFolderList[i]}')
        j += 1
        f.write(f'%d,%d\n' % (j,i))

f.close()

