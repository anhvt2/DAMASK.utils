#!/usr/bin/env python3

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
Step-1: mask according to phase
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

import argparse
import glob

import numpy as np
from natsort import natsorted  # natural-sort
PARSER = argparse.ArgumentParser()

PARSER.add_argument(
    "-n",
    "--npyFolderName",
    help='provide folders that supply all *.npy',
    type=str,
    default='',
    required=True,
)

PARSER.add_argument(
    "-p", "--phaseFileName", help='provide masked phase', type=str, default='', required=True
)

ARGS = PARSER.parse_args()
NPY_FOLDER_NAME = ARGS.npyFolderName  # 'npy'
PHASE_FILE_NAME = ARGS.phaseFileName  # 'phase_dump_12_out.npy'


def _mask_ms(phase, ms):
    '''
    This function masks a microstructure (ms) using phase.
    The resulting microstructure has grainID += 1 due to masking.
    '''
    masked_phase = np.array(~np.isinf(phase), dtype=int)
    masked_ms = np.multiply(masked_phase, ms)
    masked_ms += 1
    return masked_ms


def _highlight_ms(currentMs, initialMs):
    '''
    This function
        (1) compares a current ms with the initial ms,
        (2) and masks every element as 1 if the same, retain the same if different
    '''
    mask = np.array(currentMs != initialMs, dtype=int)
    return np.multiply(currentMs, mask)


NPY_FOLDER_LIST = natsorted(glob.glob(NPY_FOLDER_NAME + '/*.npy'))
PHASE = np.load(PHASE_FILE_NAME)
INITIAL_MS = _mask_ms(PHASE, np.load(NPY_FOLDER_LIST[0]))
_mask_ms(PHASE, np.load(NPY_FOLDER_LIST[-1]))

J = 0
for i in range(len(NPY_FOLDER_LIST)):
    currentMs = _mask_ms(PHASE, np.load(NPY_FOLDER_LIST[i]))
    previousMs = _mask_ms(PHASE, np.load(NPY_FOLDER_LIST[i - 1]))
    if np.any(currentMs != previousMs):
        highlightedCurrentMs = _highlight_ms(currentMs, INITIAL_MS)
        np.save('highlighted_ms_%d.npy' % J, highlightedCurrentMs)
        print(f'done {NPY_FOLDER_LIST[i]}')
        J += 1
