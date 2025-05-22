#!/usr/bin/env python3

# https://stackoverflow.com/questions/77696805/pyvista-3d-visualization-issue
# https://tutorial.pyvista.org/tutorial/02_mesh/solutions/c_create-uniform-grid.html#sphx-glr-tutorial-02-mesh-solutions-c-create-uniform-grid-py

import argparse
import gc
from distutils.util import strtobool

import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
import numpy as np
import pyvista

  # natural-sort
PARSER = argparse.ArgumentParser()

PARSER.add_argument("-n", "--npy", help='*.npy file', type=str, default='', required=True)

PARSER.add_argument(
    "-threshold", "--threshold", help='threshold', type=int, default=-1, required=False
)

PARSER.add_argument("-nameTag", "--nameTag", help='', type=str, default='', required=False)

PARSER.add_argument(
    "-show_edges",
    "--show_edges",
    help='pyvista show_edges',
    type=lambda x: bool(strtobool(x)),
    default=True,
    required=False,
    nargs='?',
    const=True,
)

ARGS = PARSER.parse_args()
NPY_FILE_NAME = ARGS.npy  # 'npy'
THRESHOLD = ARGS.threshold
NAME_TAG = ARGS.nameTag
SHOW_EDGES = bool(ARGS.show_edges)
# https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
# Ranking: (1) 'coolwarm', (2) 'ocean', (3) 'plasma' or 'inferno' or 'viridis'
CMAP = plt.cm.get_cmap('coolwarm')
# https://matplotlib.org/cmocean/#thermal

MS = np.load(NPY_FILE_NAME)
GRID = pyvista.UniformGrid()  # old pyvista
GRID.dimensions = np.array(MS.shape) + 1
GRID.origin = (0, 0, 0)     # The bottom left corner of the data set
GRID.spacing = (1, 1, 1)    # These are the cell sizes along each axis
GRID.cell_data["microstructure"] = MS.flatten(order="F")  # ImageData()

PL = pyvista.Plotter(off_screen=True)
PL.add_mesh(
    GRID.threshold(value=THRESHOLD + 1e-6),
    scalars="microstructure",
    show_edges=SHOW_EDGES,
    line_width=1,
    cmap=CMAP,
)

PL.background_color = "white"
PL.remove_scalar_bar()
# Leidong/Zihan custom view

if MS.shape[2] == 1:
    PL.camera_position = 'xy'

if NAME_TAG == '':
    PL.screenshot(NPY_FILE_NAME[:-4] + '.png', window_size=[1860 * 6, 968 * 6])
else:
    PL.screenshot(NPY_FILE_NAME[:-4] + '_' + NAME_TAG + '.png', window_size=[1860 * 6, 968 * 6])


gc.collect()
