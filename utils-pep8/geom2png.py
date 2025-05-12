#!/usr/bin/env python3

import argparse

  # natural-sort
import gc

import matplotlib.pyplot as plt
import numpy as np
import pyvista

PARSER = argparse.ArgumentParser()
PARSER.add_argument("-g", "--geom", type=str, required=True)
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

FILENAME = ARGS.geom
THRESHOLD = ARGS.threshold
SHOW_EDGES = bool(ARGS.show_edges)
NAME_TAG = ARGS.nameTag


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


# deprecate filename.split('.')[0] to avoid '.' in outFileName
with open(FILENAME) as fileHandler:
    txt = fileHandler.readlines()


NUM_SKIPPING_LINES = int(txt[0].split(' ')[0]) + 1
# Search for 'size' within header:
for j in range(NUM_SKIPPING_LINES):
    if 'grid' in txt[j]:
        cleanString = _delete(txt[j].replace('\n', '').split(' '), '')
        Nx = int(cleanString[2])
        Ny = int(cleanString[4])
        Nz = int(cleanString[6])

GEOM_BLOCK = txt[NUM_SKIPPING_LINES:]
GEOM = sum(GEOM_BLOCK)

GEOM = GEOM.split(' ')
GEOM = list(filter(('').__ne__, GEOM))

# Convert from 1 line format to 3d format
GEOM = np.array(GEOM, dtype=int).reshape(Nz, Ny, Nx).T
  # to reverse: geom = geom.T.flatten()

GRID = pyvista.UniformGrid()  # old pyvista
GRID.dimensions = np.array(GEOM.shape) + 1
GRID.origin = (0, 0, 0)     # The bottom left corner of the data set
GRID.spacing = (1, 1, 1)    # These are the cell sizes along each axis
GRID.cell_data["microstructure"] = GEOM.flatten(order="F")  # ImageData()

PL = pyvista.Plotter(off_screen=True)
CMAP = plt.cm.get_cmap('coolwarm')
PL.add_mesh(
    GRID.threshold(value=THRESHOLD + 1e-6),
    scalars="microstructure",
    show_edges=SHOW_EDGES,
    line_width=1,
    cmap=CMAP,
)

PL.background_color = "white"
PL.remove_scalar_bar()

if GEOM.shape[2] == 1:
    PL.camera_position = 'xy'

if NAME_TAG == '':
    PL.screenshot(FILENAME[:-5] + '.png', window_size=[1860 * 6, 968 * 6])
else:
    PL.screenshot(FILENAME[:-5] + '_' + NAME_TAG + '.png', window_size=[1860 * 6, 968 * 6])


gc.collect()
