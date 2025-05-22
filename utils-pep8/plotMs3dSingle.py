#!/usr/bin/env python3


import argparse

import matplotlib.pyplot as plt
import pyvista

PARSER = argparse.ArgumentParser()


PARSER.add_argument("-f", "--filename", help='.vtr file', type=str, default='', required=True)

PARSER.add_argument(
    "-n", "--name_tag", help='append to filename', type=str, default='', required=False
)

PARSER.add_argument(
    "-show_edges",
    "--show_edges",
    help='append to filename',
    type=bool,
    default=True,
    required=False,
)

ARGS = PARSER.parse_args()
FILENAME = ARGS.filename
NAME_TAG = ARGS.name_tag
SHOW_EDGES = ARGS.show_edges

NAME_TAG = NAME_TAG.split('/')[0]
print(NAME_TAG)
# https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
# Ranking: (1) 'coolwarm', (2) 'ocean', (3) 'plasma' or 'inferno' or 'viridis'
CMAP = plt.cm.get_cmap('coolwarm')
# https://matplotlib.org/cmocean/#thermal
# for filename in glob.glob('*.vtr'): # screenshot for all *.vtr files

READER = pyvista.get_reader(FILENAME)
MS_MESH = READER.read()
MS_MESH.get_array('microstructure')
MS_MESH.set_active_scalars('microstructure', preference='cell')

PL = pyvista.Plotter(off_screen=True)
PL.add_mesh(MS_MESH, show_edges=SHOW_EDGES, line_width=1, cmap=CMAP)
PL.background_color = "white"
PL.remove_scalar_bar()
if NAME_TAG == '':
    PL.screenshot(FILENAME[:-4] + '.png', window_size=[1860 * 6, 968 * 6])
else:
    PL.screenshot(FILENAME[:-4] + '_' + NAME_TAG + '.png', window_size=[1860 * 6, 968 * 6])
