#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import pyvista

PARSER = argparse.ArgumentParser()


PARSER.add_argument("-f", "--filename", help='.vtr file', type=str, default='', required=True)

PARSER.add_argument(
    "-n", "--nameTag", help='append to filename', type=str, default='', required=False
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
NAME_TAG = ARGS.nameTag

NAME_TAG = NAME_TAG.split('/')[0]
print(NAME_TAG)
# https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
# Ranking: (1) 'coolwarm', (2) 'ocean', (3) 'plasma' or 'inferno' or 'viridis'
CMAP = plt.cm.get_cmap('coolwarm')
# https://matplotlib.org/cmocean/#thermal


PL = pyvista.Plotter(off_screen=True)
READER = pyvista.get_reader(FILENAME)
MS_MESH = READER.read()
MS_MESH.set_active_scalars('texture', preference='cell')
THRESHED_MS = MS_MESH.threshold(value=(779, 816), scalars='texture')
THRESHED_MS.set_active_scalars('Mises(Cauchy)', preference='cell')

MS_MESH.set_active_scalars('Mises(Cauchy)', preference='cell')
PL.add_mesh(MS_MESH.threshold(value=1 + 1e-6), opacity=0.02, show_edges=False, line_width=0.01)
  # show original geometry
# warped by deforming geometry with displacement field
# https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.DataSetFilters.warp_by_vector.html#pyvista.DataSetFilters.warp_by_vector
PL.add_mesh(
    THRESHED_MS.warp_by_vector(vectors='avg(f).pos', factor=1.0),
    opacity=1.0,
    show_edges=True,
    line_width=1,
    cmap=CMAP,
)


PL.background_color = "white"
PL.remove_scalar_bar()

if NAME_TAG == '':
    PL.screenshot(FILENAME.split('.')[0] + '.png', window_size=[1860 * 6, 968 * 6])
else:
    PL.screenshot(FILENAME.split('.')[0] + '_' + NAME_TAG + '.png', window_size=[1860 * 6, 968 * 6])
