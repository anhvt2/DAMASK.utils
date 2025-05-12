#!/usr/bin/env python3


import argparse
import glob

import matplotlib.pyplot as plt
import pyvista

PARSER = argparse.ArgumentParser()

PARSER.add_argument(
    "-n", "--name_tag", help='append to filename', type=str, default='', required=False
)

ARGS = PARSER.parse_args()
NAME_TAG = ARGS.name_tag

NAME_TAG = NAME_TAG.split('/')[0]
print(NAME_TAG)
# https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
# Ranking: (1) 'coolwarm', (2) 'ocean', (3) 'plasma' or 'inferno' or 'viridis'
CMAP = plt.cm.get_cmap('coolwarm')
# https://matplotlib.org/cmocean/#thermal

for filename in glob.glob('*.vtr'):  # screenshot for all *.vtr files
    reader = pyvista.get_reader(filename)
    ms_mesh = reader.read()
    ms_mesh.get_array('microstructure')
    ms_mesh.set_active_scalars('microstructure', preference='cell')

    pl = pyvista.Plotter(off_screen=True)
    pl.add_mesh(ms_mesh, show_edges=False, line_width=1, cmap=CMAP)
    pl.background_color = "white"
    pl.view_xy()
    pl.remove_scalar_bar()
    if NAME_TAG == '':
        pl.screenshot(filename.split('.')[0] + '.png', window_size=[1860 * 3, 968 * 3])
    else:
        pl.screenshot(
            filename.split('.')[0] + '_' + NAME_TAG + '.png', window_size=[968 * 3, 968 * 3]
        )
