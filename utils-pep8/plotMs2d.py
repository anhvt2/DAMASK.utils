#!/usr/bin/env python3


import pyvista
import matplotlib.pyplot as plt
import glob
import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-n", "--name_tag", help='append to filename',
                    type=str, default='', required=False)
args = parser.parse_args()
name_tag = args.name_tag

name_tag = name_tag.split('/')[0]
print(name_tag)


# cmap = plt.cm.get_cmap("viridis", 5)
# https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
# Ranking: (1) 'coolwarm', (2) 'ocean', (3) 'plasma' or 'inferno' or 'viridis'
cmap = plt.cm.get_cmap('coolwarm')
# cmap = plt.cm.get_cmap('viridis')
# cmap = plt.cm.get_cmap('plasma')
# cmap = plt.cm.get_cmap('inferno')
# cmap = plt.cm.get_cmap('ocean')
# cmap = plt.cm.get_cmap('gnuplot2')

# https://matplotlib.org/cmocean/#thermal
# import cmocean
# cmap = cmocean.cm.phase

# filename = 'single_phase_equiaxed_8x8x8.vtr'
for filename in glob.glob('*.vtr'):  # screenshot for all *.vtr files
    reader = pyvista.get_reader(filename)
    # camera = pyvista.Camera()
    ms_mesh = reader.read()
    ms = ms_mesh.get_array('microstructure')
    ms_mesh.cell_data['microstructure']
    ms_mesh.set_active_scalars('microstructure', preference='cell')

    # pl = pyvista.Plotter()
    pl = pyvista.Plotter(off_screen=True)
    pl.add_mesh(ms_mesh, show_edges=False, line_width=1, cmap=cmap)
    pl.background_color = "white"
    # light = pyvista.Light(position=(16, 16, 10), color='white')
    # light.positional = True
    pl.view_xy()
    pl.remove_scalar_bar()
    # pl.add_light(light)
    # pl.show(screenshot='%s.png' % filename.split('.')[0])
    # pl.show()
    # pl.close()
    if name_tag == '':
        pl.screenshot(filename.split(
            '.')[0] + '.png', window_size=[1860*3, 968*3])
        # pl.screenshot(filename.split('.')[0] + '.png', window_size=[3200,3200])
    else:
        pl.screenshot(filename.split(
            '.')[0] + '_' + name_tag + '.png', window_size=[968*3, 968*3])
