
import pyvista
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import argparse
parser = argparse.ArgumentParser()

'''
    This util script uses PyVista to visualize and create a 3d stress fields, deformed by a displacement field

    Parameters
    ----------
    filename: str
    e.g. 'main_tension_inc16_pos(cell)_added.vtr'

    Output
    ------
    an image: filename.split('.')[0] + '_' + nameTag + '.png'

    How to use
    ----------
    python3 ../../../plotStress3d.py  --filename='main_tension_inc16_pos(cell)_added.vtr'
'''

parser.add_argument("-f", "--filename", help='.vtr file',
                    type=str, default='', required=True)
parser.add_argument("-n", "--nameTag", help='append to filename',
                    type=str, default='', required=False)
parser.add_argument("-show_edges", "--show_edges",
                    help='append to filename', type=bool, default=True, required=False)
args = parser.parse_args()
filename = args.filename
nameTag = args.nameTag
show_edges = args.show_edges

nameTag = nameTag.split('/')[0]
print(nameTag)


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

# displacementField = np.loadtxt('main_tension_inc16_nodal.txt', skiprows=3)
# displacementVector = displacementField[:,3]

pl = pyvista.Plotter(off_screen=True)
# reader = pyvista.get_reader('main_tension_inc16_pos(cell).vtr')
reader = pyvista.get_reader(filename)
msMesh = reader.read()
msMesh.set_active_scalars('texture', preference='cell')
threshedMs = msMesh.threshold(value=(779, 816), scalars='texture')
threshedMs.set_active_scalars('Mises(Cauchy)', preference='cell')

msMesh.set_active_scalars('Mises(Cauchy)', preference='cell')
pl.add_mesh(msMesh.threshold(value=1+1e-6), opacity=0.02,
            show_edges=False, line_width=0.01)  # show original geometry
# pl.add_mesh(threshedMs, opacity=0.05, show_edges=True, line_width=0.01) # show original geometry

# warped by deforming geometry with displacement field
# https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.DataSetFilters.warp_by_vector.html#pyvista.DataSetFilters.warp_by_vector
pl.add_mesh(threshedMs.warp_by_vector(vectors='avg(f).pos', factor=1.0),
            opacity=1.0, show_edges=True, line_width=1, cmap=cmap)
# pl.add_mesh(threshedMs, opacity=0.90, show_edges=True, line_width=1, cmap=cmap) # functional

pl.background_color = "white"
pl.remove_scalar_bar()

if nameTag == '':
    pl.screenshot(filename.split('.')[0] + '.png', window_size=[1860*6, 968*6])
else:
    pl.screenshot(filename.split('.')[0] + '_' +
                  nameTag + '.png', window_size=[1860*6, 968*6])
