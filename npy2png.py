
# https://stackoverflow.com/questions/77696805/pyvista-3d-visualization-issue
# https://tutorial.pyvista.org/tutorial/02_mesh/solutions/c_create-uniform-grid.html#sphx-glr-tutorial-02-mesh-solutions-c-create-uniform-grid-py

import numpy as np
import pyvista
import matplotlib.pyplot as plt
import glob, os
import argparse
from distutils.util import strtobool # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
import gc
from natsort import natsorted, ns # natural-sort
parser = argparse.ArgumentParser()

parser.add_argument("-n", "--npy", help='*.npy file', type=str, default='', required=True)
parser.add_argument("-threshold", "--threshold", help='threshold', type=int, default=-1, required=False)
parser.add_argument("-nameTag", "--nameTag", help='', type=str, default='', required=False)
parser.add_argument("-show_edges", "--show_edges", help='pyvista show_edges', type=lambda x:bool(strtobool(x)), default=True, required=False, nargs='?', const=True)
args = parser.parse_args()
npyFileName = args.npy # 'npy'
threshold = args.threshold
nameTag = args.nameTag
show_edges = bool(args.show_edges)

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

ms = np.load(npyFileName)
grid = pyvista.UniformGrid() # old pyvista
# grid = pyvista.ImageData() # new pyvista
# grid = pyvista.RectilinearGrid()
# grid['microstructure'] = ms
grid.dimensions = np.array(ms.shape) + 1
grid.origin = (0, 0, 0)     # The bottom left corner of the data set
grid.spacing = (1, 1, 1)    # These are the cell sizes along each axis
grid.cell_data["microstructure"] = ms.flatten(order="F") # ImageData()

# reader = pyvista.get_reader(fileName)
# msMesh = reader.read()
# ms = msMesh.get_array('microstructure')
# msMesh.cell_data['microstructure']
# msMesh.set_active_scalars('microstructure', preference='cell')
# grainInfo = np.loadtxt('grainInfo.dat')
# threshed = msMesh.threshold(value=1.0+1e-3)

# pl = pyvista.Plotter()
pl = pyvista.Plotter(off_screen=True)
# pl.add_mesh(grid, scalars='microstructure', show_edges=True, line_width=1, cmap=cmap)
# pl.add_mesh(grid.threshold(value=1e-6), scalars='microstructure', opacity=0.01, show_edges=True, line_width=1, cmap=cmap) # maybe replace by an optional phase for background masking
pl.add_mesh(grid.threshold(value=threshold+1e-6), scalars='microstructure', show_edges=show_edges, line_width=1, cmap=cmap)
pl.background_color = "white"
pl.remove_scalar_bar()
###
# Leidong/Zihan custom view
# pl.camera_position = 'yz'
# pl.camera.elevation += 25
# pl.camera.roll += 0
# pl.camera.azimuth += 25
###
# pl.show(screenshot='%s.png' % fileName[:-4])
# pl.show()
# pl.add_axes(color='k')
# pl.show_axes() # https://docs.pyvista.org/api/plotting/_autosummary/pyvista.renderer.add_axes

if ms.shape[2] == 1:
    pl.camera_position = 'xy'

if nameTag == '':
    pl.screenshot(npyFileName[:-4] + '.png', window_size=[1860*6,968*6])
else:
    pl.screenshot(npyFileName[:-4] + '_' + nameTag + '.png', window_size=[1860*6,968*6])

# pl.close()
gc.collect()


