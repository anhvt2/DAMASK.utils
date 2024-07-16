
import pyvista
import matplotlib.pyplot as plt
import glob, os
import numpy as np
import argparse
parser = argparse.ArgumentParser()

'''
    This util script uses PyVista to visualize and create a 3d stress fields, deformed by a displacement field

    Parameters
    ----------
    fileName: str
    e.g. 'main_tension_inc16_pos(cell)_added.vtr'

    Output
    ------
    an image: fileName.split('.')[0] + '_' + nameTag + '.png'

    How to use
    ----------
    python3 ../../../plotStress3d.py  --fileName='main_tension_inc16_pos(cell)_added.vtr'
'''

parser.add_argument("-f", "--fileName", help='.vtr file', type=str, default='', required=True) 
parser.add_argument("-n", "--nameTag", help='append to fileName', type=str, default='', required=False) 
parser.add_argument("-show_edges", "--show_edges", help='append to fileName', type=bool, default=True, required=False)
args = parser.parse_args()
fileName = args.fileName
nameTag = args.nameTag
show_edges = args.show_edges

nameTag = nameTag.split('/')[0]
print(nameTag)


# cmap = plt.cm.get_cmap("viridis", 5)
# https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
# https://matplotlib.org/stable/users/explain/colors/colormaps.html
# Ranking: (1) 'coolwarm', (2) 'ocean', (3) 'plasma' or 'inferno' or 'viridis'
cmap = plt.cm.get_cmap('coolwarm')
# cmap = plt.cm.get_cmap('RdBu')
# cmap = plt.cm.get_cmap('viridis')
# cmap = plt.cm.get_cmap('plasma')
# cmap = plt.cm.get_cmap('inferno')
# cmap = plt.cm.get_cmap('ocean')
# cmap = plt.cm.get_cmap('gnuplot2')

# https://matplotlib.org/cmocean/#thermal
# import cmocean
# cmap = cmocean.cm.phase

pl = pyvista.Plotter(off_screen=True)
reader = pyvista.get_reader(fileName) # a *.vtr or *.vti
msMesh = reader.read()
msMesh.set_active_scalars('microstructure', preference='cell')
try:
    grainInfo = np.loadtxt('../grainInfo.dat')
except:
    grainInfo = np.loadtxt('./grainInfo.dat')

## warped by deforming geometry with displacement field
# https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.DataSetFilters.warp_by_vector.html#pyvista.DataSetFilters.warp_by_vector
# args_cbar = dict(height=0.75, vertical=True, position_x=0.25, position_y=0.15,
#                  title_font_size=144, label_font_size=96, 
#                  color='k') 
args_cbar = dict(height=0.05, vertical=False, position_x=0.25, position_y=0.025,
                 title_font_size=144, label_font_size=96, 
                 color='k') 

# see more options at https://docs.pyvista.org/version/stable/_downloads/3ee46f61736bb2769dbf5ed73c33d5dc/scalar-bars.py

msMesh.set_active_scalars('microstructure', preference='cell')
threshedMs = msMesh.threshold(value=(grainInfo[3],grainInfo[4]), scalars='microstructure')
pl.add_mesh(threshedMs, opacity=0.05, show_edges=True, line_width=0.01, cmap=cmap) # show original 

threshedMsVoid = msMesh.threshold(value=(grainInfo[1],grainInfo[2]), scalars='microstructure')
pl.add_mesh(threshedMsVoid, opacity=0.95, show_edges=False, line_width=0.01, cmap=cmap) # show original 
pl.background_color = "white"
pl.remove_scalar_bar()
# labels = dict(xlabel='X', ylabel='Y', zlabel='Z', color='black', line_width=5)
# pl.add_axes(**labels)
if nameTag == '':
    pl.screenshot(fileName.split('.')[0] + '.png', window_size=[1860*6,968*6])
else:
    pl.screenshot(fileName.split('.')[0] + '_' + nameTag + '.png', window_size=[1860*6,968*6])

