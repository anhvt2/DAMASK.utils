
import pyvista
import matplotlib.pyplot as plt
import glob, os
import numpy as np
import argparse
from distutils.util import strtobool
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
    # deprecated: python3 ../../../plotStress3d.py  --fileName='main_tension_inc16_pos(cell)_added.vtr'
    python3 ../../plotFieldsGeomDeformed.py --vtr='main_tension_inc19_pos(cell)_added.vtr' --field='Mises(ln(V))' --show_edges='False'
    python3 ../../plotFieldsGeomDeformed.py --vtr='main_tension_inc19_pos(cell)_added.vtr' --field='Mises(Cauchy)' --show_edges='False'
'''

parser.add_argument("-f", "--vtr", help='.vtr file', type=str, default='', required=True) 
parser.add_argument("-field", "--field", help='field to plot', type=str, default=True, required=True) # e.g. 'Mises(ln(V))', 'Mises(Cauchy)'
# parser.add_argument("-n", "--nameTag", help='append to fileName', type=str, default='', required=False) 

parser.add_argument("-show_edges", "--show_edges",
    default=True, help='show mesh edge?',
    type=lambda x:bool(strtobool(x)),
    nargs='?', const=True, required=False)

args = parser.parse_args()
fileName = args.vtr
field = args.field # e.g. 'Mises(ln(V))', 'Mises(Cauchy)'
# nameTag = args.nameTag
show_edges = args.show_edges

# nameTag = nameTag.split('/')[0]
nameTag = field.replace('(', '').replace(')', '')
nameTag = 'MisesLnV' if field == 'Mises(ln(V))' else nameTag

print(nameTag)


# cmap = plt.cm.get_cmap("viridis", 5)
# https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
# https://matplotlib.org/stable/users/explain/colors/colormaps.html
# Ranking: (1) 'coolwarm', (2) 'ocean', (3) 'plasma' or 'inferno' or 'viridis'
# cmap = plt.cm.get_cmap('PiYG')
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

# displacementField = np.loadtxt('main_tension_inc16_nodal.txt', skiprows=3)
# displacementVector = displacementField[:,3]

pl = pyvista.Plotter(off_screen=True)
# reader = pyvista.get_reader('main_tension_inc16_pos(cell).vtr')
reader = pyvista.get_reader(fileName)
msMesh = reader.read()
msMesh.set_active_scalars('texture', preference='cell')
try:
    grainInfo = np.loadtxt('../../grainInfo.dat')
except:
    grainInfo = np.loadtxt('../grainInfo.dat')

## warped by deforming geometry with displacement field
# https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.DataSetFilters.warp_by_vector.html#pyvista.DataSetFilters.warp_by_vector
args_cbar = dict(height=0.75, width=0.05, vertical=True, 
                position_x=0.75, position_y=0.10,
                title_font_size=144, label_font_size=96, 
                color='k') 

# see more options at https://docs.pyvista.org/version/stable/_downloads/3ee46f61736bb2769dbf5ed73c33d5dc/scalar-bars.py


threshedMs = msMesh.threshold(value=(grainInfo[3],grainInfo[4]), scalars='texture')
threshedMs.set_active_scalars(field, preference='cell')

msMesh.set_active_scalars(field, preference='cell')
pl.add_mesh(threshedMs, opacity=0.05, show_edges=show_edges, line_width=0.01, scalar_bar_args=args_cbar) 

pl.add_mesh(threshedMs.warp_by_vector(vectors='avg(f).pos', factor=1.0), opacity=1.0, show_edges=show_edges, line_width=1, cmap=cmap, scalar_bar_args=args_cbar, log_scale=True) # add scalar_bar_args
# pl.add_mesh(threshedMs.warp_by_vector(vectors='avg(f).pos', factor=1.0), opacity=1.0, show_edges=show_edges, line_width=1, cmap=cmap)
# pl.add_mesh(threshedMs, opacity=0.90, show_edges=show_edges, line_width=1, cmap=cmap) # functional

pl.background_color = "white"
# pl.remove_scalar_bar()
# add_scalar_bar
labels = dict(xlabel='X', ylabel='Y', zlabel='Z', color='black', line_width=5)
# pl.add_axes(**labels)
pl.hide_axes()
# https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.Plotter.add_axes.html#pyvista.Plotter.add_axes
# pl.show_grid(**labels)
# p.add_axes(x_color='pink', y_color='navy', z_color='tan', line_width=5)

if nameTag == '':
    pl.screenshot(fileName.split('.')[0] + '.png', window_size=[1860*6,968*6])
else:
    pl.screenshot(fileName.split('.')[0] + '_' + nameTag + '.png', window_size=[1860*6,968*6])

