
import pyvista
import matplotlib.pyplot as plt
import glob, os
import argparse
import numpy as np

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

reader = pyvista.get_reader('main_tension_inc16_pos(cell).vtr')
msMesh = reader.read()
msMesh.set_active_scalars('texture', preference='cell')
threshedMs = msMesh.threshold(value=(4807,8685), scalars='texture')
threshedMs.set_active_scalars('Mises(Cauchy)', preference='cell')

pl = pyvista.Plotter(off_screen=True)
pl.add_mesh(threshedMs.warp_by_vector(), opacity=0.90, show_edges=True, line_width=1, cmap=cmap)
# pl.add_mesh(threshedMs, opacity=0.90, show_edges=True, line_width=1, cmap=cmap) # functional
# pl.add_mesh(msMesh, opacity=0.90, show_edges=True, line_width=1)
pl.background_color = "white"
pl.remove_scalar_bar()
pl.screenshot('test.png', window_size=[1860*6,968*6])
