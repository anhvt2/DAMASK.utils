
"""
This script 
	(1) converts a series of microstructures formatted in *.npy (must be)
	(2) along with a chosen phase (for example, dogbone)
	(3) to a series of images (that could be converted to video) for illustration purpose

* Work for any voxelized-STL file.

How?
- Convert *.vti to *.npy
- Dump masked *.npy based on (1) phase and (2) difference with the initial condition *.npy
- Convert masked *.npy to *.geom
- Convert *.geom to *.vtr

Show us with threshold, hide the rest.
- Always show the final us (with opacity=0 for consistent grain ID colormap)
- Only show the diff b/w the current us and the initial us, but NOT include masked phase
"""

import numpy as np
import pyvista
import matplotlib.pyplot as plt
import glob, os
import argparse
import gc
from natsort import natsorted, ns # natural-sort
parser = argparse.ArgumentParser()

parser.add_argument("-n", "--npyFolderName", help='provide folders that supply all *.npy', type=str, default='', required=True)
parser.add_argument("-p", "--phaseFileName", help='provide masked phase', type=str, default='', required=True)
args = parser.parse_args()
npyFolderName = args.npyFolderName # 'npy'
phaseFileName = args.phaseFileName # 'phase_dump_12_out.npy'

npyFolderList = natsorted(glob.glob(npyFolderName + '/*.npy'))
initialVti = np.load(npyFolderList[0])
lastVti = np.load(npyFolderList[-1])
phase = np.load(phaseFileName)

for i in range(len(npyFolderList) - 1):
	currentVti = np.load(npyFolderList[i])
	previousVti = np.load(npyFolderList[i-1])
	nextVti = np.load(npyFolderList[i+1])



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

# for fileName in glob.glob('masked*.vti'): # screenshot for all *.vtr files

reader = pyvista.get_reader(fileName)
msMesh = reader.read()
ms = msMesh.get_array('microstructure')
msMesh.cell_data['microstructure']
msMesh.set_active_scalars('microstructure', preference='cell')
grainInfo = np.loadtxt('grainInfo.dat')
# threshed = msMesh.threshold(value=1.0+1e-3)
threshed = msMesh.threshold(value=(grainInfo[3], grainInfo[4]))
# pl = pyvista.Plotter()
pl = pyvista.Plotter(off_screen=True)
# pl.add_mesh(threshed, show_edges=True, line_width=1, cmap=cmap)
pl.add_mesh(threshed, show_edges=True, line_width=1, cmap=cmap)
pl.background_color = "white"
pl.remove_scalar_bar()
# pl.camera_position = 'xz'
# pl.camera.azimuth = -10
# pl.camera.elevation = +10
# pl.show(screenshot='%s.png' % fileName[:-4])
# pl.show()
if nameTag == '':
	pl.screenshot(fileName[:-4] + '.png', window_size=[1860*6,968*6])
else:
	pl.screenshot(fileName[:-4] + '_' + nameTag + '.png', window_size=[1860*6,968*6])
# pl.close()
gc.collect()

