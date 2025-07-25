

import pyvista
import matplotlib.pyplot as plt
import glob, os
import argparse
import gc
parser = argparse.ArgumentParser()

parser.add_argument("-n", "--nameTag", help='append to fileName', type=str, default='', required=False)
parser.add_argument("-f", "--fileName", type=str, required=True)
args = parser.parse_args()
nameTag = args.nameTag
fileName = args.fileName

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

# for fileName in glob.glob('masked*.vti'): # screenshot for all *.vtr files

reader = pyvista.get_reader(fileName)
msMesh = reader.read()
ms = msMesh.get_array('Spin')
msMesh.cell_data['Spin']
msMesh.set_active_scalars('Spin', preference='cell')
threshed = msMesh.threshold(value=1.0+1e-3)
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

