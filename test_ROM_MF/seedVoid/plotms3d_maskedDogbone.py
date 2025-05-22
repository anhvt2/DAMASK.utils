
import numpy as np
import pyvista
import matplotlib.pyplot as plt
import glob, os
import argparse
import gc
parser = argparse.ArgumentParser()

parser.add_argument("-f", "--filename", type=str, required=True)
parser.add_argument("-n", "--nametag", type=str, default='', required=False)

args = parser.parse_args()
filename = args.filename
nametag  = args.nametag

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

grainInfo = np.loadtxt('grainInfo.dat')

# for filename in glob.glob('masked*.vti'): # screenshot for all *.vtr files

reader = pyvista.get_reader(filename)
msMesh = reader.read()
ms = msMesh.get_array('microstructure')
msMesh.cell_data['microstructure']
msMesh.set_active_scalars('microstructure', preference='cell')

# pl = pyvista.Plotter()
pl = pyvista.Plotter(off_screen=True)
print(f'nametag = {nametag}')

if nametag == 'voids':
    pl.add_mesh(msMesh.threshold(value=1.0+1e-3), show_edges=True, line_width=1, cmap=cmap, opacity=0.2) # enable dogbone background -- only when plotting voids
    threshed = msMesh.threshold(value=(grainInfo[1], grainInfo[2])) # plot voids
    pl.add_mesh(threshed, show_edges=True, line_width=1, cmap=cmap)
elif nametag == 'solids':
    threshed = msMesh.threshold(value=(grainInfo[3], grainInfo[4])) # plot solids
    pl.add_mesh(threshed, show_edges=True, line_width=1, cmap=cmap) # plot solids
else:
    threshed = msMesh.threshold(value=1.0+1e-3) # general settings
    pl.add_mesh(threshed, show_edges=True, line_width=1, cmap=cmap)


pl.background_color = "white"
pl.remove_scalar_bar()
# pl.camera_position = 'xz'
# pl.camera.azimuth = -10
# pl.camera.elevation = +10
# pl.show(screenshot='%s.png' % filename[:-4])
# pl.show()
# pl.add_axes(color='k')
# pl.add_axes(line_width=5,cone_radius=0.6,shaft_length=0.7,tip_length=0.3,ambient=0.5,label_size=(0.4, 0.16), color='k')
# pl.add_axes_at_origin()
# pl.show_axes() # https://docs.pyvista.org/api/plotting/_autosummary/pyvista.renderer.add_axes
pl.store_image = True
# pl.show()

if nametag == '':
    pl.screenshot(filename[:-4] + '.png', window_size=[1860*6,968*6])
else:
    pl.screenshot(filename[:-4] + '_' + nametag + '.png', window_size=[1860*6,968*6])
# pl.close()
gc.collect()

