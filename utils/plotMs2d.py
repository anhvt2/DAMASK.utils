#!/usr/bin/env python3


import pyvista
import matplotlib.pyplot as plt
import glob, os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-n", "--nameTag", help='append to filename', type=str, default='', required=False)
args = parser.parse_args()
nameTag = args.nameTag

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

# filename = 'single_phase_equiaxed_8x8x8.vtr'
for filename in glob.glob('*.vtr'): # screenshot for all *.vtr files
	reader = pyvista.get_reader(filename)
	# camera = pyvista.Camera()
	msMesh = reader.read()
	ms = msMesh.get_array('microstructure')
	msMesh.cell_data['microstructure']
	msMesh.set_active_scalars('microstructure', preference='cell')

	# pl = pyvista.Plotter()
	pl = pyvista.Plotter(off_screen=True)
	pl.add_mesh(msMesh, show_edges=False, line_width=1, cmap=cmap)
	pl.background_color = "white"
	# light = pyvista.Light(position=(16, 16, 10), color='white')
	# light.positional = True
	pl.view_xy()
	pl.remove_scalar_bar()
	# pl.add_light(light)
	# pl.show(screenshot='%s.png' % filename.split('.')[0])
	# pl.show()
	# pl.close()
	if nameTag == '':
		pl.screenshot(filename.split('.')[0] + '.png', window_size=[1860*3,968*3])
		# pl.screenshot(filename.split('.')[0] + '.png', window_size=[3200,3200])
	else:
		pl.screenshot(filename.split('.')[0] + '_' + nameTag + '.png', window_size=[968*3,968*3])


