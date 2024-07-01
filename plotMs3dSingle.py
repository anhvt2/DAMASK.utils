

import pyvista
import matplotlib.pyplot as plt
import glob, os
import argparse
parser = argparse.ArgumentParser()

'''
    This util script uses PyVista to visualize and create a 3d microstructure image

    Parameters
    ----------

    Output
    ------
    an image: fileName.split('.')[0] + '_' + nameTag + '.png'
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

# fileName = 'single_phase_equiaxed_8x8x8.vtr'
# for fileName in glob.glob('*.vtr'): # screenshot for all *.vtr files

reader = pyvista.get_reader(fileName)
msMesh = reader.read()
# print(msMesh.array_names)
ms = msMesh.get_array('microstructure')
msMesh.cell_data['microstructure']
msMesh.set_active_scalars('microstructure', preference='cell')

# pl = pyvista.Plotter()
pl = pyvista.Plotter(off_screen=True)
pl.add_mesh(msMesh, show_edges=show_edges, line_width=1, cmap=cmap)
# pl.add_mesh(msMesh.threshold(0.1), show_edges=show_edges, line_width=1, cmap=cmap)
pl.background_color = "white"
pl.remove_scalar_bar()
# pl.camera_position = 'yz'
# pl.camera.elevation += 25
# pl.camera.roll += 0
# pl.camera.azimuth += 25
# pl.show(screenshot='%s.png' % fileName.split('.')[0])
# pl.show()
if nameTag == '':
    # pl.screenshot(fileName.split('.')[0] + '.png', window_size=[1860*6,968*6])
    pl.screenshot(fileName[:-4] + '.png', window_size=[1860*6,968*6])
else:
    # pl.screenshot(fileName.split('.')[0] + nameTag + '.png', window_size=[1860*6,968*6])
    pl.screenshot(fileName[:-4] + '_' + nameTag + '.png', window_size=[1860*6,968*6])
# pl.close()


