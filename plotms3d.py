

import pyvista
import matplotlib.pyplot as plt
# cmap = plt.cm.get_cmap("viridis", 5)
# https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
cmap = plt.cm.get_cmap('coolwarm')

filename = 'single_phase_equiaxed_8x8x8.vtr'
reader = pyvista.get_reader(filename)
msMesh = reader.read()
ms = msMesh.get_array('microstructure')
msMesh.cell_data['microstructure']
msMesh.set_active_scalars('microstructure', preference='cell')

# pl = pyvista.Plotter()
pl = pyvista.Plotter(off_screen=True)
pl.add_mesh(msMesh, show_edges=True, line_width=5, cmap=cmap)
pl.background_color = "white"
pl.remove_scalar_bar()
# pl.show(screenshot='%s.png' % filename.split('.')[0])
# pl.show()
pl.screenshot('%s.png' % filename.split('.')[0])
# pl.close()


