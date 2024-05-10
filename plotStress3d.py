
import pyvista
reader = pyvista.get_reader('main_tension_inc16_pos(cell).vtr')
msMesh = reader.read()
msMesh.set_active_scalars('texture', preference='cell')
threshedMs = msMesh.threshold(value=(4807,865), scalars='texture')
threshedMs.set_active_scalars('Mises(Cauchy)', preference='cell')

pl = pyvista.Plotter(off_screen=True)
pl.add_mesh(threshedMs, opacity=0.90, show_edges=True, line_width=1)
pl.background_color = "white"
pl.remove_scalar_bar()
pl.screenshot('test.png', window_size=[1860*6,968*6])
