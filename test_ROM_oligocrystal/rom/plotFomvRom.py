
from natsort import natsorted, ns
import pyvista
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob, os, time
import numpy as np
import argparse
from distutils.util import strtobool
import logging
import pandas as pd
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
cmap = plt.cm.get_cmap('coolwarm')

t_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--DamaskCase", help='specific DAMASK simulation (1<i<1000)', type=int, default='', required=True) 
args = parser.parse_args()
DamaskCase = args.DamaskCase

grainInfo = np.loadtxt('grainInfo.dat')
x_test       = np.loadtxt('inputRom_Test.dat',  delimiter=',', skiprows=1)
DamaskIdxs   = x_test[:,5].astype(int)
PostProcIdxs = x_test[:,6].astype(int)
NumCases = len(DamaskIdxs)
SolidIdx = np.loadtxt('SolidIdx.dat', dtype=int)

pl = pyvista.Plotter(off_screen=True)
# reader = pyvista.get_reader('main_tension_inc16_pos(cell).vtr')

ms   = np.load('main.npy')
try:
    grid = pyvista.UniformGrid() # old pyvista
except:
    grid = pyvista.ImageData() # new pyvista
grid.dimensions = np.array(ms.shape) + 1
grid.origin = (0, 0, 0)     # The bottom left corner of the data set
grid.spacing = (1, 1, 1)    # These are the cell sizes along each axis
grid.cell_data["microstructure"] = ms.flatten(order="F") # ImageData()

# for damaskNpyFile in natsorted(glob.glob('main_tension_inc??.npy')):
for i in range(NumCases):
    if PostProcIdxs[i] == 19 and DamaskIdxs[i] == DamaskCase:
        # Load and parse FOM/ROM data
        true = np.load('../damask/%d/postProc/pred_main_tension_inc%s.npy' % (DamaskIdxs[i], str(PostProcIdxs[i]).zfill(2)))
        pred = np.load('../damask/%d/postProc/pred_main_tension_inc%s.npy' % (DamaskIdxs[i], str(PostProcIdxs[i]).zfill(2)))
        MisesCauchy = np.hstack((true[SolidIdx,0], pred[SolidIdx,0]))
        MisesLnV    = np.hstack((true[SolidIdx,1], pred[SolidIdx,1]))
        # Assign data
        grid.cell_data["Mises(Cauchy)-FOM"] = true[:,0]
        grid.cell_data["Mises(LnV)-FOM"]    = true[:,1]
        grid.cell_data["Mises(Cauchy)-ROM"] = pred[:,0]
        grid.cell_data["Mises(LnV)-ROM"]    = pred[:,1]
        grid.cell_data["L1Error(Cauchy)"]   = np.abs(pred[:,0] - true[:,0])
        grid.cell_data["L1Error(LnV)"]      = np.abs(pred[:,1] - true[:,1])
        # Set limits for colorbar
        climMisesCauchy = (np.min(np.abs(MisesCauchy)), np.max(np.abs(MisesCauchy)))
        climMisesLnV = (np.min(np.abs(MisesLnV)), np.max(np.abs(MisesLnV)))
        climL1ErrorCauchy = (np.min(grid.cell_data["L1Error(Cauchy)"]), np.max(grid.cell_data["L1Error(Cauchy)"]))
        climL1ErrorLnV = (np.min(grid.cell_data["L1Error(LnV)"]), np.max(grid.cell_data["L1Error(LnV)"]))
        clims = [climMisesCauchy, climMisesLnV, climMisesCauchy, climMisesLnV, climL1ErrorCauchy, climL1ErrorLnV]

        fois = ["Mises(Cauchy)-FOM", "Mises(LnV)-FOM", "Mises(Cauchy)-ROM", "Mises(LnV)-ROM", "L1Error(Cauchy)", "L1Error(LnV)"]
        filenames = ["MisesCauchy-FOM", "MisesLnV-FOM", "MisesCauchy-ROM", "MisesLnV-ROM", "L1ErrorCauchy", "L1ErrorLnV"]

        for foi, clim, filename in zip(fois, clims, filenames):
            threshedGrid = grid.threshold(value=(grainInfo[3],grainInfo[4]), scalars='microstructure')
            threshedGrid.set_active_scalars(foi, preference='cell')
            args_cbar = dict(height=0.75, width=0.05, vertical=True, 
                            position_x=0.75, position_y=0.10,
                            title_font_size=144, label_font_size=96, 
                            color='k') 
            pl.add_mesh(threshedGrid, opacity=1.0, show_edges=False, line_width=1, cmap=cmap, scalar_bar_args=args_cbar, log_scale=True, clim=clim)
            pl.background_color = "white"
            pl.hide_axes()
            pl.screenshot(f'png/damask-{DamaskIdxs[i]:<d}-inc{str(PostProcIdxs[i]).zfill(2)}-{filename}.png', window_size=[1860*6,968*6])
            pl.clear()
            print(f'Finished damask/{DamaskIdxs[i]:<d}/inc{str(PostProcIdxs[i]).zfill(2)}/{filename}.png')

print(f'Elapsed: {time.time() - t_start} seconds.')