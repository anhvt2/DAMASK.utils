
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
from matplotlib.colors import Normalize, LogNorm
from scipy.interpolate import interpn
from sklearn.metrics import r2_score
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
cmap = plt.cm.get_cmap('coolwarm')
# cmap = plt.cm.get_cmap('RdBu_r')
# cmap = plt.cm.get_cmap('Greys')

t_start = time.time()

grainInfo = np.loadtxt('grainInfo.dat')
x_test       = np.loadtxt('inputRom_Test.dat',  delimiter=',', skiprows=1)
DamaskIdxs   = x_test[:,5].astype(int)
PostProcIdxs = x_test[:,6].astype(int)
NumCases = len(DamaskIdxs)
SolidIdx = np.loadtxt('SolidIdx.dat', dtype=int)
podBasis_MisesCauchy = np.load('podBasis_MisesCauchy.npy')
podBasis_MisesLnV = np.load('podBasis_MisesLnV.npy')

ms   = np.load('main.npy')
try:
    grid = pyvista.UniformGrid() # old pyvista
except:
    grid = pyvista.ImageData() # new pyvista

grid.dimensions = np.array(ms.shape) + 1
grid.origin = (0, 0, 0)     # The bottom left corner of the data set
grid.spacing = (1, 1, 1)    # These are the cell sizes along each axis
grid.cell_data["microstructure"] = ms.flatten(order="F") # ImageData()
NumPodModes = 10 # Set the number of POD modes

for i in range(NumPodModes):
    # Assign data
    grid.cell_data[f"Mises(Cauchy)-POD Mode {i+1:<d}"] = np.zeros([576000])
    grid.cell_data[f"Mises(LnV)-POD Mode {i+1:<d}"] = np.zeros([576000])
    grid.cell_data[f"Mises(Cauchy)-POD Mode {i+1:<d}"][SolidIdx] = podBasis_MisesCauchy[:,i]
    grid.cell_data[f"Mises(LnV)-POD Mode {i+1:<d}"][SolidIdx]    = podBasis_MisesLnV[:,i]

grid.save(f'png/PodModes.vtk')

for i in range(NumPodModes):
    fois = ["Mises(Cauchy)", "Mises(LnV)"]
    filetags = ["MisesCauchy", "MisesLnV"]
    for foi, filetag in zip(fois, filetags):
        # Set associated field and file names
        fieldName = f"{foi}-POD Mode {i+1:<d}"
        filename  = f"{filetag}-POD-Mode-{i+1:<d}"
        # Plot
        pl = pyvista.Plotter(off_screen=True)
        threshedGrid = grid.threshold(value=(grainInfo[3],grainInfo[4]), scalars='microstructure')
        threshedGrid.set_active_scalars(fieldName, preference='cell')
        args_cbar = dict(vertical=False, 
                        height=0.05, width=0.5, 
                        position_x=0.25, position_y=0.00,
                        title_font_size=144, label_font_size=96, 
                        color='k') 
        pl.add_mesh(threshedGrid, opacity=1.0, show_edges=False, line_width=1, cmap=cmap, scalar_bar_args=args_cbar, log_scale=False)
        pl.background_color = "white"
        pl.hide_axes()
        pl.screenshot(f'png/damask-{filename}.png', window_size=[1860*6,968*6])
        pl.clear()
        print(f'Finished png/damask-{filename}.png')

print(f'Elapsed: {time.time() - t_start} seconds.')

hang()
