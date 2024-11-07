
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
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
cmap = plt.cm.get_cmap('coolwarm')
# cmap = plt.cm.get_cmap('RdBu_r')
# cmap = plt.cm.get_cmap('Greys')

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
        # print(i) # debug: e.g. i = 3832
        # Load and parse FOM/ROM data
        true = np.load('../damask/%d/postProc/main_tension_inc%s.npy' % (DamaskIdxs[i], str(PostProcIdxs[i]).zfill(2)))
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
        climL1ErrorCauchy = (np.min(grid.cell_data["L1Error(Cauchy)"][SolidIdx]), np.max(grid.cell_data["L1Error(Cauchy)"][SolidIdx]))
        climL1ErrorLnV = (np.min(grid.cell_data["L1Error(LnV)"][SolidIdx]), np.max(grid.cell_data["L1Error(LnV)"][SolidIdx]))
        clims = [climMisesCauchy, climMisesLnV, climMisesCauchy, climMisesLnV, climL1ErrorCauchy, climL1ErrorLnV]
        # Assign list to iterate
        fois = ["Mises(Cauchy)-FOM", "Mises(LnV)-FOM", "Mises(Cauchy)-ROM", "Mises(LnV)-ROM", "L1Error(Cauchy)", "L1Error(LnV)"]
        filenames = ["MisesCauchy-FOM", "MisesLnV-FOM", "MisesCauchy-ROM", "MisesLnV-ROM", "L1ErrorCauchy", "L1ErrorLnV"]

        for foi, clim, filename in zip(fois, clims, filenames):
            threshedGrid = grid.threshold(value=(grainInfo[3],grainInfo[4]), scalars='microstructure')
            threshedGrid.set_active_scalars(foi, preference='cell')
            args_cbar = dict(height=0.75, width=0.05, vertical=True, 
                            position_x=0.75, position_y=0.10,
                            title_font_size=144, label_font_size=96, 
                            color='k') 
            if 'Cauchy' in foi:
                # set log_scale = True for MisesCauchy
                pl.add_mesh(threshedGrid, opacity=1.0, show_edges=False, line_width=1, cmap=cmap, scalar_bar_args=args_cbar, log_scale=True, clim=clim)
            elif 'LnV' in foi: 
                # set log_scale = False for MisesLnV
                pl.add_mesh(threshedGrid, opacity=1.0, show_edges=False, line_width=1, cmap=cmap, scalar_bar_args=args_cbar, log_scale=True, clim=clim)
            else:
                print(f'Check pl.add_mesh() for {foi}!')
            pl.background_color = "white"
            pl.hide_axes()
            pl.screenshot(f'png/damask-{DamaskIdxs[i]:<d}-inc{str(PostProcIdxs[i]).zfill(2)}-{filename}.png', window_size=[1860*6,968*6])
            pl.clear()
            print(f'Finished damask/{DamaskIdxs[i]:<d}/inc{str(PostProcIdxs[i]).zfill(2)}/{filename}.png')


# Scatter plot - adopted from eshelby-3d/plotScatterDensityTrain.py

def r2(y,f):
    '''
    https://en.wikipedia.org/wiki/Coefficient_of_determination
    Parameters
    ----------
    y: observations
    f: predictions
    '''
    ymean = np.mean(y)
    ssRes = np.sum((y - f)**2)
    ssTot = np.sum((y - ymean)**2)
    r2Score = 1 - ssRes / ssTot
    return r2Score

def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    plt.gcf().set_size_inches(14, 14)
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)
    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    cmap = plt.cm.get_cmap('coolwarm')
    ax.scatter( x, y, c=z, cmap=cmap, **kwargs )
    # norm = Normalize(vmin = np.max([np.min(z),0]), vmax = np.max(z))
    norm = LogNorm(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap=cmap), ax=ax)
    # cbar.ax.set_ylabel('density', fontsize=18)
    # cbar.set_ticks([]) # https://stackoverflow.com/questions/56094845/matplotlib-remove-the-ticks-axis-from-the-colorbar
    # cbar.remove() # remove colorbar
    return ax

pl.close()
plt.close()

grid.cell_data["Mises(Cauchy)-FOM"] = true[:,0]
grid.cell_data["Mises(LnV)-FOM"]    = true[:,1]
grid.cell_data["Mises(Cauchy)-ROM"] = pred[:,0]
grid.cell_data["Mises(LnV)-ROM"]    = pred[:,1]

fois = ['Mises(Cauchy)', 'Mises(LnV)']
filenames = [f'damask-{DamaskCase}-Scatter-MisesCauchy', f'damask-{DamaskCase}-Scatter-MisesLnV']
titles = [r'$\sigma_{vM}$', r'$\varepsilon_{vM}$']
js = [0,1] # column index

for foi, filename, j, title in zip(fois, filenames, js, titles):
    fig = plt.figure(num=None, figsize=(16, 9), dpi=400, facecolor='w', edgecolor='k') # screen size
    y = np.hstack((true[SolidIdx,j], pred[SolidIdx,j]))
    refs = np.linspace(np.min(y), np.max(y), num=100)
    plt.plot(refs, refs, linewidth=0.25, alpha=0.2, c='k')
    density_scatter(true[SolidIdx,j], pred[SolidIdx,j], bins=[50,50])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('FOM', fontsize=24)
    plt.ylabel('ROM', fontsize=24)
    plt.title(title + f': $R^2$ = {r2(true[:,j], pred[:,j]):<.4f}')
    plt.savefig(f'png/{filename}.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
    plt.clf()
    plt.close()

print(f'Elapsed: {time.time() - t_start} seconds.')

