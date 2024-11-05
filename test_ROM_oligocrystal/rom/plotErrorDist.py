import matplotlib.pyplot as plt
import glob, os, time
import numpy as np
import matplotlib as mpl
import logging
import pandas as pd
cmap = plt.cm.get_cmap('coolwarm')

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

level    = logging.INFO
format   = '  %(message)s'
logFileName = 'plotErrorDist.py.log'
os.system('rm -fv %s' % logFileName)
handlers = [logging.FileHandler(logFileName), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)

controlInfo = np.loadtxt('control.log', skiprows=1, delimiter=',')
dotVarEps = controlInfo[:,1]
loadingTime = controlInfo[:,2] # dependent - not an input
initialT = controlInfo[:,3]

TrainIdx   = np.loadtxt('TrainIdx.dat', dtype=int)
TestIdx    = np.loadtxt('TestIdx.dat', dtype=int)
TestIdxOOD = np.loadtxt('TestIdxOOD.dat', dtype=int)
TestIdxID  = np.loadtxt('TestIdxID.dat', dtype=int)
fois   = ['MisesCauchy', 'MisesLnV'] # fields of interest
labels = ['test (OOD)','test (ID)']
colors = ['tab:orange','tab:green']
cols   = ['MeanRelError_MisesCauchy', 'MeanRelError_MisesLnV']
titles = [r'Relative Error [%]: $\sigma_{vM}$', r'Relative Error [%]: $\varepsilon_{vM}$']

x_test       = np.loadtxt('inputRom_Test.dat',  delimiter=',', skiprows=1)
dfError      = pd.read_csv('FomRomErrors.dat', skipinitialspace=True)
DamaskIdxs   = x_test[:,5].astype(int)
PostProcIdxs = x_test[:,6].astype(int)
NumCases = len(DamaskIdxs)

# Augment dfError with error_type = 'OOD' or 'ID'
error_types = []
for i in range(dfError.shape[0]):
    if dfError['DamaskIndex'].iloc[i] in TestIdxOOD:
        error_types += ['OOD']
    elif dfError['DamaskIndex'].iloc[i] in TestIdxID:
        error_types += ['ID']

dfError['ErrorTypes'] = error_types

dfError19 = dfError[dfError['PostProcIndex'] == 19]
dfError19_OOD = dfError[(dfError['PostProcIndex'] == 19) & (dfError['ErrorTypes'] == 'OOD')]
dfError19_ID  = dfError[(dfError['PostProcIndex'] == 19) & (dfError['ErrorTypes'] == 'ID')]

t_start = time.time()

def plotDataframe(df, marker, label):
    z = df[col]
    sc = plt.scatter(df['dotVareps'], df['initialT'], 
            c=scalarMap.to_rgba(z),
            s=4+(100-4)*(np.log(z) - np.log(z).min()) / (np.log(z).max() - np.log(z).min()),
            vmin=z.min(),
            vmax=z.max(),
            marker=marker, 
            alpha=1.0,
            label=label)
    return sc

# foi, label, col = fois[0], labels[0], cols[0] # debug
for foi, label, col, title in zip(fois, labels, cols, titles):
    # Get column data of interest
    z = dfError19[col]
    # Normalize colors
    cNorm = mpl.colors.Normalize(vmin=z.min(), vmax=z.max())
    LogNorm = mpl.colors.LogNorm(vmin=z.min(), vmax=z.max())
    # scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap) # linear colorbar
    scalarMap = mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=z.min(), vmax=z.max()), cmap=cmap) # log colorbar
    fig, ax = plt.subplots(num=None, figsize=(20, 20), dpi=300, facecolor='w', edgecolor='k')
    scOOD = plotDataframe(dfError19_OOD, marker='o', label='test (OOD)')
    scID  = plotDataframe(dfError19_ID , marker='h', label='test (ID)')
    # Set legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # leg = plt.legend(by_label.values(), by_label.keys(), fontsize=24, loc='upper left', bbox_to_anchor=(1.05,.0),frameon=False, markerscale=5)
    leg = plt.legend(fontsize=24, loc='upper left', bbox_to_anchor=(1.05, 1.0),frameon=False, markerscale=4)
    for marker in leg.legendHandles:
        marker.set_color('tab:green')
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.LogNorm(vmin=z.min(), vmax=z.max()))
    sm.set_clim(vmin=z.min(), vmax=z.max())
    cbar = fig.colorbar(sm, cmap=cmap, orientation='horizontal', aspect=50, pad=0.1)
    # cbar = plt.colorbar(scOOD, cmap=cmap, orientation='horizontal', aspect=50, pad=0.1)
    cbar.set_label('Relative Error [%]', fontsize=24, rotation=0)
    cbar.ax.minorticks_on()
    plt.xscale('log',base=10)
    plt.title(title, fontsize=24)
    plt.xlabel(r'$\dot{\varepsilon}$ [s$^{-1}$]', fontsize=24)
    plt.ylabel(r'$T$ [K]', fontsize=24)
    plt.savefig(f'MeanRelErr{foi}.png', dpi=None, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)


logging.info(f'plotErrorDist.py: Elapsed in {time.time() - t_start} seconds.')

