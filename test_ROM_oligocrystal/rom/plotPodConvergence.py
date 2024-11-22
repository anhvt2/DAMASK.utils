
from natsort import natsorted, ns
# import pyvista
# import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import glob, os, time
import numpy as np
# from distutils.util import strtobool
import logging
# import pandas as pd
# from scipy.interpolate import interpn
# from sklearn.metrics import r2_score
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
cmap = plt.cm.get_cmap('coolwarm')
from matplotlib.ticker import LogLocator, ScalarFormatter

level    = logging.INFO
format   = '  %(message)s'
logFileName = 'plotPodConvergence.py.log'
os.system('rm -fv %s' % logFileName)
handlers = [logging.FileHandler(logFileName), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)

t_start = time.time()

x_test       = np.loadtxt('inputRom_Test.dat',  delimiter=',', skiprows=1)
DamaskIdxs   = x_test[:,5].astype(int)
PostProcIdxs = x_test[:,6].astype(int)
NumCases = len(DamaskIdxs)

t_local = time.time()
SolidIdx = np.loadtxt('SolidIdx.dat', dtype=int)
NumFtrs = [1,2,4,8,16,32,64,128,256]

if not os.path.exists('PodConvergenceMean_Rmse.npy') or not os.path.exists('PodConvergenceStd_Rmse.npy'):
    # Initialize
    mean_rmse, std_rmse = np.zeros([len(NumFtrs), 2]), np.zeros([len(NumFtrs), 2])
    for j, NumFtr in zip(range(len(NumFtrs)), NumFtrs):
        NumObs = 0
        # Initialize
        aeCauchy, aeLnV = [], []
        # Read every case
        SelIdxs = np.sort(np.random.randint(0, high=NumCases, size=100, dtype=int))
        for i in SelIdxs:
            predFileName = '../damask/%d/postProc/pred_main_tension_inc%s_NumFtr_%d.npy' % (DamaskIdxs[i], str(PostProcIdxs[i]).zfill(2), NumFtr)
            trueFileName = '../damask/%d/postProc/main_tension_inc%s.npy' % (DamaskIdxs[i], str(PostProcIdxs[i]).zfill(2))
            if os.path.exists(predFileName) and os.path.exists(trueFileName):
                logging.info(f'Processing NumFtr={NumFtr}, {i+1:<d}/{NumCases} folders...')
                pred = np.load(predFileName)[SolidIdx,:]
                true = np.load(trueFileName)[SolidIdx,:]
                ae = np.abs(pred - true)
                _aeCauchy, _aeLnV = ae[:,0], ae[:,1]
                aeCauchy += [list(_aeCauchy)]
                aeLnV += [list(_aeLnV)]
        aeCauchy, aeLnV = np.array(aeCauchy), np.array(aeLnV)
        mean_rmse[j,0], mean_rmse[j,1] = np.sqrt(np.mean(np.sum(aeCauchy**2))), np.sqrt(np.mean(np.sum(aeLnV**2)))
        std_rmse[j,0], std_rmse[j,1] = np.std(aeCauchy), np.std(aeLnV)
    np.save('PodConvergenceMean_Rmse', mean_rmse)
    np.save('PodConvergenceStd_Rmse', std_rmse)

RmseMean = np.load('PodConvergenceMean_Rmse.npy')
RmseStd  = np.load('PodConvergenceStd_Rmse.npy')

fois = ["Mises(Cauchy)", "Mises(LnV)"]
filetags = ["MisesCauchy", "MisesLnV"]
titles = [r'POD convergence for $\sigma_{vM}$', r'POD convergence for $\varepsilon_{vM}$']

for j, foi, filetag, title in zip(range(2), fois, filetags, titles):
    plt.figure(figsize=(12,12))
    plt.errorbar(NumFtrs, RmseMean[:,j], yerr=RmseStd[:,j], marker='o', linewidth=2, capsize=3, markersize=5)
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.gca().xaxis.set_minor_formatter(ScalarFormatter())
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
    plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=10))
    plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=10))
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.title(title, fontsize=24)
    plt.xlabel(r'Number of POD modes', fontsize=24)
    plt.ylabel(r'RMSE', fontsize=24)
    plt.savefig(f'PodConvergence_{filetag}.png', dpi=None, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)


logging.info(f'plotPodConvergence.py: Total elapsed time: {time.time() - t_start} seconds.')


