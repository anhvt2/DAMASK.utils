
from natsort import natsorted, ns
import pyvista
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob, os, time
import numpy as np
from distutils.util import strtobool
import logging
import pandas as pd
from matplotlib.colors import Normalize, LogNorm
from scipy.interpolate import interpn
from sklearn.metrics import r2_score
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
cmap = plt.cm.get_cmap('coolwarm')

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
basis_MisesCauchy = np.load('podBasis_MisesCauchy.npy')
basis_MisesLnV    = np.load('podBasis_MisesLnV.npy')
mean_MisesCauchy  = np.load('mean_MisesCauchy.npy')
mean_MisesLnV     = np.load('mean_MisesLnV.npy')
SolidIdx = np.loadtxt('SolidIdx.dat', dtype=int)
logging.info(f'plotPodConvergence.py: Load POD basis in {time.time() - t_local:<.2f} seconds.')

for NumFtrs in [1,2,4,8,16,32,64,128,256]:
    # Initialize
    trueList = []
    predList = []
    # Read every case
    for i in range(NumCases):
        predFileName = '../damask/%d/postProc/pred_main_tension_inc%s_NumFtrs_%d.npy' % (DamaskIdxs[i], str(PostProcIdxs[i]).zfill(2), NumFtrs)
        trueFileName = '../damask/%d/postProc/main_tension_inc%s.npy' % (DamaskIdxs[i], str(PostProcIdxs[i]).zfill(2))
        if os.path.exists(predFileName) and os.path.exists(trueFileName):
            logging.info(f'Processing NumFtrs={NumFtrs}, {i+1:<d}/{NumCases} folders...')
            pred = np.load(predFileName)
            true = np.load(trueFileName)

logging.info(f'plotPodConvergence.py: Total elapsed time: {time.time() - t_start} seconds.')


