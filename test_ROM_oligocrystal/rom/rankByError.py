import matplotlib.pyplot as plt
import glob, os, time
import numpy as np
import matplotlib as mpl
import logging
import pandas as pd
# cmap = plt.cm.get_cmap('coolwarm')
# cmap = plt.cm.get_cmap('RdBu_r')
# cmap = plt.cm.get_cmap('Reds')
cmap = plt.cm.get_cmap('PuRd')

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
# fois   = ['MisesCauchy', 'MisesLnV'] # fields of interest
# labels = ['test (OOD)','test (ID)']
# colors = ['tab:orange','tab:green']
cols   = ['MeanRelError_MisesCauchy', 'MeanRelError_MisesLnV', 'MeanAbsError_MisesCauchy', 'MeanAbsError_MisesLnV']
titles = [r'Mean Relative Absolute Error [%]: $\sigma_{vM}$', r'Mean Relative Absolute Error [%]: $\varepsilon_{vM}$', r'Mean Absolute Error [MPa]: $\sigma_{vM}$', r'Mean Absolute Error [-]: $\varepsilon_{vM}$']
filenames = ['MRAE-MisesCauchy', 'MRAE-MisesLnV', 'MAE-MisesCauchy', 'MAE-MisesLnV']

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

def printStat(dfError19, idx):
    return dfError19[dfError19['DamaskIndex'] == idx].iloc[0]

# Print ranking
for col in cols:
    dfSorted = dfError19.sort_values(by=col, ascending=False)
    print(f"----------------------------------------------")
    print(f"Ranking by {col}:")
    print(f"Best: {dfSorted.iloc[0]['DamaskIndex']}")
    print(f"Average: {dfSorted.iloc[int(dfSorted.shape[0]/2)]['DamaskIndex']}")
    print(f"Worst: {dfSorted.iloc[-1]['DamaskIndex']}")

print(f"----------------------------------------------")

# Print statistics min/max/avg

fois   = ['MisesCauchy', 'MisesLnV'] # fields of interest
error_types = ['OOD', 'ID']
error_metrics = ['Rel', 'Abs']

for error_metric in error_metrics:
    print(f'Error metric: {error_metric}')
    for foi in fois:
        for error_type in error_types:
            col = f'Mean{error_metric}Error_{foi}'
            subdf = dfError[(dfError['PostProcIndex'] == 19) & (dfError['ErrorTypes'] == error_type)]
            if error_metric == 'Rel':
                print(f'{foi} ({error_type}): {subdf[col].min():<.4e}%, {subdf[col].mean():<.4e}%, {subdf[col].max():<.4e}%')
            elif error_metric == 'Abs':
                print(f'{foi} ({error_type}): {subdf[col].min():<.4e}, {subdf[col].mean():<.4e}, {subdf[col].max():<.4e}')

# print(f"----------------------------------------------")
