
import numpy as np
import glob, os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys, glob, datetime
import pandas as pd
import sklearn
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm # The standard Normal distribution
from scipy.stats import gaussian_kde as GKDE # A standard kernel density estimator

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24 

def getMetaInfo(StressStrainFile):
    fileHandler = open(StressStrainFile)
    txtInStressStrainFile = fileHandler.readlines()
    fileHandler.close()
    try:
        numLinesHeader = int(txtInStressStrainFile[0].split('\t')[0])
        fieldsList = txtInStressStrainFile[numLinesHeader].split('\t')
    except:
        numLinesHeader = int(txtInStressStrainFile[0].split(' ')[0])
        fieldsList = txtInStressStrainFile[numLinesHeader].split(' ')
        fieldsList = list(filter(('').__ne__, fieldsList)) # remove all ''
        print('%s is not natural - i.e. it may have been copied/pasted.' % (StressStrainFile))
    else:
        print('Reading results in %s...' % (StressStrainFile))
    for i in range(len(fieldsList)):
        fieldsList[i] = fieldsList[i].replace('\n', '')
    # print('numLinesHeader = ', numLinesHeader)
    # print('fieldsList = ', fieldsList)
    return numLinesHeader, fieldsList

def readLoadFile(LoadFile):
    load_data = np.loadtxt(LoadFile, dtype=str)
    n_fields = len(load_data)
    # assume uniaxial:
    for i in range(n_fields):
        if load_data[i] == 'Fdot' or load_data[i] == 'fdot':
            print('Found *Fdot*!')
            Fdot11 = float(load_data[i+1])
        if load_data[i] == 'time':
            print('Found *totalTime*!')
            totalTime = float(load_data[i+1])
        if load_data[i] == 'incs':
            print('Found *totalIncrement*!')
            totalIncrement = float(load_data[i+1])
        if load_data[i] == 'freq':
            print('Found *freq*!')
            freq = float(load_data[i+1])
    return Fdot11, totalTime, totalIncrement


def getTrueStressStrain(StressStrainFile):
    numLinesHeader, fieldsList = getMetaInfo(StressStrainFile)
    d = np.loadtxt(StressStrainFile, skiprows=numLinesHeader+1)
    df = pd.DataFrame(d, columns=fieldsList)
    vareps = list(df['Mises(ln(V))'])  # strain -- pad original
    sigma  = list(df['Mises(Cauchy)']) # stress -- pad original
    _, uniq_idx = np.unique(np.array(vareps), return_index=True)
    vareps = np.array(vareps)[uniq_idx]
    sigma  = np.array(sigma)[uniq_idx]
    x = (vareps)
    y = sigma / 1e6
    return x, y

def getInterpStressStrain(StressStrainFile):
    x, y = getTrueStressStrain(StressStrainFile)
    interp_x = np.linspace(x.min(), x.max(), num=100)
    # splineInterp = interp1d(x, y, kind='cubic', fill_value='extrapolate')
    splineInterp = PchipInterpolator(x, y, extrapolate=True)
    interp_y = splineInterp(interp_x)
    return interp_x, interp_y

# Set indices
idxStressStrain = 2
poroType = 'local'
poroIdx = 1

# Read data
tmp = []
for mltype, color, alpha, marker in zip(['train/', 'test/test-run-437-', 'test/test-run-180-', 'test/test-run-90-', 'test/test-run-20-'], ['tab:gray', 'tab:red', 'tab:green', 'tab:orange', 'tab:blue'], [0.75, 0.75, 0.65, 0.55, 0.45], ['o','s','p','h','D']):
    for folderName in glob.glob(f'{mltype}*/'):
        StressStrainFile = folderName + '/' + 'stress_strain.log'
        porosity = np.loadtxt(f'{folderName}/porosity.txt', dtype=float) # 0: global, 1: local, 2: target
        interpStressStrainFile = folderName + '/' + 'interp_stress_strain.log'
        interpStressStrain = np.loadtxt(interpStressStrainFile, skiprows=1, delimiter=',')
        interpStrain, interpStress = interpStressStrain[idxStressStrain,0], interpStressStrain[idxStressStrain,1] # 2 columns: stress, strain, see interp_stress_strain.log
        tmp.append({
            'mltype': 'train' if 'train' in mltype else 'test',
            'testMsIdx': folderName.split('-')[2] if 'test' in mltype else np.nan,
            'testCaseIdx': folderName.split('-')[3].replace('/','') if 'test' in mltype else np.nan,
            'folderName': folderName,
            'interpStrain': interpStrain,
            'interpStress': interpStress,
            'global': porosity[0],
            'local': porosity[1],
            })

df = pd.DataFrame(tmp)

# Build surrogate model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
x = df[(df['mltype'] == 'train')]['local'].to_numpy() 
y = df[(df['mltype'] == 'train')]['interpStress'].to_numpy()
x, y = np.atleast_2d(x).T, np.atleast_2d(y).T
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=np.var(y)*9)
gp.fit(x, y)
x_pred = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_pred, sigma = gp.predict(x_pred, return_std=True)
print("Predicted Values:", y_pred)
print("Uncertainty (Standard Deviation):", sigma)
plt.figure(num=None, figsize=(14, 14), dpi=300, facecolor='w', edgecolor='k')
plt.scatter(x, y, c='r', label='Training data')
plt.plot(x_pred, y_pred, c='b', linewidth=4, label='Prediction')
plt.fill_between(x_pred.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, color='blue', alpha=0.2, label='95% Confidence Interval')
plt.title('Gaussian Process Regression', fontsize=24)
plt.xlabel(r'(local) $\phi$', fontsize=24)
plt.ylabel(r'conditional stress $\sigma | \varepsilon$ at fixed $\varepsilon$', fontsize=24)
plt.legend(fontsize=24)
plt.xlim(left=x.min(), right=x.max())
plt.savefig('gpr.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
plt.clf()

# Solve consistent Bayesian (or DCI)
def QoI(lam, gp): # defing a QoI mapping function
    return gp.predict(lam.reshape(-1, 1), return_std=False)

def rejection_sampling(r):
    N = r.size # size of proposal sample set
    check = np.random.uniform(low=0,high=1,size=N) # create random uniform weights to check r against
    M = np.max(r)
    new_r = r/M # normalize weights 
    idx = np.where(new_r>=check)[0] # rejection criterion
    return idx

N = int(1e5)
xPrior = np.random.uniform(low=x.min(), high=x.max(), size=N)
yPrior = QoI(xPrior, gp)
# testIdx = 90 # ['20', '90', '180', '437']
for testIdx in ['20', '90', '180', '437']:
    yObs = df[df['testMsIdx'] == str(testIdx)]['interpStress'].to_numpy()
    xObs = df[df['testMsIdx'] == str(testIdx)]['local'].to_numpy() # true xObs

    qPrior = GKDE(yPrior)
    qObs  = GKDE(yObs)

    r = np.divide(qObs(QoI(xPrior, gp)), qPrior(yPrior))
    samples_to_keep = rejection_sampling(r)
    xPost = xPrior[samples_to_keep]
    qPost = GKDE(QoI(xPost, gp))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12)) # horizontal stacked subplots
    xPlot = np.linspace(x.min(), x.max(), num=10000)
    qxTrue = GKDE(xObs)
    qxTest = GKDE(xPost) # posterior in x
    ax1.plot(xPlot, 1/(x.max()-x.min()*np.ones(xPlot.shape)), c='tab:blue', linestyle='-', linewidth=4, label=r'$\pi^{init}(\phi)$')
    ax1.plot(xPlot, qxTest(xPlot), c='tab:red',  linestyle='--', linewidth=4, label=r'$\pi^{up}(\phi)$')
    ax1.plot(xPlot, qxTrue(xPlot), c='tab:green',  linestyle=':', linewidth=4, label=r'$\pi^{true}(\phi)$')
    ax1.axvline(x=xPlot[np.argmax(qxTest(xPlot))], color='tab:red', linestyle='-', linewidth=4, label=r'MAP: $\pi^{up}(\phi)$', alpha=0.25)
    ax1.axvline(x=xPlot[np.argmax(qxTrue(xPlot))], color='tab:green', linestyle='-', linewidth=4, label=r'MAP: $\pi^{true}(\phi)$', alpha=0.25)
    ax1.legend(loc='best',fontsize=24, frameon=False)
    ax1.set_xlabel(r'$\phi$', fontsize=24)
    ax1.set_ylabel(r'pdf($\phi$)', fontsize=24)
    ax1.set_ylim(bottom=1e-4)
    ax1.set_xlim(left=x.min(), right=x.max())

    qplot = np.linspace(yPrior.min(), yPrior.max(), num=1000)
    ax2.plot(qplot, qPrior(qplot), c='tab:blue', linestyle='-', linewidth=4, label=r'$Q(\pi^{init}(\phi))$')
    ax2.plot(qplot, qObs(qplot) ,  c='tab:red', linestyle='--', linewidth=4, label=r'$\pi^{obs}$', alpha=1)
    ax2.plot(qplot, qPost(qplot),  c='tab:green',  linestyle=':', linewidth=4, label=r'$Q(\pi^{up}(\phi))$', alpha=1)
    ax2.legend(loc='best',fontsize=24, frameon=False)
    ax2.set_xlabel(r'$\sigma$', fontsize=24)
    ax2.set_ylabel(r'pdf($\sigma$)', fontsize=24)
    ax2.set_ylim(bottom=1e-4)
    ax2.set_xlim(left=yPrior.min(), right=yPrior.max())
    plt.savefig('CBayesResults-%s.png' % str(testIdx), dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
