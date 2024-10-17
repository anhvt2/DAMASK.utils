import pyvista
import matplotlib.pyplot as plt
import glob, os
import numpy as np
import matplotlib as mpl
np.random.seed(8)
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

d = np.loadtxt('control.log', skiprows=1, delimiter=',')
dotVarEps = d[:,1]
loadingTime = d[:,2] # dependent - not an input
initialT = d[:,3]

# Define train/test fraction
trainFraction = 0.6
testFraction  = 1 - trainFraction
numData = d.shape[0]
# Sample train/test datatsets
trainIdx = np.sort(np.random.choice(np.arange(numData), size=int(trainFraction*numData), replace=False))
testIdx  = np.sort(np.setdiff1d(np.arange(numData), trainIdx))
TrainData = d[trainIdx]
TestData  = d[testIdx]

dotVarEps_Train, initialT_Train = TrainData[:,1], TrainData[:,3]
dotVarEps_Test, initialT_Test = TestData[:,1], TestData[:,3]
# Save train/test datasets
np.savetxt('TrainData.dat', TrainData, 
        fmt='%d, %.8e, %.8e, %.1f', 
        header='i, dotVareps, loadingTime, initialT',
        comments='')
np.savetxt('TestData.dat',  TestData,  
        fmt='%d, %.8e, %.8e, %.1f', 
        header='i, dotVareps, loadingTime, initialT',
        comments='')
np.savetxt('TrainIdx.dat', trainIdx, fmt='%d')
np.savetxt('TestIdx.dat', testIdx, fmt='%d')

# Plot
# plt.figure(figsize=(14,12))
fig = plt.figure(num=None, figsize=(14, 12), dpi=300, facecolor='w', edgecolor='k')
# plt.scatter(dotVarEps, initialT, marker='o', s=8, c='k')
plt.scatter(dotVarEps_Train, initialT_Train, marker='o', s=20, c='tab:blue', label='train')
plt.scatter(dotVarEps_Test, initialT_Test, marker='o', s=20, c='tab:red', label='test')

# plt.legend(fontsize=24, markerscale=2, loc='best') # bbox
plt.legend(fontsize=24, loc='upper left', bbox_to_anchor=(1.05, 1.0),frameon=True)

plt.title('Input Distribution', fontsize=24)
plt.xlabel(r'$\dot{\varepsilon}$ [s$^{-1}$]', fontsize=24)
plt.ylabel(r'$T_0$ [K]', fontsize=24)
plt.xscale('log',base=10) 
# plt.show()
plt.savefig('TrainTestDistribution', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        metadata=None)

