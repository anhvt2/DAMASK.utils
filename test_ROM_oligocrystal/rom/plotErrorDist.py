import pyvista
import matplotlib.pyplot as plt
import glob, os
import numpy as np
import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

controlInfo = np.loadtxt('control.log', skiprows=1, delimiter=',')
dotVarEps = controlInfo[:,1]
loadingTime = controlInfo[:,2] # dependent - not an input
initialT = controlInfo[:,3]

TrainIdx   = np.loadtxt('TrainIdx.dat', dtype=int)
TestIdx    = np.loadtxt('TestIdx.dat', dtype=int)
TestIdxOOD = np.loadtxt('TestIdxOOD.dat', dtype=int)
TestIdxID  = np.loadtxt('TestIdxID.dat', dtype=int)
fois = ['MisesCauchy', 'MisesLnV'] # fields of interest
labels = [r'$\sigma_{vM}$', r'$\varepsilon_{vM}$']

for i in TestIdx:
    


fig = plt.figure(num=None, figsize=(14, 12), dpi=300, facecolor='w', edgecolor='k')
plt.scatter(dotVarEps_Train, initialT_Train, marker='o', s=30, c='tab:blue', label='train')
plt.scatter(dotVarEps_Test, initialT_Test, marker='o', s=30, c='tab:orange', label='test')
plt.legend(fontsize=24, loc='upper left', bbox_to_anchor=(1.05, 1.0),frameon=True, markerscale=3)

plt.title('Input Distribution', fontsize=24)
plt.xlabel(r'$\dot{\varepsilon}$ [s$^{-1}$]', fontsize=24)
plt.ylabel(r'$T$ [K]', fontsize=24)
plt.xscale('log',base=10) 
# plt.show()
plt.savefig('TrainTestDistribution', dpi=300, facecolor='w', edgecolor='w',
    orientation='portrait', format=None,
    transparent=False, bbox_inches='tight', pad_inches=0.1,
    metadata=None)

# Plot: train/test-OOD/test-ID
fig = plt.figure(num=None, figsize=(14, 12), dpi=300, facecolor='w', edgecolor='k')
plt.scatter(dotVarEps_Train, initialT_Train, marker='o', s=30, c='tab:blue', label='train')
plt.scatter(dotVarEps_Test_OOD, initialT_Test_OOD, marker='o', s=30, c='tab:orange', label='test (OOD)')
plt.scatter(dotVarEps_Test_ID, initialT_Test_ID, marker='o', s=30, c='tab:green', label='test (ID)')
plt.legend(fontsize=24, loc='upper left', bbox_to_anchor=(1.05, 1.0),frameon=False, markerscale=3)

plt.title('Input Distribution', fontsize=24)
plt.xlabel(r'$\dot{\varepsilon}$ [s$^{-1}$]', fontsize=24)
plt.ylabel(r'$T$ [K]', fontsize=24)
plt.xscale('log',base=10) 
# plt.show()
plt.savefig('TrainTestDistribution-OOD-ID', dpi=300, facecolor='w', edgecolor='w',
    orientation='portrait', format=None,
    transparent=False, bbox_inches='tight', pad_inches=0.1,
    metadata=None)




