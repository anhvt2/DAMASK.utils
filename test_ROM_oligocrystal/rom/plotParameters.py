import pyvista
import matplotlib.pyplot as plt
import glob, os
import numpy as np
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

d = np.loadtxt('control.log', skiprows=1, delimiter=',')
dotVarEps = d[:,1]
loadingTime = d[:,2] # dependent - not an input
initialT = d[:,3]

plt.figure(figsize=(12,12))
plt.scatter(dotVarEps, initialT, marker='o', s=8, c='k')
plt.title('Input Distribution', fontsize=24)
plt.xlabel(r'$\dot{\varepsilon}$ [s$^{-1}$]', fontsize=24)
plt.ylabel(r'$T$ [K]', fontsize=24)
plt.xscale('log',base=10) 
plt.savefig('InputDistribution', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        metadata=None)

# See more in sampleTrainTest.py
