
# postResults single_phase_equiaxed_tension.spectralOut --cr f,p
# filterTable < single_phase_equiaxed_tension.txt --white inc,1_f,1_p > stress_strain.log
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

d = np.loadtxt('stress_strain.log', skiprows=7)

plt.plot(d[:,1] - 1, d[:,2], 'bo-', markersize=2)
plt.xlabel(r'$\varepsilon$', fontsize=30)
plt.ylabel(r'$\sigma$', fontsize=30)
plt.show()

