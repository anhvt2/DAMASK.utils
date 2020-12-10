

# filterTable < single_phase_equiaxed_tension.txt --white inc,1_f,1_p > stress_strain.log
import numpy as np
import matplotlib.pyplot as plt

d = np.loadtxt('stress_strain.log', skiprows=7)

plt.plot(d[:,1] - 1, d[:,2], 'bo-', markersize=2)
plt.xlabel(r'$\varepsilon$', fontsize=30)
plt.ylabel(r'$\sigma$', fontsize=30)
plt.show()

