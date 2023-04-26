
import numpy as np
import matplotlib as mpl
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import StrMethodFormatter
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

mc_cost = np.loadtxt('vanilla_mc_cost.dat', delimiter=',')
varepsilon, num_samples, computational_cost, final_vareps = mc_cost[:,0], mc_cost[:,1], mc_cost[:,2], mc_cost[:,3]

plt.plot(final_vareps, computational_cost, color='tab:red', linestyle='-', marker='o', 
	markersize=15, markerfacecoloralt='white', markeredgecolor='k', 
	fillstyle='left', label='MC')

plt.legend(fontsize=24, frameon=False, loc='best')
plt.xlabel(r'tolerance $\varepsilon$', fontsize=24)
plt.ylabel(r'time [$s$]', fontsize=24)
plt.xscale('log',base=10)
plt.yscale('log',base=10)
plt.show()
