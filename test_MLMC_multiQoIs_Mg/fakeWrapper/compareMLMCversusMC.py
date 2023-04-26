
import numpy as np
import matplotlib as mpl
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import StrMethodFormatter
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

mc_cost = np.loadtxt('vanilla_mc_cost.dat', delimiter=',')

plt.plot(mc_cost[:,0], mc_cost[:,2], color='tab:red', linestyle='-', marker='o', 
	markersize=15, markerfacecoloralt='white', markeredgecolor='k', 
	fillstyle='left', label='MC')

plt.legend(fontsize=24, frameon=False, loc='best')
plt.xlabel(r'tolerance $\varepsilon$', fontsize=24)
plt.ylabel(r'time [$s$]', fontsize=24)
plt.xscale('log',base=10)
plt.yscale('log',base=10)
plt.show()
