
import numpy as np
import matplotlib as mpl
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import StrMethodFormatter
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

mc_cost = np.loadtxt('vanilla_mc_cost.dat', delimiter=',')
mc_varepsilon, mc_num_samples, mc_computational_cost, mc_rmse = mc_cost[:,0], mc_cost[:,1], mc_cost[:,2], mc_cost[:,3]

mlmc_cost = np.loadtxt('mlmc_cost.dat', skiprows=1, delimiter=',')
mlmc_cost = mlmc_cost[mlmc_cost[:, -1].argsort()] # sort by rmse
mlmc_varepsilon, mlmc_n, mlmc_computational_cost, mlmc_rmse = mlmc_cost[:,0], mlmc_cost[:,1:3], mlmc_cost[:,3], mlmc_cost[:,4]

if mc_computational_cost.shape[0] == mlmc_computational_cost.shape[0]:
	print(f"Average computational speedup: {np.mean(mc_computational_cost / mlmc_computational_cost)}")

plt.plot(mc_rmse, mc_computational_cost, color='tab:red', linestyle='-', marker='o', 
	markersize=15, markerfacecoloralt='white', markeredgecolor='k', 
	fillstyle='left', label='MC')

print(mlmc_rmse, mlmc_computational_cost)
plt.plot(mlmc_rmse, mlmc_computational_cost, color='tab:blue', linestyle='-', marker='s', 
	markersize=15, markerfacecoloralt='white', markeredgecolor='k', 
	fillstyle='top', label='MLMC')

plt.legend(fontsize=24, frameon=False, loc='best')
plt.xlabel(r'tolerance $\varepsilon$', fontsize=24)
plt.ylabel(r'time [$s$]', fontsize=24)
plt.xscale('log',base=10)
plt.yscale('log',base=10)
plt.show()
