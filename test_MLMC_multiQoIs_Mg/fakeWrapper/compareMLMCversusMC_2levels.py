
import numpy as np
import matplotlib as mpl
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib.ticker import StrMethodFormatter
from scipy.spatial.distance import euclidean
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

# https://stackoverflow.com/questions/32536226/log-log-plot-linear-regression
def myComplexFunc(x, a, b, c):
    return a * np.power(x, b) + c

mc_cost = np.loadtxt('vanilla_mc_cost.dat', delimiter=',')
mc_varepsilon, mc_num_samples, mc_computational_cost, mc_rmse = mc_cost[:,0], mc_cost[:,1], mc_cost[:,2] / 60, mc_cost[:,3] # convert to hr
popt, pcov = curve_fit(myComplexFunc, mc_rmse, mc_computational_cost)
mc_x = np.linspace(mc_rmse.min(), mc_rmse.max(), num=10)
# plt.plot(mc_x, myComplexFunc(mc_x, *popt), 'r-', label="({0:.3f}*x**{1:.3f}) + {2:.3f}".format(*popt)) # debug

mlmc_cost = np.loadtxt('mlmc_cost.dat', skiprows=1, delimiter=',')
mlmc_cost = mlmc_cost[mlmc_cost[:, -1].argsort()] # sort by rmse
mlmc_varepsilon, mlmc_n, mlmc_computational_cost, mlmc_rmse = mlmc_cost[:,0], mlmc_cost[:,1:3], mlmc_cost[:,3] / 60, mlmc_cost[:,4] # convert to hr

# ### impose monotonicity
# good_idx = []
# i = 0; tmp = 1e9 # very large initial value
# while i < len(mlmc_computational_cost) - 1:
# 	if tmp > mlmc_computational_cost[i]:
# 		good_idx.append(i)
# 		tmp = mlmc_computational_cost[i]
# 	i += 1

# mlmc_varepsilon = mlmc_varepsilon[good_idx]
# mlmc_n = mlmc_n[good_idx]
# mlmc_computational_cost = mlmc_computational_cost[good_idx]
# mlmc_rmse = mlmc_rmse[good_idx]

### filter too close data points
# print(np.diff(np.log(mlmc_computational_cost)))
# del_idx = []
# for i in range(len(mlmc_computational_cost) - 1):
# 	if np.diff(np.log(mlmc_computational_cost))[i] > -0.08:
# 		del_idx.append(i)
# print(np.diff(np.log(mlmc_computational_cost)))

i = 0
while i < 10:
	del_idx = []
	for i in range(len(mlmc_rmse) - 1):
		if np.diff(np.log(mlmc_rmse))[i] < 0.02: # change this parameter to refine the MLMC plot - default: 0.04
			del_idx.append(i)

	mlmc_varepsilon = np.delete(mlmc_varepsilon, del_idx)
	mlmc_n = np.delete(mlmc_n, del_idx)
	mlmc_computational_cost = np.delete(mlmc_computational_cost, del_idx)
	mlmc_rmse = np.delete(mlmc_rmse, del_idx)
	i += 1

print(np.diff(np.log(mlmc_rmse)))
print(del_idx)

if mc_computational_cost.shape[0] == mlmc_computational_cost.shape[0]:
	print(f"Average computational speedup: {np.mean(mc_computational_cost / mlmc_computational_cost)}")

plt.plot(mc_rmse, mc_computational_cost, color='tab:red', linestyle='-', marker='s', 
	markersize=15, markerfacecoloralt='white', markeredgecolor='k', 
	fillstyle='top', label=r"Monte Carlo")
	# fillstyle='top', label=r"Monte Carlo: $%.2f \varepsilon^{%.2f} + %.2f$" % (popt[0], popt[1], popt[2]))

plt.plot(mlmc_rmse, mlmc_computational_cost, color='tab:blue', linestyle='-', marker='o', 
	markersize=15, markerfacecoloralt='white', markeredgecolor='k', 
	fillstyle='left', label='Multi-level Monte Carlo')

# plt.legend(fontsize=24, frameon=False, bbox_to_anchor=(1, 1))
plt.legend(fontsize=24, frameon=False, loc='best') 
plt.xlabel(r'tolerance $\varepsilon$', fontsize=24)
plt.ylabel(r'time [hour]', fontsize=24)
plt.xscale('log',base=10)
plt.yscale('log',base=10)
# plt.xlim(left=1e-2,right=1)
# plt.grid(which='both', axis='both')
plt.show()
