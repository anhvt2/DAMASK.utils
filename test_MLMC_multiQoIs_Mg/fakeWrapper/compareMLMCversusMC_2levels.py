
import numpy as np
import matplotlib as mpl
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib.ticker import StrMethodFormatter
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

# https://stackoverflow.com/questions/32536226/log-log-plot-linear-regression
def myComplexFunc(x, a, b, c):
    return a * np.power(x, b) + c

mc_cost = np.loadtxt('vanilla_mc_cost.dat', delimiter=',')
mc_varepsilon, mc_num_samples, mc_computational_cost, mc_rmse = mc_cost[:,0], mc_cost[:,1], mc_cost[:,2], mc_cost[:,3]
popt, pcov = curve_fit(myComplexFunc, mc_rmse, mc_computational_cost)
mc_x = np.linspace(mc_rmse.min(), mc_rmse.max(), num=10)
# plt.plot(mc_x, myComplexFunc(mc_x, *popt), 'r-', label="({0:.3f}*x**{1:.3f}) + {2:.3f}".format(*popt)) # debug

mlmc_cost = np.loadtxt('mlmc_cost.dat', skiprows=1, delimiter=',')
mlmc_cost = mlmc_cost[mlmc_cost[:, -1].argsort()] # sort by rmse
mlmc_varepsilon, mlmc_n, mlmc_computational_cost, mlmc_rmse = mlmc_cost[:,0], mlmc_cost[:,1:3], mlmc_cost[:,3], mlmc_cost[:,4]

if mc_computational_cost.shape[0] == mlmc_computational_cost.shape[0]:
	print(f"Average computational speedup: {np.mean(mc_computational_cost / mlmc_computational_cost)}")

plt.plot(mc_rmse, mc_computational_cost, color='tab:red', linestyle='-', marker='s', 
	markersize=15, markerfacecoloralt='white', markeredgecolor='k', 
	fillstyle='top', label=r"Monte Carlo")
	# fillstyle='top', label=r"Monte Carlo: $%.2f \varepsilon^{%.2f} + %.2f$" % (popt[0], popt[1], popt[2]))

print(mlmc_rmse, mlmc_computational_cost)
plt.plot(mlmc_rmse, mlmc_computational_cost, color='tab:blue', linestyle='-', marker='o', 
	markersize=15, markerfacecoloralt='white', markeredgecolor='k', 
	fillstyle='left', label='Multi-level Monte Carlo')

plt.legend(fontsize=24, frameon=False, loc='best')
plt.xlabel(r'tolerance $\varepsilon$', fontsize=24)
plt.ylabel(r'time [$s$]', fontsize=24)
plt.xscale('log',base=10)
plt.yscale('log',base=10)
# plt.xlim(left=1e-2,right=1)
plt.show()
