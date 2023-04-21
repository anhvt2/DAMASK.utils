
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import StrMethodFormatter
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

# read dat file
data = np.loadtxt("MultilevelEstimators-multiQoIs.dat", delimiter=",")
nb_of_rows = data.shape[0]
nb_of_qoi = data.shape[1] - 1

# get max level
max_level = int(np.amax(data, axis=0)[0]) + 1 #: ignore the last level due to numerical instability
levels = np.arange(max_level)
print(f"max_level: {max_level}")
print(f"number of samples: {nb_of_rows}")

# gather all samples
samples = []
dsamples = []
for level in range(max_level):
	samples.append([])
	dsamples.append([])
	for row in range(nb_of_rows - 1):
		if data[row, 0] == level and data[row + 1, 0] == max(0, level - 1):
			samples[level].append(row)
			if level > 0:
				dsamples[level].append(row + 1)
	print(f"Found {len(samples[level])} samples on level {level}")

# print statistics
for qoi in range(nb_of_qoi):
	print(f"Statistics for qoi {qoi}:")
	print(*[f"{s:<12s}" for s in ("level", "E_l", "dE_l", "V_l", "dV_l")])
	collocated_strain = (qoi + 1) / 10
	
	E_l, dE_l, V_l, dV_l = [], [], [], []
	for level in range(max_level):
		x = data[samples[level], qoi + 1]
		dx = data[dsamples[level], qoi + 1]
		E_l.append(np.mean(x))
		dE_l.append(np.abs(np.mean(x - dx)) if level > 0 else np.nan)
		V_l.append(np.var(x))
		dV_l.append(np.abs(np.var(x - dx)) if level > 0 else np.nan)
		print(f"{level:<13d}", end="")
		print(f"{np.mean(x):<13.5f}", end="")
		print(f"{np.abs(np.mean(x - dx)):<13.5f}" if level > 0 else "-".ljust(13), end="")
		print(f"{np.var(x):<13.5f}", end="")
		print(f"{np.abs(np.var(x - dx)):<13.5f}" if level > 0 else "-".ljust(13))
	
	E_l, dE_l, V_l, dV_l = np.array(E_l), np.array(dE_l), np.array(V_l), np.array(dV_l)
	# plot expectation
	plt.figure(figsize=(12,12))
	plt.plot(levels, np.log2(E_l), color='r', marker='o', markersize=8, linestyle='-', linewidth=2, label=r'$Q_{\ell}$')
	plt.plot(levels, np.log2(dE_l), color='r', marker='o', markersize=8, linestyle='--', linewidth=2, label=r'$\Delta Q_{\ell}$')

	slope, intercept, r_value, p_value, std_err = stats.linregress(levels[1:], np.log2(dE_l[1:]))
	x = np.linspace(levels[1], levels[-1], num=100)
	plt.plot(x, intercept + x * slope, linestyle='--', color='k', linewidth=1, label=r'$\alpha \approx %.2f$' % -slope)
	plt.legend(fontsize=24, loc='best', frameon=False)
	plt.xlabel(r'level $\ell$', fontsize=24)
	plt.ylabel(r'$\log_2(\mathrm{\mathbb{E}}[|\cdot|])$', fontsize=24)

	plt.xlim(left=levels[0], right=levels[-1])
	plt.title(r'$\sigma(\varepsilon = %.1f)$' % collocated_strain, fontsize=24)
	# plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # 0 decimal places
	plt.xticks(levels)
	plt.savefig('E_l-qoi-%d.png' % qoi, dpi=600)
	plt.clf()


	# plot variance
	plt.figure(figsize=(12,12))
	plt.plot(levels, np.log2(V_l), color='b', marker='o', markersize=8, linestyle='-', linewidth=2, label=r'$Q_{\ell}$')
	plt.plot(levels, np.log2(dV_l), color='b', marker='o', markersize=8, linestyle='--', linewidth=2, label=r'$\Delta Q_{\ell}$')

	slope, intercept, r_value, p_value, std_err = stats.linregress(levels[1:], np.log2(dV_l[1:]))
	x = np.linspace(levels[1], levels[-1], num=100)
	plt.plot(x, intercept + x * slope, linestyle='--', color='k', linewidth=1, label=r'$\beta \approx %.2f$' % -slope)
	plt.legend(fontsize=24, loc='best', frameon=False)
	plt.xlabel(r'level $\ell$', fontsize=24)
	plt.ylabel(r'$\log_2(\mathrm{\mathbb{V}}[|\cdot|])$', fontsize=24)

	plt.xlim(left=levels[0], right=levels[-1])
	plt.title(r'$\sigma(\varepsilon = %.1f)$' % collocated_strain, fontsize=24)
	# plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # 0 decimal places
	plt.xticks(levels)
	plt.savefig('V_l-qoi-%d.png' % qoi, dpi=600)
	plt.clf()

# plt.show()

