
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24


cost_per_level = [39,    365,    1955,    3305,    12487]
cost_per_level = np.array(cost_per_level)
levels = np.arange(len(cost_per_level))
num_levels = len(levels)

plt.figure()

plt.plot(levels, np.log2(cost_per_level), color='g', marker='o', markersize=8, linestyle='-', linewidth=2, label=r'$Q_{\ell}$')

slope, intercept, r_value, p_value, std_err = stats.linregress(levels, np.log2(cost_per_level))

x = np.linspace(levels[0], levels[-1], num=100)
plt.plot(x, intercept + x * slope, linestyle='--', linewidth=1, label=r'$\gamma \approx %.2f$' % slope)

plt.legend(fontsize=24, loc='best', frameon=False)
plt.xlabel(r'level $\ell$', fontsize=24)
plt.ylabel(r'$\log_2($Cost$(\cdot))$ [seconds]', fontsize=24)

plt.xlim(left=levels[0], right=levels[-1])
plt.ylim(bottom=np.min(np.log2(cost_per_level)), top=np.max(np.log2(cost_per_level)))

plt.show()

