
import numpy as np
import glob, os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24 
import matplotlib.ticker as ticker


### copy and paste from {strain,stress}Yield.dat -- data order = {main, total}
sobol_index_strain = np.array([
                      [2.3429848766e-01,  7.8575444762e-01],
                      [2.4413879872e-05, -1.9825853515e-03],
                      [1.5014926427e-01,  7.0351600222e-01],
                      [1.2646414824e-02,  1.9222524769e-01],
                      [3.7368969257e-04, -1.0383381805e-02],
])

sobol_index_stress = np.array([
                      [5.5548733103e-01,  8.2583609373e-01],
                      [5.3202395126e-04, -9.9903474307e-04],
                      [1.4338468304e-01,  4.1594228554e-01],
                      [1.1045144968e-02,  7.6588363013e-02],
                      [1.8287396914e-03, -1.0912962742e-02],
])

### plot
d = sobol_index_stress.shape[0] # dimensionality
x_axis = np.arange(d)
x_label = [r'$\tau_0$', r'$\tau_{\infty}$', r'$h_0$', r'$n$', r'$a$']

plt.figure()
plt.bar(x_axis - 0.2, sobol_index_strain[:,0], width=0.4, label='main effects')
plt.bar(x_axis + 0.2, sobol_index_strain[:,1], width=0.4, label='total effects')
plt.xticks(x_axis, x_label)
plt.legend(loc='best', fontsize=24)
plt.ylim(bottom=0) # plt.yscale('log')
plt.title(r'Sobol indices for $\varepsilon_Y$ fcc Cu', fontsize=24)

plt.figure()
plt.bar(x_axis - 0.2, sobol_index_stress[:,0], width=0.4, label='main effects')
plt.bar(x_axis + 0.2, sobol_index_stress[:,1], width=0.4, label='total effects')
plt.xticks(x_axis, x_label)
plt.legend(loc='best', fontsize=24)
plt.ylim(bottom=0) # plt.yscale('log')
plt.title(r'Sobol indices for $\sigma_Y$ fcc Cu', fontsize=24)

plt.show()


