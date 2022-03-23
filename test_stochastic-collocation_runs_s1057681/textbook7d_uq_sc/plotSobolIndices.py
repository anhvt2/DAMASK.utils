
import numpy as np
import glob, os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24 
import matplotlib.ticker as ticker


### copy and paste from {strain,stress}Yield.dat -- data order = {main, total}
sobol_index_strain = np.array([
                      [8.6929387781e-03,  1.3230462273e-02],
                      [1.2927981486e-03,  6.4562575716e-03],
                      [2.5571044792e-01,  8.3885488972e-01],
                      [9.8822611138e-04, -9.4267196469e-03],
                      [1.5399324167e-01,  5.6165622401e-01],
                      [3.7562102050e-02,  1.4234505835e-02],
                      [2.5789017197e-03,  7.4657596633e-02],
])

sobol_index_stress = np.array([
                      [7.2034766913e-01,  7.3307612422e-01],
                      [1.6368460226e-02,  1.2432259748e-02],
                      [6.5181430655e-03,  4.7096451517e-02],
                      [1.5662686298e-01,  1.6579109453e-01],
                      [7.2315423049e-02,  9.4751622945e-02],
                      [1.2812236475e-03,  8.7085371605e-03],
                      [4.3735667033e-05,  2.8279114468e-03],
])

### plot
d = sobol_index_stress.shape[0] # dimensionality
x_axis = np.arange(d)
x_label = [r'$\nu_0$', r'$\rho_0^\alpha$', r'$\tau_{Peierls}$', r'$p$', r'$q$', r'$\Delta H_0$', r'$C_\lambda$']

plt.figure()
plt.bar(x_axis - 0.2, sobol_index_strain[:,0], width=0.4, label='main effects')
plt.bar(x_axis + 0.2, sobol_index_strain[:,1], width=0.4, label='total effects')
plt.xticks(x_axis, x_label)
plt.legend(loc='best', fontsize=24)
plt.ylim(bottom=0) # plt.yscale('log')
plt.title(r'Sobol indices for $\varepsilon_Y$ bcc W', fontsize=24)

plt.figure()
plt.bar(x_axis - 0.2, sobol_index_stress[:,0], width=0.4, label='main effects')
plt.bar(x_axis + 0.2, sobol_index_stress[:,1], width=0.4, label='total effects')
plt.xticks(x_axis, x_label)
plt.legend(loc='best', fontsize=24)
plt.ylim(bottom=0) # plt.yscale('log')
plt.title(r'Sobol indices for $\sigma_Y$ bcc W', fontsize=24)

plt.show()


