
import numpy as np
import glob, os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24 
import matplotlib.ticker as ticker


### copy and paste from {strain,stress}Yield.dat -- data order = {main, total}
sobol_index_strain = np.array([
                      [3.1229937686e-03, -3.4436490846e-02],
                      [7.3351945061e-03,  1.5556471715e-01],
                      [3.9033560017e-02,  5.7121263296e-02],
                      [6.8694892924e-01,  7.3564408386e-01],
                      [2.2158904311e-01,  1.9416159944e-01],
                      [9.8917381492e-02,  2.1020595874e-01],
                      [6.3483822214e-02,  2.0892355152e-01],
])

sobol_index_stress = np.array([
                      [2.3663272867e-03, -3.6630176314e-02],
                      [7.0459617523e-03,  1.5597424952e-01],
                      [4.3267009077e-02,  6.1720221714e-02],
                      [6.9412759951e-01,  7.4201738631e-01],
                      [2.1086177189e-01,  1.7945281927e-01],
                      [1.0314595995e-01,  2.0948829049e-01],
                      [6.5679058180e-02,  2.0787062165e-01],
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
for i in range(d):
    plt.text(x_axis[i] - 0.2, sobol_index_strain[i,0], '%.4e' % sobol_index_strain[i,0])
    plt.text(x_axis[i] + 0.2, sobol_index_strain[i,1], '%.4e' % sobol_index_strain[i,1])
plt.title(r'Sobol indices for $\varepsilon_Y$ bcc W', fontsize=24)

plt.figure()
plt.bar(x_axis - 0.2, sobol_index_stress[:,0], width=0.4, label='main effects')
plt.bar(x_axis + 0.2, sobol_index_stress[:,1], width=0.4, label='total effects')
plt.xticks(x_axis, x_label)
plt.legend(loc='best', fontsize=24)
plt.ylim(bottom=0) # plt.yscale('log')
for i in range(d):
    plt.text(x_axis[i] - 0.2, sobol_index_stress[i,0], '%.4e' % sobol_index_stress[i,0])
    plt.text(x_axis[i] + 0.2, sobol_index_stress[i,1], '%.4e' % sobol_index_stress[i,1])
plt.title(r'Sobol indices for $\sigma_Y$ bcc W', fontsize=24)

plt.show()


