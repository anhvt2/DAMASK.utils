
import numpy as np
import glob, os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 24 
import matplotlib.ticker as ticker


### copy and paste from {strain,stress}Yield.dat -- data order = {main, total}
sobol_index_strain = np.array([
                      [2.1441798857e-01,  5.6678110734e-01],
                      [1.1491843990e-02, -2.5238978045e-02],
                      [1.4445302065e-02,  6.1311539811e-02],
                      [1.2487369867e-05, -9.5843831639e-05],
                      [1.2897368491e-06, -7.2289372194e-05],
                      [1.4008428734e-01,  4.7724489427e-01],
                      [1.7735128735e-02,  7.2490781683e-02],
                      [2.2981314720e-07,  6.0422880036e-06],
                      [3.7888470358e-03,  1.0907226445e-02],
                      [3.7143430369e-06, -1.2016828276e-04],
                      [8.8368858386e-12,  3.8083358317e-07],
                      [3.3617555587e-02,  1.0207720586e-01],
                      [1.1326304687e-06, -6.7280560063e-05],
                      [7.4071978644e-03,  2.0823756258e-02],
                      [6.5083519827e-02,  2.4387717611e-01],
                      [7.2519560197e-07, -9.8330225941e-06],
])

sobol_index_stress = np.array([
                      [ 8.7908005543e-02,  3.5660973009e-01],
                      [ 3.6325398693e-02,  3.8610521569e-02],
                      [ 2.6880281555e-02,  1.1810420209e-01],
                      [ 6.5802734651e-04,  9.4973692053e-04],
                      [ 7.3641602114e-06,  2.2085181930e-05],
                      [ 1.0438058321e-01,  3.7290520516e-01],
                      [ 2.9796924080e-02,  1.0638227185e-01],
                      [ 1.4119626142e-06,  2.0709100393e-06],
                      [ 7.9862123552e-06,  3.6115357230e-05],
                      [ 2.1223048246e-05,  4.1625328880e-05],
                      [-9.5363964508e-13, -2.1704291561e-10],
                      [ 2.9276291384e-02,  1.0614125791e-01],
                      [ 6.4723270022e-06,  1.9245702078e-05],
                      [ 6.0078024711e-06,  1.8188958100e-05],
                      [ 1.2353993042e-01,  3.6844724161e-01],
                      [ 5.1925194126e-06,  5.1675641970e-06],
])

### plot
d = sobol_index_stress.shape[0] # dimensionality
x_axis = np.arange(d)
x_label = [r'$\tau_{0,basal}$', r'$\tau_{0,pris}$', r'$\tau_{0,pyr \langle a \rangle}$', r'$\tau_{0,pyr \langle c+a \rangle}$', r'$\tau_{0,T1}$', r'$\tau_{0,C2}$', r'$\tau_{\infty,basal}$', r'$\tau_{\infty,pris}$', r'$\tau_{\infty,pyr \langle a \rangle}$', r'$\tau_{\infty,pyr \langle c+a \rangle}$', r'$h_{0}^{tw-tw}$', r'$h_{0}^{s-s}$', r'$h_{0}^{tw-s}$', r'$n_s$', r'$n_{tw}$', r'$a$']

plt.figure()
plt.bar(x_axis - 0.2, sobol_index_strain[:,0], width=0.4, label='main effects')
plt.bar(x_axis + 0.2, sobol_index_strain[:,1], width=0.4, label='total effects')
plt.xticks(x_axis, x_label)
plt.legend(loc='best', fontsize=24)
plt.ylim(bottom=0) # plt.yscale('log')
for i in range(d):
    plt.text(x_axis[i] - 0.2, sobol_index_strain[i,0], '%.4e' % sobol_index_strain[i,0])
    plt.text(x_axis[i] + 0.2, sobol_index_strain[i,1], '%.4e' % sobol_index_strain[i,1])
plt.title(r'Sobol indices for $\varepsilon_Y$ hcp Mg', fontsize=24)

plt.figure()
plt.bar(x_axis - 0.2, sobol_index_stress[:,0], width=0.4, label='main effects')
plt.bar(x_axis + 0.2, sobol_index_stress[:,1], width=0.4, label='total effects')
plt.xticks(x_axis, x_label)
plt.legend(loc='best', fontsize=24)
plt.ylim(bottom=0) # plt.yscale('log')
for i in range(d):
    plt.text(x_axis[i] - 0.2, sobol_index_stress[i,0], '%.4e' % sobol_index_stress[i,0])
    plt.text(x_axis[i] + 0.2, sobol_index_stress[i,1], '%.4e' % sobol_index_stress[i,1])
plt.title(r'Sobol indices for $\sigma_Y$ hcp Mg', fontsize=24)

plt.show()


