
import numpy as np
import os
import time
import matplotlib as mpl
import matplotlib.pyplot as plt

fois = ['MisesCauchy', 'MisesLnV'] # fields of interest
# fois = ['MisesCauchy'] # fields of interest
# fois = ['MisesLnV'] # fields of interest

for foi in fois:
    s = np.load('podEigen_%s.npy' % foi)
    fig, ax1 = plt.subplots(num=None, figsize=(16, 9), dpi=300, facecolor='w', edgecolor='k')
    ax1.plot(s)
    ax1.plot(s, c='b', marker='o', markersize=1)
    # ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('POD components', fontsize=12)
    ax1.set_ylabel(r'$\sigma_i$', fontsize=12)
    ax1.set_title('eigenspectrum', fontsize=12)
    ax1.set_xlim(left=0)
    # ax1.grid(axis='both',visible=True, which='both')
    # plt.xaxis.set_minor_locator(MultipleLocator(200))

    ax2 = ax1.twinx()
    ax2.plot(np.cumsum(s) / np.sum(s), c='tab:green', marker='s', markersize=1)
    ax2.set_ylabel(r'relative "energy" ($\ell_2$)', fontsize=12)
    ax2.set_ylim(top=1.0)
    # ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(200))
    # ax2.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(100))
    # ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(200))
    # ax2.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(100))
    # ax2.grid(axis='both')
    # plt.show()
    plt.savefig('eigendecay_%s.png' % foi, dpi=300, facecolor='w', edgecolor='w', 
        orientation='landscape', format=None, transparent=False, 
        bbox_inches='tight', pad_inches=0.1, metadata=None)
    # Diagnostics
    q = np.cumsum(s) / np.sum(s) # quantile
    # Print quantile with POD components
    for qPrint in [0.90, 0.95, 0.99, 0.995]:
        print(f'{foi:10}: {qPrint*100:<.1f}% quantile = {np.where(q>qPrint)[0][0]:<d} POD components')
    print('\n')
