
import numpy as np
import os
import time
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 24

fois = ['MisesCauchy', 'MisesLnV'] # fields of interest
labels = [r'$\sigma_{vM}$', r'$\varepsilon_{vM}$']

for foi, label in zip(fois, labels):
    s = np.load('podEigen_%s.npy' % foi)
    fig, ax1 = plt.subplots(num=None, figsize=(16, 9), dpi=300, facecolor='w', edgecolor='k')
    ax1.plot(s)
    ax1.plot(s, c='b', marker='o', linestyle='None', markersize=1, label=r'eigenvalue')
    # ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('POD components', fontsize=24)
    ax1.set_ylabel(r'Eigenvalue', fontsize=24)
    ax1.set_title(r'Eigenspectrum: %s' % label, fontsize=24)
    ax1.set_xlim(left=0,right=int(len(s)/1.2))
    ax1.grid(None)

    ax2 = ax1.twinx()
    ax2.plot(np.cumsum(s**2) / np.sum(s**2), c='tab:green', linestyle='None', marker='s', markersize=1, label=r'relative energy')
    ax2.set_ylim(top=1.0)
    ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(500))
    ax2.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(250))
    # ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
    ax2.set_ylabel(r'relative "energy" ($\ell_2$)', fontsize=24)
    ax2.grid(visible=True, which='both', axis='x', linestyle='-', linewidth='2')
    # plt.show()
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=24, markerscale=10, loc='best')
    plt.savefig('eigendecay_%s.png' % foi, dpi=300, facecolor='w', edgecolor='w', 
        orientation='landscape', format=None, transparent=False, 
        bbox_inches='tight', pad_inches=0.1, metadata=None)
    # Diagnostics
    q = np.cumsum(s**2) / np.sum(s**2) # quantile
    # Print quantile with POD components
    for qPrint in [0.90, 0.95, 0.99, 0.995]:
        print(f'{foi:10}: {qPrint*100:<.1f}% quantile = {np.where(q>qPrint)[0][0]:<d} POD components')
    print('\n')
