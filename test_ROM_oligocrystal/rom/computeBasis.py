
import numpy as np
import os
import time
import scipy
import numpy.linalg as nla
import scipy.linalg as sla # Do not use due to memory inefficiency
import sklearn.utils.extmath as skmath
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 24

fois = ['MisesCauchy', 'MisesLnV'] # fields of interest
labels = [r'$\sigma_{vM}$', r'$\varepsilon_{vM}$']
# fois = ['MisesCauchy'] # fields of interest
# fois = ['MisesLnV'] # fields of interest

t_start = time.time()

for foi in fois:
    d = np.load('d_%s.npy' % foi)
    print(f'Loading time: {time.time() - t_start:<.2f} seconds.')
    # Count/extract non-zero columns
    tmpTime = time.time() # tic
    normCols = np.linalg.norm(d, axis=0)
    nzElems = np.count_nonzero(normCols)
    print(f'Non-zero elements = {int(nzElems):<d} elements.')
    d = d[:,:nzElems]
    # Compute mean column
    meanCol = np.mean(d, axis=1)
    # Subtract mean column
    d = d - np.atleast_2d(meanCol).T
    print(f'Centering time: {time.time() - tmpTime:<.2f} seconds.')
    # Perform thin SVD
    tmpTime = time.time()
    # u, s, vT = sla.svd(d, full_matrices=False) # full_matrices: full or thin SVD
    u, s, vT = nla.svd(d, full_matrices=False) # full_matrices: full or thin SVD
    # u, s, vT = skmath.randomized_svd(d, n_components=3000)
    print(f'SVD time: {time.time() - tmpTime:<.2f} seconds.')
    # Save POD basis and eigendecay
    tmpTime = time.time()
    np.save('podBasis_%s' % foi, u)
    np.save('podEigen_%s' % foi, s)
    print(f'Save time: {time.time() - tmpTime:<.2f} seconds.')
    # Verify that: d = np.dot(u, np.dot(np.diag(s), vT))
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
    ax2.plot(np.cumsum(s) / np.sum(s), c='tab:green', linestyle='None', marker='s', markersize=1, label=r'relative energy')
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
    q = np.cumsum(s) / np.sum(s) # quantile
    # Print quantile with POD components
    for qPrint in [0.90, 0.95, 0.99, 0.995]:
        print(f'{foi:10}: {qPrint*100:<.1f}% quantile = {np.where(q>qPrint)[0][0]:<d} POD components')
    print('\n')


print(f'Total time for POD basis: {time.time() - t_start:<.2f} seconds.')
