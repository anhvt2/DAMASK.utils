
import numpy as np
import os
import time
import scipy
import numpy.linalg as nla # 155G/188G -- can run on attaway-login12
import scipy.linalg as sla # Do not use due to memory inefficiency
import sklearn.utils.extmath as skmath
import matplotlib as mpl
import matplotlib.pyplot as plt

fois = ['MisesCauchy', 'MisesLnV'] # fields of interest
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
    ax1.plot(s, c='b', marker='o', markersize=1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('index', fontsize=12)
    ax1.set_ylabel(r'$\sigma_i$', fontsize=12)
    ax1.set_title('eigenspectrum', fontsize=12)

    ax2 = ax1.twinx()
    ax2.plot(np.cumsum(s) / np.sum(s), c='tab:green', marker='s', markersize=1)
    ax2.set_ylabel(r'relative "energy" ($\ell_2$)', fontsize=12)
    ax2.set_ylim(top=1.0)
    # plt.show()
    plt.savefig('eigendecay_%s.png' % foi, dpi=300, facecolor='w', edgecolor='w', 
        orientation='landscape', format=None, transparent=False, 
        bbox_inches='tight', pad_inches=0.1, metadata=None)

print(f'Total time for POD basis: {time.time() - t_start:<.2f} seconds.')

# Loading time: 221.89 seconds.
# Non-zero elements = 5524 elements.
# Centering time: 34.98 seconds.
# SVD time: 633.08 seconds.
# Save time: 215.23 seconds.
# Loading time: 1334.92 seconds.
# Non-zero elements = 5524 elements.
# Centering time: 24.71 seconds.
# SVD time: 594.55 seconds.
# Save time: 220.47 seconds.
# Total time for POD basis: 2177.61 seconds.
