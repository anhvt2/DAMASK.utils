

"""
This script dumps an image for visualizing 2D slice from a phase
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

p = np.load('phase_dump_12_out.npy')
plt.imshow(np.swapaxes(p[:,0,:], 0, 1), cmap='plasma', alpha=0.25)
plt.savefig('Discretized50umDogbone.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
