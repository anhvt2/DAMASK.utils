import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
 
axes = [5, 5, 5]
data = np.ones(axes, dtype=np.bool)
 
colors = np.empty(axes + [4], dtype=np.float32)
 
# Control Transparency
alpha = .7
 
# Control colors
colors[0] = [1, 0, 0, alpha]
colors[1] = [0, 1, 0, alpha]
colors[2] = [0, 0, 1, alpha]
colors[3] = [1, 1, 0, alpha]
colors[4] = [0, 1, 1, alpha]
 
# set all internal colors to
# black with alpha=1
colors[1:-1, 1:-1, 1:-1, 0:3] = 0
colors[1:-1, 1:-1, 1:-1, 3] = 1
 
# Plot figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
 
# Control number of slice
data[-1] = True
data[-2] = False
data[-3] = False
data[-4] = False
data[-5] = True
 
# Voxels is used to customizations of
# the sizes, positions and colors.
ax.voxels(data, facecolors=colors, edgecolors='pink')
 
# it can be used to change the axes view
ax.view_init(100, 0)
plt.show()