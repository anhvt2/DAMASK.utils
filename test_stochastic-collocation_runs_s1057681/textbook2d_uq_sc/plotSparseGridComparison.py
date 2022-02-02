
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


l1 = np.loadtxt('dakota_sparse_tabular-2d-level1.dat', skiprows=1)[:,2:]
l2 = np.loadtxt('dakota_sparse_tabular-2d-level2.dat', skiprows=1)[:,2:]
l3 = np.loadtxt('dakota_sparse_tabular-2d-level3.dat', skiprows=1)[:,2:]
l4 = np.loadtxt('dakota_sparse_tabular-2d-level4.dat', skiprows=1)[:,2:]
l5 = np.loadtxt('dakota_sparse_tabular-2d-level5.dat', skiprows=1)[:,2:]


# 
plt.figure()
grid1Plt, = plt.plot(l1[:,0], l1[:,1], marker='X', c='tab:brown' , linestyle='None', linewidth=2, markersize=10, label=r'$\ell=1$')
grid2Plt, = plt.plot(l2[:,0], l2[:,1], marker='v', c='tab:green' , linestyle='None', linewidth=2, markersize=8, label=r'$\ell=2$')
grid3Plt, = plt.plot(l3[:,0], l3[:,1], marker='^', c='tab:olive' , linestyle='None', linewidth=2, markersize=6, label=r'$\ell=3$')
grid4Plt, = plt.plot(l4[:,0], l4[:,1], marker='s', c='tab:red'	 , linestyle='None', linewidth=2, markersize=4, label=r'$\ell=4$')
grid5Plt, = plt.plot(l5[:,0], l5[:,1], marker='o', c='tab:blue'  , linestyle='None', linewidth=2, markersize=2, label=r'$\ell=5$')



# ax = plt.axes()
# ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
# ax.yaxis.get_major_formatter().set_powerlimits((0, 3))
# ax.set_yscale('log')

plt.tick_params(axis='both', which='major', labelsize=24)
plt.tick_params(axis='both', which='minor', labelsize=24)
# plt.legend(handles=[grid1Plt, grid2Plt, grid3Plt, grid4Plt, grid5Plt], fontsize=24, markerscale=2, bbox_to_anchor=(1.05,1), loc='best') # bbox
# plt.legend(handles=[grid1Plt, grid2Plt, grid3Plt, grid4Plt, grid5Plt], fontsize=12, markerscale=2, bbox_to_anchor=(1,1.05), ncol = 5)
plt.legend(handles=[grid1Plt, grid2Plt, grid3Plt, grid4Plt, grid5Plt], fontsize=18, markerscale=2, bbox_to_anchor=(0,-0.20,1.,.102), loc='lower left', ncol = 5, mode='expand', frameon=False)
plt.xlabel(r'$x_1$', fontsize=24)
plt.ylabel(r'$x_2$', fontsize=24)
plt.locator_params(axis = 'x', nbins = 8)
plt.locator_params(axis = 'y', nbins = 8)
plt.title(r'Sparse grids for $d=2$ at different level $\ell$', fontsize=24)
plt.xlim(-1.0,1.0)
plt.ylim(-1.0,1.0)
# plt.grid()
plt.axis('equal')
plt.show()
