# -*- coding: utf-8 -*-

import glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar as clb
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.animation as animation
import time
from mpl_toolkits.mplot3d import Axes3D
import os

lab_s = ['$\sigma_{P}$, MPa']
p_alpha = 1

fig = plt.figure(figsize=(16, 12))

exp_1 = []
exp_1 = np.loadtxt('850.dat')
exp_2 = []
exp_2 = np.loadtxt('900.dat')
exp_3 = []
exp_3 = np.loadtxt('1000.dat')
exp_4 = []
exp_4 = np.loadtxt('1150.dat')

extra = []
extra = np.loadtxt('pred_700.dat')


marksize = 150
ax = fig.add_subplot(1,1,1)

comp = []

xmlfiles = []
os.chdir(".")

for file in glob.glob("cal*.dat"):
	comp = np.loadtxt(file)

	linecol = 'b'
	plt.plot(comp[:,0], comp[:,1], c=linecol, linewidth=4.0)

	'''
	plt.scatter(exp[:,0],exp[:,1],c='b',marker='o',s=marksize, zorder=5)
	plt.scatter(exp2[:,0],exp2[:,1],c='b',marker='p',s=marksize, zorder=5)
	plt.scatter(exp3[:,0],exp3[:,1],c='b',marker='*',s=marksize, zorder=5)

	plt.scatter(exp_2[:,0],exp_2[:,1],c='r',marker='o',s=marksize, zorder=5)
	plt.scatter(exp_2_2[:,0],exp_2_2[:,1],c='r',marker='p',s=marksize, zorder=5)
	plt.scatter(exp_2_3[:,0],exp_2_3[:,1],c='r',marker='*',s=marksize, zorder=5)
	'''

plt.scatter(exp_1[:,0]*100,exp_1[:,1],c='r',marker='o',s=marksize, zorder=2,label='1123K')
plt.scatter(exp_2[:,0]*100,exp_2[:,1],c='r',marker='^',s=marksize, zorder=2,label='1173K')
plt.scatter(exp_3[:,0]*100,exp_3[:,1],c='r',marker='*',s=marksize, zorder=2,label='1273K')
plt.scatter(exp_4[:,0]*100,exp_4[:,1],c='r',marker='s',s=marksize, zorder=2,label='1423K')

#plt.plot(extra[:,0], extra[:,1], 'b--', linewidth=4.0)


ax.set_xlabel(r'$\varepsilon$, %', fontsize=40, labelpad=15)
ax.set_ylabel(r'$\sigma$, MPa', fontsize=40, labelpad=15)
ax.tick_params(axis='x', labelsize=30, pad = 10)
ax.tick_params(axis='y', labelsize=30, pad = 10)

#	ax.grid(True)
fig.tight_layout()

ax.set_xlim(0, 10)
ax.set_ylim(0, 250)

ax.locator_params(axis = 'x', nbins = 4)
ax.locator_params(axis = 'y', nbins = 4)

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, loc='lower right',numpoints=1, prop={'size':30})


plt.show()

