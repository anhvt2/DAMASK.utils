
# ../dakota -i textbook7d_uq_sc_pyImport.in > dakota.log
# grep -inr ' f1' dakota.log  > tmp.txt
# sed -i  's/ f1//g' tmp.txt

# adopt from ./testCBayes-Damask-Phase_Dislotwin_TWIP-Steel-FeMnC-64x64x64/plotQoI.py


import numpy as np
import glob, os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24 
import matplotlib.ticker as ticker
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-q", "--qois", type=str) # options: 'stressYield' or 'strainYield'
args = parser.parse_args()
qois = args.qois

from scipy.stats import norm # The standard Normal distribution
from scipy.stats import gaussian_kde as gaussian_kde # A standard kernel density estimator
d1 = np.loadtxt(qois + '_level1.dat', delimiter=':')[:,1]
d2 = np.loadtxt(qois + '_level2.dat', delimiter=':')[:,1]
d3 = np.loadtxt(qois + '_level3.dat', delimiter=':')[:,1]
if qois == 'stressYield':
	d1 /= 1e6
	d2 /= 1e6
	d3 /= 1e6

q1 = gaussian_kde(d1)
q2 = gaussian_kde(d2)
q3 = gaussian_kde(d3)

x1 = np.linspace(np.min(d1), np.max(d1), 5000)
x2 = np.linspace(np.min(d2), np.max(d2), 5000)
x3 = np.linspace(np.min(d3), np.max(d3), 5000)
# fig = plt.figure(figsize=(8,8))
fig, ax = plt.subplots(1, 1)
sgPlt1, = plt.plot(x1, q1(x1), c='tab:blue',   marker='o', linestyle='-' , markersize=2, label='sparse_grid_level = 1')
sgPlt2, = plt.plot(x2, q1(x2), c='tab:orange', marker='s', linestyle='--', markersize=2, label='sparse_grid_level = 2')
sgPlt3, = plt.plot(x3, q1(x3), c='tab:purple', marker='^', linestyle=':' , markersize=2, label='sparse_grid_level = 3')
plt.legend(handles=[sgPlt1, sgPlt2, sgPlt3], fontsize=24, markerscale=2, loc='best') # bbox
plt.xlim([np.min(np.array([d1, d2, d3])), np.max(np.array([d1, d2, d3]))])
plt.ylim(bottom=0)
if qois == 'stressYield':
	plt.xlabel(r'$\sigma_Y$ [MPa]', fontsize=24)
	plt.ylabel(r'$p(\sigma_Y)$', fontsize=24)
	plt.title(r'Stochastic Collocation: p.d.f of bcc W $\sigma_Y$', fontsize=24)
	plt.xlim(left=1000, right=10000)
	# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.0001))
elif qois == 'strainYield':
	plt.xlabel(r'$\varepsilon_Y$ [-]', fontsize=24)
	plt.ylabel(r'$p(\varepsilon_Y)$', fontsize=24)
	plt.title(r'Stochastic Collocation: p.d.f of bcc W $\varepsilon_Y$', fontsize=24)
	# plt.xlim(left=0.002)
	plt.xlim([0.002, 0.024])
	# ax.xaxis.set_major_locator(ticker.MultipleLocator(0.004))
else:
	print('plotQoIpdf.py: qois %s is not implemented.' % qois)

plt.show()


