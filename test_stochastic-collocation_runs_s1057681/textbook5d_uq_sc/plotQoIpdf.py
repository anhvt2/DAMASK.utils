
# ../dakota -i textbook5d_uq_sc_pyImport.in > dakota.log
# grep -inr ' f1' dakota.log  > tmp.txt
# sed -i  's/ f1//g' tmp.txt
# mv tmp.txt strainYield.dat

# python3 plotQoIpdf.py --file=stressYield.dat

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
parser.add_argument("-f", "--file", type=str)
args = parser.parse_args()
fileName = args.file

from scipy.stats import norm # The standard Normal distribution
from scipy.stats import gaussian_kde as gaussian_kde # A standard kernel density estimator
d = np.loadtxt(fileName, delimiter=':')[:,1]
if fileName == 'stressYield.dat':
	d /= 1e6
q = gaussian_kde(d)

x = np.linspace(np.min(d), np.max(d), 5000)
# fig = plt.figure(figsize=(8,8))
fig, ax = plt.subplots(1, 1)
plt.plot(x, q(x), c='tab:blue', marker='o', linestyle='-', markersize=2)
plt.xlim([np.min(d), np.max(d)])
plt.ylim(bottom=0)
if fileName == 'stressYield.dat':
	plt.xlabel(r'$\sigma_Y$ [MPa]', fontsize=24)
	plt.ylabel(r'$p(\sigma_Y)$', fontsize=24)
	plt.title(r'Stochastic Collocation: p.d.f of fcc Cu $\sigma_Y$', fontsize=24)
	# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
elif fileName == 'strainYield.dat':
	plt.xlabel(r'$\varepsilon_Y$ [-]', fontsize=24)
	plt.ylabel(r'$p(\varepsilon_Y)$', fontsize=24)
	plt.title(r'Stochastic Collocation: p.d.f of fcc Cu $\varepsilon_Y$', fontsize=24)
	plt.xlim([0.002, 0.004])
	ax.xaxis.set_major_locator(ticker.MultipleLocator(0.001))
else:
	print('plotQoIpdf.py: fileName %s is not implemented.' % fileName)

plt.show()


