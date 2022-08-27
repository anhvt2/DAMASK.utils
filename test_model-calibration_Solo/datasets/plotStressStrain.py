
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--f", help='file name with first header row', type=str, required=True)
args = parser.parse_args()

fileName = args.f

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

x = np.loadtxt(fileName, skiprows=1)
plt.plot(x[:,0], x[:,1], 'bo', ms=5)
plt.xlabel(r'$\varepsilon$ [-]', fontsize=24)
plt.ylabel(r'$\sigma$ [MPa]', fontsize=24)
plt.title(fileName, fontsize=24)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.show()
