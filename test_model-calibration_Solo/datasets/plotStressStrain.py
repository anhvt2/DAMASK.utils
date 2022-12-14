
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.ndimage as ndi
import scipy.fftpack
import statsmodels.api as sm
from scipy.interpolate import interp1d

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--f", help='file name with first header row', type=str, required=True)
args = parser.parse_args()

fileName = args.f
# fileName = 'true_SS304L_EngStress_EngStrain_exp_4A1.dat'

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

x = np.loadtxt(fileName, skiprows=1)
h = x[1,0] - x[0,0] # strain step size

fig, ax = plt.subplots()
ax.plot(x[:,0], x[:,1], 'bo', ms=5)
ax.set_xlabel(r'$\varepsilon$ [-]', fontsize=24)
ax.set_ylabel(r'$\sigma$ [MPa]', fontsize=24, color='blue')
ax.set_title(fileName, fontsize=24)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# https://stackoverflow.com/questions/36252434/predicting-on-new-data-using-locally-weighted-regression-loess-lowess
lowess = sm.nonparametric.lowess
lowess_data = lowess(x[:,1], x[:,0], frac=1./len(x))
lowess_x = list(zip(*lowess_data))[0]
lowess_y = list(zip(*lowess_data))[1]
f = interp1d(lowess_x, lowess_y, bounds_error=False)
from scipy.interpolate import interp1d
vareps_linspace = np.linspace(x[:,0].min(), x[:,0].max(), num=200)
sigma_linspace = f(vareps_linspace)

ax2 = ax.twinx()
ax2.plot(vareps_linspace, np.gradient(sigma_linspace), 'rs', ms=2)
ax2.set_ylabel(r'$\frac{d\sigma}{d\varepsilon}$', fontsize=24, color='red')

plt.figure()

# print(np.log(vareps_linspace))
plt.plot(np.log(vareps_linspace), np.log(sigma_linspace), 'bo')
slope, intercept, r, p, se  = scipy.stats.linregress(np.log(vareps_linspace), np.log(sigma_linspace))
x_ = np.linspace(np.log(vareps_linspace).min(), np.log(vareps_linspace).max(), num=500)
print(slope)
plt.xlabel(r'$\log(\varepsilon)$', fontsize=24)
plt.ylabel(r'$\log(\sigma)$', fontsize=24)
plt.plot(x_, slope * x_ + intercept, 'rx', label='OLS')
plt.legend()
plt.show()
