
import numpy as np
import glob, os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24 

from scipy.stats import norm # The standard Normal distribution
from scipy.stats import gaussian_kde as GKDE # A standard kernel density estimator

def readYoungModulus(folderPrefix):
	folders = glob.glob(folderPrefix + '*')
	results = []
	for folder in folders:
		tmpResult = np.loadtxt(folder + '/postProc/youngModulus.out')
		# print(tmpResult)
		results += [tmpResult]
	# print(results)
	youngModulusArray = np.array(results)
	# print stats
	print('\nStats for %s' % folderPrefix)
	print('%s: min = %.6e' % (folderPrefix, np.min(youngModulusArray)))
	print('%s: max = %.6e' % (folderPrefix, np.max(youngModulusArray)))
	print('%s: avg = %.6e' % (folderPrefix, np.average(youngModulusArray)))
	print('%s: std = %.6e' % (folderPrefix, np.std(youngModulusArray)))
	print('\n')
	return youngModulusArray

dimCell = 64

a = readYoungModulus('%dx%dx%d-mu-1.50-sigma-0.15' % (dimCell, dimCell, dimCell))
b = readYoungModulus('%dx%dx%d-mu-1.75-sigma-0.15' % (dimCell, dimCell, dimCell))
c = readYoungModulus('%dx%dx%d-mu-2.00-sigma-0.15' % (dimCell, dimCell, dimCell))
d = readYoungModulus('%dx%dx%d-mu-2.20-sigma-0.15' % (dimCell, dimCell, dimCell))
e = readYoungModulus('%dx%dx%d-mu-2.30-sigma-0.15' % (dimCell, dimCell, dimCell))
f = readYoungModulus('%dx%dx%d-mu-2.30-sigma-0.40' % (dimCell, dimCell, dimCell))
g = readYoungModulus('%dx%dx%d-mu-2.50-sigma-0.50' % (dimCell, dimCell, dimCell))

minLim = np.min(np.vstack((a,b,c,d,e,f,g)))
maxLim = np.max(np.vstack((a,b,c,d,e,f,g)))

qplot = np.linspace(minLim, maxLim, num=100)
q1, q2, q3, q4 = GKDE(a), GKDE(b), GKDE(c), GKDE(d)
q5, q6, q7 = GKDE(e), GKDE(f), GKDE(g)


cs =  ['b','g', 'r', 'c', 'm', 'y', 'k']
qs =  [q1, q2, q3, q4, q5, q6, q7]
labels = [  r'$\mu = 1.50; \sigma = 0.15$',
			r'$\mu = 1.75; \sigma = 0.15$',
			r'$\mu = 2.00; \sigma = 0.15$',
			r'$\mu = 2.20; \sigma = 0.15$',
			r'$\mu = 2.30; \sigma = 0.15$',
			r'$\mu = 2.30; \sigma = 0.40$',
			r'$\mu = 2.50; \sigma = 0.50$',
			]

plt.figure()
for q,c,labelstr in zip(qs, cs, labels):
	plt.plot(qplot, q(qplot), c, linewidth=4, label=labelstr)


plt.legend(fontsize=24, markerscale=2, loc=7) # bbox_to_anchor=(0.35, 0.20))
plt.xlabel('Young Modulus (GPa)',fontsize=26)
plt.ylabel('pdf',fontsize=26)
plt.show()

