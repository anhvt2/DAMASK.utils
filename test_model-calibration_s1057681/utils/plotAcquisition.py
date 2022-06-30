
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys, datetime

# adopted from 'utils/plotConvergence.py'

### declare input params
magnifiedScale = 1e0 # scale y *= magnifiedScale

## assume
# (1) solving for MINIMIZATION problem
# (2) using flipped sign with a BO with MAXIMIZATION settings

if os.path.isfile('batchSettings.dat'):
		batchSettings = np.loadtxt('batchSettings.dat', dtype=int)
else:
		batchSettings = np.loadtxt('../batchSettings.dat', dtype=int)

if os.path.isfile('modelName.dat'):
		modelName = np.loadtxt('modelName.dat', dtype=str)
else:
		modelName = np.loadtxt('../modelName.dat', dtype=str)


if os.path.exists('postproc.output.dat'):
	flagRebuild = input("Do you want to overwrite current postproc.*.dat by re-running utils/buildLogs.sh? [y/n or 1/0]: ")
	try:
		flagRebuild = int(flagRebuild)
	except:
		print()
	if isinstance(flagRebuild, int):
		if flagRebuild == 1:
			os.system('sh utils/buildLogs.sh')
	elif isinstance(flagRebuild, str):
		if flagRebuild.lower() == 'y' or flagRebuild.lower() == 'yes':
			os.system('sh utils/buildLogs.sh')
else:
	os.system('sh utils/buildLogs.sh')


feasible = np.loadtxt('postproc.feasible.dat')
Y = - np.loadtxt('postproc.output.dat') # flip signs here if want to solve for minimization
folderName = np.loadtxt('postproc.folder.dat', dtype=str)
acquisitionScheme = np.loadtxt('postproc.acquisitionScheme.dat', dtype=str)

## 

# init counters
counterUCB = [0]
counterEI  = [0]
counterPI  = [0]
n = len(acquisitionScheme)

for i in range(n):
	if acquisitionScheme[i] == 'UCB':
		counterUCB.append( int(counterUCB[-1] + 1) )
		counterEI.append(  int(counterEI[-1])      )
		counterPI.append(  int(counterPI[-1])      )
	elif acquisitionScheme[i] == 'EI':
		counterUCB.append( int(counterUCB[-1])     )
		counterEI.append(  int(counterEI[-1] + 1)  )
		counterPI.append(  int(counterPI[-1])      )
	elif acquisitionScheme[i] == 'PI':
		counterUCB.append( int(counterUCB[-1])     )
		counterEI.append(  int(counterEI[-1])      )
		counterPI.append(  int(counterPI[-1] + 1)  )
	else:
		counterUCB.append( int(counterUCB[-1])     )
		counterEI.append(  int(counterEI[-1])      )
		counterPI.append(  int(counterPI[-1])      )


# UCB: top
# EI: middle
# PI: bottom
ucb_ei_borderline = []
ei_pi_borderline = []

for i in range(n):
	tmp_total = counterUCB[i] + counterEI[i] + counterPI[i]
	if tmp_total > 0:
		ucb_ei_borderline.append( float((counterEI[i] + counterPI[i]) / tmp_total) )
		ei_pi_borderline.append( float( counterPI[i] / tmp_total) )
	else:
		ucb_ei_borderline.append( float(2/3.) )
		ei_pi_borderline.append(  float(1/3.) )

ucb_ei_borderline = np.array(ucb_ei_borderline)
ei_pi_borderline = np.array(ei_pi_borderline)

## print diagnostics
print('Total number of UCB acquisition function used = %d' % counterUCB[-1])
print('Total number of EI acquisition function used = %d' % counterEI[-1])
print('Total number of PI acquisition function used = %d' % counterPI[-1])

### plot (optional)
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

# plot portfolio
fig = plt.figure()

plt.plot(range(n), ucb_ei_borderline, color='k', marker='o', markersize=5, linestyle='-', linewidth=2)
plt.plot(range(n), ei_pi_borderline,  color='k', marker='o', markersize=5, linestyle='-', linewidth=2)

plt.fill_between(range(n), ucb_ei_borderline, np.ones(n), color='b', alpha=0.2, label='UCB')
plt.fill_between(range(n), ei_pi_borderline, ucb_ei_borderline, color='r', alpha=0.2, label='EI')
plt.fill_between(range(n), np.zeros(n), ei_pi_borderline, color='g', alpha=0.2, label='PI')

plt.legend(loc='best', fontsize=24)

plt.xlabel('iteration', fontsize=24)
# plt.ylabel('objective', fontsize=24)
plt.title('Acquisition portfolio: %s' % os.getcwd().split('/')[-1], fontsize=24)
plt.ylim([0,1])
plt.xlim([0,n-1])


# plot number of acquisition functions used
fig = plt.figure()
plt.plot(range(len(counterUCB)), counterUCB, color='b', marker='o', markersize=8, linestyle='-' , linewidth=2, label='UCB', markevery=5)
plt.plot(range(len(counterEI)) , counterEI , color='r', marker='s', markersize=8, linestyle='--', linewidth=2, label='EI' , markevery=5)
plt.plot(range(len(counterPI)) , counterPI , color='g', marker='D', markersize=8, linestyle=':' , linewidth=2, label='PI' , markevery=5)

plt.legend(loc='best', fontsize=24, markerscale=2)
plt.title('Acquisition track: %s' % os.getcwd().split('/')[-1], fontsize=24)
plt.xlabel('iteration', fontsize=24)
plt.ylabel('number of acquisition function used', fontsize=24)
plt.xlim([0,n-1])
plt.ylim([0, np.max([counterUCB[-1], counterEI[-1], counterPI[-1]])])

plt.show()





