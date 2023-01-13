
# python postproc script for log.postproc.txt
# clone from $doc/giw/convPlot.py or test.py (deprecated)
# adopt from /home/anhvt89/Documents/cellular/numExample/results/visualResContinuous_5Oct17.py

# use 6Jan18 for paper results

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

### declare input params
magnifiedScale = 1e0 # scale y *= magnifiedScale


### read outputs

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

print('By default, BO solves for maxmimization constrained optimization problems. ')
flagMaxOrMin = input("What settings do you want, min or max? [max or 1/min or 0]: ")

try:
	flagMaxOrMin = int(flagMaxOrMin)
except:
	print()


"""
By DEFAULT, the package is set up to solve a MAXIMIZATION problem. 
If minimization problem is considered, then we assume that the objective has been flipped previously.
Now we will flip it again to restore its originality. 
"""

# boolMaxOrMin = 0: maximization problem is considered
# boolMaxOrMin = 1: minimization problem is considered

if isinstance(flagMaxOrMin, int):
	if flagMaxOrMin == 0: # max
		Y = np.loadtxt('postproc.output.dat')
		boolMaxOrMin = 0 # assign bool
	elif flagMaxOrMin == 1: # min
		Y = - np.loadtxt('postproc.output.dat')
		boolMaxOrMin = 1 # assign bool
elif isinstance(flagMaxOrMin, str):
	if flagMaxOrMin.lower() == 'max':
		Y = np.loadtxt('postproc.output.dat')
		boolMaxOrMin = 0 # assign bool
	elif flagMaxOrMin.lower() == 'min':
		Y = - np.loadtxt('postproc.output.dat')
		boolMaxOrMin = 1 # assign bool

feasible = np.loadtxt('postproc.feasible.dat')
feasible[np.where(np.isnan(feasible))] = 0 # replace nan with 0
print(len(Y))
print(len(feasible))
trncLen = len(Y)

folderName = np.loadtxt('postproc.folder.dat', dtype=str)
# for i in range(len(folderName)): # polish folderName
# 	folderName[i] = np.string_.split(folderName[i],'/')[0]

# delete selected point
# Y = np.delete(Y,8)
# feasible = np.delete(feasible,8)
# folderName = np.delete(folderName,8)

# # with initial-sampling
# feasible = feasible[:(trncLen + 1) ]
# Y = np.array(Y[:(trncLen + 1)] ) 

# # without initial sampling
# Y = np.delete(Y, np.arange(1, 480))
# feasible = np.delete(feasible, np.arange(1, 480))
# folderName = np.delete(folderName, np.arange(1, 480))

# print(len(Y))
# print(lenfeasible)
Y[feasible == 0] = 0


if boolMaxOrMin: # if minimization problem is considered

	minList = [np.nonzero(feasible)[0][0]] # first index that is non-zero
	minY = Y[np.nonzero(feasible)[0][0]] # first index that is non-zero

	for i in range(len(Y)):
		if Y[i] < minY and feasible[i] == 1:
			minList.append(i)
			minY = Y[i]

	minListPlot = [np.nonzero(feasible)[0][0]]
	for i in range(1,len(minList) - 1):
		minListPlot.append(minList[i])

	minListPlot.append(minList[-1])
	minListPlot.append(len(Y))
	Yplot = Y[minListPlot[:len(minListPlot) - 1]]
	Yplot = list(Yplot)
	Yplot.append(Y[minList[-1]])

	# print results
	print('Convergence folderName:')
	for j in range(len(minListPlot) - 1): # remove the last folderName
		print('Iter %d: %s: %0.8f' % (minListPlot[j], folderName[minListPlot[j]], Yplot[j]))

	print(minListPlot)


	### plot 
	mpl.rcParams['xtick.labelsize'] = 24
	mpl.rcParams['ytick.labelsize'] = 24

	iterList = np.array(range(len(Y)))
	feasPlot, = plt.plot(iterList[feasible == 1], magnifiedScale * Y[feasible == 1], 'bo', label='feasible', ms=8, mew=5)
	infeasPlot, = plt.plot(iterList[feasible == 0], magnifiedScale * Y[feasible == 0], 'rx', label='infeasible', ms=8, mew=5)
	plt.legend(handles=[feasPlot, infeasPlot], fontsize=26, markerscale=2, loc='best')# bbox_to_anchor=(0.35, 0.20)) # loc = {'lower/upper right/left', 'best'}
	# plt.legend()
	plt.step(minListPlot, magnifiedScale * np.array(Yplot), where='post', linewidth=3, color='g', linestyle='-')
	# plt.errorbar(range(len(Y)), Y, yerr=mse, fmt='o',ecolor='g')
	# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	plt.xlabel('Number of functional evaluations',fontsize=26)
	plt.ylabel(r'Objective',fontsize=26)
	plt.title('aphBO-2GP-3B: Convergence plot of %s' % modelName,fontsize=26)
	plt.ylim(0, 50)
	# plt.ylim(-5, 500)
	# plt.ylim(0.60, 0.80)
	plt.show()

else: # if maximization problem is considered

	maxList = [np.nonzero(feasible)[0][0]] # first index that is non-zero
	maxY = Y[np.nonzero(feasible)[0][0]] # first index that is non-zero

	for i in range(len(Y)):
		if Y[i] > maxY and feasible[i] == 1:
			maxList.append(i)
			maxY = Y[i]

	maxListPlot = [np.nonzero(feasible)[0][0]]
	for i in range(1,len(maxList) - 1):
		maxListPlot.append(maxList[i])

	maxListPlot.append(maxList[-1])
	maxListPlot.append(len(Y))
	Yplot = Y[maxListPlot[:len(maxListPlot) - 1]]
	Yplot = list(Yplot)
	Yplot.append(Y[maxList[-1]])

	# print results
	print('Convergence folderName:')
	for j in range(len(maxListPlot) - 1): # remove the last folderName
		print('Iter %d: %s: %0.8f' % (maxListPlot[j], folderName[maxListPlot[j]], Yplot[j]))

	print(maxListPlot)


	### plot 
	mpl.rcParams['xtick.labelsize'] = 24
	mpl.rcParams['ytick.labelsize'] = 24

	iterList = np.array(range(len(Y)))
	feasPlot, = plt.plot(iterList[feasible == 1], magnifiedScale * Y[feasible == 1], 'bo', label='feasible', ms=8, mew=5)
	infeasPlot, = plt.plot(iterList[feasible == 0], magnifiedScale * Y[feasible == 0], 'rx', label='infeasible', ms=8, mew=5)
	plt.legend(handles=[feasPlot, infeasPlot], fontsize=26, markerscale=2, loc='best')# bbox_to_anchor=(0.35, 0.20)) # loc = {'lower/upper right/left', 'best'}
	# plt.legend()
	plt.step(maxListPlot, magnifiedScale * np.array(Yplot), where='post', linewidth=3, color='g', linestyle='-')
	# plt.errorbar(range(len(Y)), Y, yerr=mse, fmt='o',ecolor='g')
	# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	plt.xlabel('Number of functional evaluations',fontsize=26)
	plt.ylabel(r'Objective',fontsize=26)
	plt.title('aphBO-2GP-3B: Convergence plot of %s' % modelName,fontsize=26)
	plt.ylim(0, 50)
	# plt.ylim(0.60, 0.80)
	plt.show()




