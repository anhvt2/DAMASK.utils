
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import argparse

# adopted from 'utils/plotConvergence.py'

### declare input params
methodName_list = ['aphBO-2GP-3B', 'apBO-2GP-3B-EI', 'apBO-2GP-3B-PI', 'apBO-2GP-3B-UCB', 'pBO-2GP-3B-EI', 'pBO-2GP-3B-PI', 'pBO-2GP-3B-UCB', 'parallelMC'] 
color_list = ['blue', 'red', 'green', 'black', 'magenta', 'navy', 'orange', 'chocolate'] # https://matplotlib.org/tutorials/colors/colors.html
marker_list = ['o', '^', 'v', '<', 'p', 'P', 'D', 'X']
linestyle_list = ['-', '--', '-.', ':', '-', '--', '-.', ':']

parser = argparse.ArgumentParser(description='')
parser.add_argument("-bf", "--benchFunc", type=str)
parser.add_argument("-ee", "--errorevery", type=int, default=5) # this option may be deprecated and no longer used
parser.add_argument("-ylog", "--ylog", type=bool, default=False)
parser.add_argument("-ylimmax", "--ylimmax", type=float, default=None)
parser.add_argument("-ylimmin", "--ylimmin", type=float, default=None)
args = parser.parse_args()
benchFunc = args.benchFunc
errorevery = args.errorevery
# benchFunc = 'egg'
ylimmax = args.ylimmax
ylimmin = args.ylimmin

benchFunc_list = ['egg', 'camel3', 'camel6', 'hart3', 'hart4', 'ackley', 'hart6', 'michal', 'rosen', 'dixonpr']
# folderNamePrefix = methodName + benchFunc # 'aphBO-2GP-3B_hart6'
numTrials = 5
magnifiedScale = 1e0 # scale y *= magnifiedScale
numInitPoint = 2 # this is ONLY TRUE for this particular benchmark

# "egg") # 2d
# "camel3") # 2d
# "camel6") # 2d
# "hart3") # 3d
# "hart4") # 4d
# "ackley") # d: set d = 5d
# "hart6") # 6d
# "michal") # d: set d = 10
# "rosen") # d: set d = 20


## ASSUMPTIONS:
# (1) solving for MINIMIZATION problem
# (2) using flipped sign with a BO with MAXIMIZATION settings


### implement user-defined function

def expandConvergenceObjective(minList, Y):
	"""
	This function takes 
		(1) minList 
		(2) Y
	and return
		(1) convergedY: a list of objectives of len(Y) that converges monotonically
	"""
	convergedY = []
	for i in range(len(Y)):
		# find the lastIndex in minList that is less than i
		warmStartIndex = 0
		for j in range(warmStartIndex, len(minList)):
			if minList[j] > i:
				lastIndex = minList[j-1]
				break
		convergedY.append(Y[minList[j-1]])
	return convergedY



### main function -- plot
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

fig = plt.figure()

# loop over methods
iterScale = [] # truncate after this many iterations
for methodName, color, linestyle, marker in zip(methodName_list, color_list, linestyle_list, marker_list):
	#
	convergedY = []
	#
	for i in range(numTrials):
		folderName = methodName + '_' + benchFunc + '_Run%d' % (i+1)
		print('processing folder %s' % folderName)
		#
		feasible = np.loadtxt(folderName + '/postproc.feasible.dat')
		Y = - np.loadtxt(folderName + '/postproc.output.dat') # flip signs here if want to solve for minimization
		iterFolderName = np.loadtxt(folderName + '/postproc.folder.dat', dtype=str)
		#
		trncLen = len(Y)
		feasible = feasible[numInitPoint:(trncLen + 1) ]
		Y = np.array(Y[numInitPoint:(trncLen + 1)] ) 
		#
		minList = [np.nonzero(feasible)[0][0]] # first index that is non-zero
		minY = Y[np.nonzero(feasible)[0][0]] # first index that is non-zero
		# 
		for i in range(len(Y)):
			if Y[i] < minY and feasible[i] == 1:
				minList.append(i)
				minY = Y[i]
		#
		minList.append(len(Y))
		# 
		# print results
		print('Convergence iterFolderName:')
		for j in range(len(minList) - 1): # remove the last iterFolderName
			print('Iter %d: %s: %0.8f' % (minList[j], iterFolderName[minList[j]], Y[minList[j]]))
		#
		print(minList)
		# 
		convergedY.append(expandConvergenceObjective(minList, Y))
	#
	# sanity check
	if len(convergedY) != numTrials: # check 
		print('len(convergedY) is NOT equal to numTrials!') # print error message
	#
	### compute mean, variance, upper bound, lower bound BY ITERATION
	meanByIter = []
	stdByIter = []
	ubByIter = []
	lbByIter = []
	# get max number of iterations
	maxIter = 0
	#
	for i in range(numTrials):
		maxIter = np.max([maxIter, len(convergedY[i])])
	#
	iterScale.append([maxIter])
	#
	for i in range(maxIter):
		tmp_list = []
		for j in range(numTrials):
			if i < len(convergedY[j]): # do not include if the list is shorter than called index
				tmp_list.append(convergedY[j][i])
		#
		meanByIter.append(np.mean(tmp_list))
		stdByIter.append(np.std(tmp_list))
		ubByIter.append(np.max(tmp_list))
		lbByIter.append(np.min(tmp_list))
	#
	meanByIter = np.array(meanByIter)
	stdByIter = np.array(stdByIter)
	ubByIter = np.array(ubByIter)
	lbByIter = np.array(lbByIter)
	#
	#
	plt.plot(meanByIter, color=color, marker=marker, markersize=8, linestyle=linestyle, label=methodName, markevery=5)
	plt.errorbar(range(len(meanByIter)), meanByIter, yerr=stdByIter, color=color, linestyle=linestyle, markevery=5, errorevery=errorevery)
	plt.fill_between(range(len(meanByIter)), meanByIter - stdByIter, meanByIter + stdByIter, color=color, alpha=0.2)

if args.ylog:
	ax = plt.axes()
	ax.set_yscale('log')

if 'ylimmin' in locals() and 'ylimmax' in locals():
	if ylimmin != None and ylimmax != None and ylimmax > ylimmin:
		plt.ylim([ylimmin, ylimmax])
# else:
#	plt.gca().set_ylim(bottom=ylimmin)
# 	plt.gca().set_ylim(top=ylimmax)

plt.legend(loc='best', fontsize=24, markerscale=2)
plt.xlabel('iteration', fontsize=24)
plt.ylabel('objective', fontsize=24)
plt.title('benchmark function = %s' % (benchFunc), fontsize=24)
# plt.ylim([0, 10])
plt.xlim([0, np.median(iterScale)])
plt.show()



