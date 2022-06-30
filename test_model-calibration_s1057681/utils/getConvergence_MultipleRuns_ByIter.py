
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# adopted from 'utils/plotConvergence.py'

### declare input params
methodName = 'aphBO-2GP-3B'
benchFunc = 'hart6' # ['egg', 'camel3', 'camel6', 'hart3', 'hart4', 'ackley', 'hart6', 'michal', 'rosen', 'dixonpr']
folderNamePrefix = methodName + benchFunc # 'aphBO-2GP-3B_hart6'
numInitPoint = 2 # this is ONLY TRUE for this particular benchmark
numTrials = 5
magnifiedScale = 1e0 # scale y *= magnifiedScale

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



### main function

convergedY = []

for i in range(numTrials):
	folderName = methodName + '_' + benchFunc + '_Run%d' % (i+1)
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

# sanity check
if len(convergedY) != numTrials: # check 
	print('len(convergedY) is NOT equal to numTrials!') # print error message

### compute mean, variance, upper bound, lower bound BY ITERATION
meanByIter = []
stdByIter = []
ubByIter = []
lbByIter = []

# get max number of iterations
maxIter = 0

for i in range(numTrials):
	maxIter = np.max([maxIter, len(convergedY[i])])

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


### plot (optional)
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

fig = plt.figure()

plt.plot(meanByIter, color='b', marker='o', markersize=5)
plt.xlabel('iteration', fontsize=24)
plt.ylabel('objective', fontsize=24)
plt.title('Convergence plot: %s' % (methodName + '_' + benchFunc), fontsize=24)


for i in range(maxIter):
	if i % 5 == 0:
		plt.errorbar(i, meanByIter[i], yerr=stdByIter[i], color='b')

plt.show()



