
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# adopted from 'utils/plotConvergence.py'

### declare input params
magnifiedScale = 1e0 # scale y *= magnifiedScale

## assume
# (1) solving for MINIMIZATION problem
# (2) using flipped sign with a BO with MAXIMIZATION settings

feasible = np.loadtxt('postproc.feasible.dat')
Y = - np.loadtxt('postproc.output.dat') # flip signs here if want to solve for minimization
folderName = np.loadtxt('postproc.folder.dat', dtype=str)
numInitPoint = 2 # this is ONLY TRUE for this particular benchmark

trncLen = len(Y)
feasible = feasible[numInitPoint:(trncLen + 1) ]
Y = np.array(Y[numInitPoint:(trncLen + 1)] ) 


minList = [np.nonzero(feasible)[0][0]] # first index that is non-zero
minY = Y[np.nonzero(feasible)[0][0]] # first index that is non-zero

for i in range(len(Y)):
	if Y[i] < minY and feasible[i] == 1:
		minList.append(i)
		minY = Y[i]

minList.append(len(Y))

# print results
print('Convergence folderName:')
for j in range(len(minList) - 1): # remove the last folderName
	print('Iter %d: %s: %0.8f' % (minList[j], folderName[minList[j]], Y[minList[j]]))

print(minList)


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


convergedY = expandConvergenceObjective(minList, Y)

