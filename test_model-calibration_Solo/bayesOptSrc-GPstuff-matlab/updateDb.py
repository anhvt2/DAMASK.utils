# only done globally
# do not touch locally
import os, glob
import numpy as np
from natsort import natsorted, ns

parentPath = os.getcwd()
modelName = np.loadtxt('modelName.dat', dtype=str)
folderList = natsorted(glob.glob('%s_Iter*/' % modelName))

# S = np.empty([len(folderList), len(np.loadtxt(parentPath + '/' + folderList[0] + '/input.dat', delimiter=','))]) 
S = np.empty([len(folderList), len(np.loadtxt(parentPath + '/' + folderList[0] + '/projectedInput.dat', delimiter=','))]) # for high-dimensional problems
Y = np.empty([len(folderList)])
F = np.empty([len(folderList)])
C = np.empty([len(folderList)])

for i in range(len(folderList)):
	folderName = folderList[i]
	# x = np.loadtxt(parentPath + '/' + folderList[i] + 'input.dat', delimiter=',')
	z = np.loadtxt(parentPath + '/' + folderList[i] + 'projectedInput.dat', delimiter=',')
	# output.dat
	if os.path.isfile(parentPath + '/' + folderList[i] + 'output.dat'):
		y = np.loadtxt(parentPath + '/' + folderList[i] + 'output.dat')
	else:
		y = 0
	# feasible.dat
	if os.path.isfile(parentPath + '/' + folderList[i] + 'feasible.dat'):
		f = np.loadtxt(parentPath + '/' + folderList[i] + 'feasible.dat')
	else: 
		f = 0
	# complete.dat
	if os.path.isfile(parentPath + '/' + folderList[i] + 'complete.dat'):
		c = np.loadtxt(parentPath + '/' + folderList[i] + 'complete.dat')
	else: 
		c = 0
	# append to list
	S[i,:] = z
	Y[i] = y
	F[i] = f
	C[i] = c

np.savetxt('S.dat', S, fmt='%.16f', delimiter=',')
np.savetxt('Y.dat', Y, fmt='%.16f', delimiter=',')
np.savetxt('F.dat', F, fmt='%d', delimiter=',')
np.savetxt('C.dat', C, fmt='%d', delimiter=',')

