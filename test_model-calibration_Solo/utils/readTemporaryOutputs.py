
import numpy as np
import os, glob

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

if isinstance(flagMaxOrMin, int):
	if flagMaxOrMin == 0: # max
		Y = np.loadtxt('postproc.output.dat')
	elif flagMaxOrMin == 1: # min
		Y = - np.loadtxt('postproc.output.dat')
elif isinstance(flagMaxOrMin, str):
	if flagMaxOrMin.lower() == 'max':
		Y = np.loadtxt('postproc.output.dat')
	elif flagMaxOrMin.lower() == 'min':
		Y = - np.loadtxt('postproc.output.dat')

feasible = np.loadtxt('postproc.feasible.dat')
trncLen = len(Y)

folderName = np.loadtxt('postproc.folder.dat', dtype=str)


feasible = feasible[:(trncLen + 1) ]
Y = np.array(Y[:(trncLen + 1)] ) 



Y[feasible == 0] = 0

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

