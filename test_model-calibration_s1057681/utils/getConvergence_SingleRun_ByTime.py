
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import os

# adopted from 'utils/plotConvergence.py' and 'utils/trackWorkerSchedule.py'

### declare input params
magnifiedScale = 1e0 # scale y *= magnifiedScale

## assume
# (1) solving for MINIMIZATION problem
# (2) using flipped sign with a BO with MAXIMIZATION settings

feasible = np.loadtxt('postproc.feasible.dat')
Y = - np.loadtxt('postproc.output.dat') # flip signs here if want to solve for minimization
folderName = np.loadtxt('postproc.folder.dat', dtype=str)
numInitPoint = 2 # this is ONLY TRUE for this particular benchmark
benchFunc = os.getcwd().split('/')[-1].split('_')[1]

startTime = np.loadtxt('postproc.startTime.dat', dtype=str)
stopTime = np.loadtxt('postproc.stopTime.dat', dtype=str)

startTimeObjs = []
stopTimeObjs = []
elapsedTimes = [] 

# eliminiate initial sampling points
trncLen = len(Y)
feasible = feasible[numInitPoint:(trncLen + 1) ]
Y = np.array(Y[numInitPoint:(trncLen + 1)] ) 
startTime = startTime[numInitPoint:]
stopTime = stopTime[numInitPoint:]

for i in range(len(Y)):
	startTimeObjs += [datetime.datetime.strptime(startTime[i][0] + ' ' + startTime[i][1], '%Y-%m-%d %H:%M:%S')] # join list
	stopTimeObjs += [datetime.datetime.strptime(stopTime[i][0] + ' ' + stopTime[i][1], '%Y-%m-%d %H:%M:%S')] # join list
	elapsedTimes += [ (stopTimeObjs[i] - startTimeObjs[i]).total_seconds() ]


elapsedTimes = np.array(elapsedTimes)
refStartTime = startTimeObjs[0] # reference start time
refStopTime = stopTimeObjs[-1] # reference stop time

# construct timeline -- time: x-axis; objective: y-axis
timeline = np.zeros(len(Y))
for i in range(len(timeline)):
	timeline[i] = (stopTimeObjs[i] - refStartTime).total_seconds()


# sort w.r.t. time

order_in_time = np.argsort(timeline)

timeline_ordered = timeline[order_in_time]
Y_ordered = Y[order_in_time]
tmpmin = Y_ordered[0] # init
for i in range(len(Y_ordered)):
	if tmpmin > Y_ordered[i]:
		tmpmin = Y_ordered[i]
	else:
		Y_ordered[i] = tmpmin # reassign 


mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24




plt.figure()
plt.plot(timeline[order_in_time], Y[order_in_time], color='b', marker='o', linestyle='', markersize=8)
plt.step(timeline_ordered, Y_ordered, where='post', linewidth=3, color='b', linestyle='-')
plt.xlabel('time (seconds)', fontsize=24)
plt.title('benchmark function = %s' % (benchFunc), fontsize=24)
plt.show()


