
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys, datetime

# adopted from 'utils/plotConvergence.py' and 'utils/trackWorkerSchedule.py'

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

ordered_timeline_list = []
Y_converged_list = []


for j in range(numTrials):
	folderName = methodName + '_' + benchFunc + '_Run%d' % (j+1)
	#
	feasible = np.loadtxt(folderName + '/postproc.feasible.dat')
	Y = - np.loadtxt(folderName + '/postproc.output.dat') # flip signs here if want to solve for minimization
	#
	startTime = np.loadtxt(folderName + '/postproc.startTime.dat', dtype=str)
	stopTime = np.loadtxt(folderName + '/postproc.stopTime.dat', dtype=str)
	#
	startTimeObjs = []
	stopTimeObjs = []
	elapsedTimes = [] 
	#
	# eliminiate initial sampling points
	trncLen = len(Y)
	feasible = feasible[numInitPoint:(trncLen + 1) ]
	Y = np.array(Y[numInitPoint:(trncLen + 1)] ) 
	startTime = startTime[numInitPoint:]
	stopTime = stopTime[numInitPoint:]
	#
	for i in range(len(Y)):
		startTimeObjs += [datetime.datetime.strptime(startTime[i][0] + ' ' + startTime[i][1], '%Y-%m-%d %H:%M:%S')] # join list
		stopTimeObjs += [datetime.datetime.strptime(stopTime[i][0] + ' ' + stopTime[i][1], '%Y-%m-%d %H:%M:%S')] # join list
		elapsedTimes += [ (stopTimeObjs[i] - startTimeObjs[i]).total_seconds() ]
	#
	elapsedTimes = np.array(elapsedTimes)
	refStartTime = startTimeObjs[0] # reference start time
	refStopTime = stopTimeObjs[-1] # reference stop time
	#
	# construct timeline -- time: x-axis; objective: y-axis
	timeline = np.zeros(len(Y))
	for i in range(len(timeline)):
		timeline[i] = (stopTimeObjs[i] - refStartTime).total_seconds()
	#
	order_in_time = np.argsort(timeline)
	Y_ordered = Y[order_in_time]
	timeline_ordered = timeline[order_in_time]
	Y_converged = Y_ordered.copy() # make a clean copy of Y_order
	for i in range(1, len(Y_converged)):
		if Y_converged[i] > Y_converged[i-1]:
			Y_converged[i] = Y_converged[i-1]
	#
	# append a list copy of 'timeline_ordered' and 'Y_converged'
	ordered_timeline_list.append(list(timeline_ordered))
	Y_converged_list.append(list(Y_converged))


## implement 'getBestobjectiveAtParticularTime()'
def getBestObjectiveAtParticularTime(queryTime, timelineOrdered, YOrdered):
	n = len(timelineOrdered)
	# find the last index of timelineOrdered[lastIndex] that timelineOrdered[lastIndex] < queryTime < timelineOrdered[lastIndex + 1]
	lastIndex = 0 # init
	# safeguard
	if queryTime < timelineOrdered[0]:
		return np.nan
	elif queryTime > timelineOrdered[-1]:
		return YOrdered[-1]
	else: # main calculation if all safeguards pass
		for i in range(n-1): # avoid the last index
			if timelineOrdered[i] <= queryTime and queryTime <= timelineOrdered[i+1]:
				lastIndex = i
				queryY = YOrdered[lastIndex]
				return queryY


## combine all time-series across numTrials
meanByTime = []
stdByTime = []
ubByTime = []
lbByTime = []

# all_timeline = np.sort(np.array(ordered_timeline_list).ravel()) # this is NOT robust if list element is NOT of the same length
all_timeline = np.array([])
	for tmp_list in ordered_timeline_list:
		all_timeline = np.hstack([all_timeline, tmp_list])

all_timeline = np.sort(all_timeline) # sort

for i in range(len(all_timeline)):
	tmp_list = []
	query_time = all_timeline[i]
	# collect best objective in all time-series in 'tmp_list'
	for j in range(numTrials):
		# see above (lines 74-84) implementations: 'ordered_timeline_list' and 'Y_converged'
		timeline_ordered = ordered_timeline_list[j]
		Y_ordered = Y_converged_list[j]
		#
		if isinstance(query_time, float) == True: # add safeguards
			tmpval = getBestObjectiveAtParticularTime(query_time, timeline_ordered, Y_ordered)
			if tmpval != np.nan:
				tmp_list.append(tmpval)
	#
	# get statistics after collecting 'tmp_list'
	if len(tmp_list) > 0: # add safeguard
		meanByTime.append(np.mean(tmp_list))
		stdByTime.append(np.std(tmp_list))
		ubByTime.append(np.max(tmp_list))
		lbByTime.append(np.min(tmp_list))

all_timeline = np.array(all_timeline)
meanByTime = np.array(meanByTime)
stdByTime = np.array(stdByTime)
ubByTime = np.array(ubByTime)
lbByTime = np.array(lbByTime)

### plot
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

fig = plt.figure()

plt.errorbar(all_timeline, meanByTime, yerr=stdByTime, color='b', marker='o', markersize=5, markevery=5, errorevery=20)
plt.xlabel('time (seconds)', fontsize=24)
plt.ylabel('objective', fontsize=24)
plt.title('Convergence plot: %s' % (methodName + '_' + benchFunc), fontsize=24)

plt.show()


