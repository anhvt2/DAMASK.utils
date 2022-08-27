
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys, datetime
import argparse

# adopted from 'utils/plotConvergence.py' and 'utils/trackWorkerSchedule.py'

### declare input params

methodName_list = ['aphBO-2GP-3B', 'apBO-2GP-3B-EI', 'apBO-2GP-3B-PI', 'apBO-2GP-3B-UCB', 'pBO-2GP-3B-EI', 'pBO-2GP-3B-PI', 'pBO-2GP-3B-UCB', 'parallelMC'] 
color_list = ['blue', 'red', 'green', 'black', 'magenta', 'navy', 'orange', 'chocolate'] # https://matplotlib.org/tutorials/colors/colors.html
marker_list = ['o', '^', 'v', '<', 'p', 'P', 'D', 'X']
linestyle_list = ['-', '--', '-.', ':', '-', '--', '-.', ':']

parser = argparse.ArgumentParser(description='')
parser.add_argument("-bf", "--benchFunc", type=str)
parser.add_argument("-ee", "--errorevery", type=int, default=30) # set default period for errorbar; this option may be deprecated and no longer used
parser.add_argument("-ylog", "--ylog", type=bool, default=False)
parser.add_argument("-ylimmax", "--ylimmax", type=float, default=None)
parser.add_argument("-ylimmin", "--ylimmin", type=float, default=None)
args = parser.parse_args()
benchFunc = args.benchFunc
errorevery = args.errorevery
ylimmax = args.ylimmax
ylimmin = args.ylimmin

numInitPoint = 2 # this is ONLY TRUE for this particular benchmark
numTrials = 5
magnifiedScale = 1e0 # scale y *= magnifiedScale

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


mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
fig = plt.figure()

# loop over methods
totalTime_list = [] # truncate after this many seconds
for methodName, color, linestyle, marker in zip(methodName_list, color_list, linestyle_list, marker_list):
	#
	ordered_timeline_list = []
	Y_converged_list = []
	# 
	jRand = np.random.randint(numTrials)
	#
	for j in range(numTrials):
		folderName = methodName + '_' + benchFunc + '_Run%d' % (j+1)
		print('processing folder %s' % folderName)
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
		#
		if methodName == 'aphBO-2GP-3B':
			totalTime_list.append(timeline_ordered[-1])
		#
		# if j == jRand:
		# 	plt.plot(timeline_ordered, Y_converged, color=color, marker=marker, markersize=8, linestyle=linestyle, label=methodName, markevery=5)
	#
	#
	## combine all time-series across numTrials
	meanByTime = []
	stdByTime = []
	ubByTime = []
	lbByTime = []
	#
	# all_timeline = np.sort(np.array(ordered_timeline_list).ravel()) # this is NOT robust if list element is NOT of the same length
	all_timeline = np.array([])
	for tmp_list in ordered_timeline_list:
		all_timeline = np.hstack([all_timeline, tmp_list])
	#
	all_timeline = np.sort(all_timeline) # sort
	# debug
	# print(methodName)
	# print('len(all_timeline) = ', len(all_timeline))
	#
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
	#
	# convert to np.array()
	all_timeline = np.array(all_timeline)
	meanByTime = np.array(meanByTime)
	stdByTime = np.array(stdByTime)
	ubByTime = np.array(ubByTime)
	lbByTime = np.array(lbByTime)

	### plot
	plt.errorbar(all_timeline[::numTrials], meanByTime[::numTrials], yerr=stdByTime[::numTrials], linestyle=linestyle, color=color, marker=marker, markersize=5, markevery=5, errorevery=20)
	plt.plot(all_timeline[::numTrials], meanByTime[::numTrials], color=color, marker=marker, markersize=8, linestyle=linestyle, label=methodName, markevery=5)
	plt.fill_between(all_timeline[::numTrials], meanByTime[::numTrials] - stdByTime[::numTrials], meanByTime[::numTrials] + stdByTime[::numTrials], color=color, alpha=0.2)


if args.ylog:
	ax = plt.axes()
	ax.set_yscale('log')

plt.legend(loc='best', fontsize=24, markerscale=2)
plt.xlabel('time (seconds)', fontsize=24)
plt.ylabel('objective', fontsize=24)
plt.title('benchmark function = %s' % (benchFunc), fontsize=24)

if 'ylimmin' in locals() and 'ylimmax' in locals():
	if ylimmin != None and ylimmax != None and ylimmax > ylimmin:
		plt.ylim([ylimmin, ylimmax])
# else:
#	plt.gca().set_ylim(bottom=ylimmin)
# 	plt.gca().set_ylim(top=ylimmax)

plt.xlim([0, np.min(totalTime_list)])

plt.show()




