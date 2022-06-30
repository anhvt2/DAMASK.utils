
import numpy as np
import os, time, glob, sys, natsort, datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

### read input files

acquisition = np.loadtxt('postproc.acquisitionScheme.dat', dtype=str)
startTime = np.loadtxt('postproc.startTime.dat', dtype=str)
stopTime = np.loadtxt('postproc.stopTime.dat', dtype=str)
batchSettings = np.loadtxt('batchSettings.dat')


n = len(acquisition)
numWorkers = int(batchSettings.sum()) # get total number of workers
numInitPoint = 0
for i in range(n):
	if acquisition[i] != '0':
		numInitPoint = i
		break


### convert and process datetime object
startTimeObjs = []
stopTimeObjs = []
elapsedTimes = []

for i in range(n):
	startTimeObjs += [datetime.datetime.strptime(startTime[i][0] + ' ' + startTime[i][1], '%Y-%m-%d %H:%M:%S')] # join list
	stopTimeObjs += [datetime.datetime.strptime(stopTime[i][0] + ' ' + stopTime[i][1], '%Y-%m-%d %H:%M:%S')] # join list
	elapsedTimes += [ (stopTimeObjs[i] - startTimeObjs[i]).total_seconds() ]

elapsedTimes = np.array(elapsedTimes)

## get reference start/stop datetime object for start/stop
refStartTime = startTimeObjs[numInitPoint] # reference start time
# get max refStopTime
refStopTime = stopTimeObjs[0]
for i in range(len(stopTimeObjs)):
	if refStopTime < stopTimeObjs[i]:
		refStopTime = stopTimeObjs[i] # reference stop time

### assign workers into working schedule
availableTime = np.zeros([numWorkers])
currentStartTimeWorker = np.zeros([numWorkers])
currentStopTimeWorker = np.zeros([numWorkers])
workAssignmentTable = np.zeros([n, 2]) # 2 cols: 1st col = iteration, 2nd = worker id

# i: index loop over number of iterations
# j: index loop over number of workers

for i in range(numInitPoint):
	workAssignmentTable[i] = i, np.nan # initialize and mask with np.nan

for i in range(numInitPoint,n): # exclude initial points
	for j in range(numWorkers):
		# if startTime[i] > availableTime[j]
		# then assign _Iter${i} to worker${j}
		# 		and return to proceed ${i+1}
		if int(startTimeObjs[i].strftime('%Y%m%d%H%M%S')) > availableTime[j]: 
			availableTime[j] = int(stopTimeObjs[i].strftime('%Y%m%d%H%M%S'))
			currentStartTimeWorker[j] = int(startTimeObjs[i].strftime('%Y%m%d%H%M%S'))
			currentStopTimeWorker[j]  = int(stopTimeObjs[i].strftime('%Y%m%d%H%M%S'))
			print('assign iteration %d to worker %d' % (i+1,j) )
			workAssignmentTable[i] = i, j
			break

### traverse workAssignmentTable track individual performance
# return the list of iteration that has been assigned to an INDIVIDUAL worker ${j}
workerSchedule = [] # assess by workerSchedule[j]
for j in range(numWorkers):
	workerSchedule += [list(np.where((workAssignmentTable==j)[:,1])[0])]

# utils
def getTimePassed(datetimeObj, refStartTime):
	deltaTInSeconds = (datetimeObj - refStartTime).total_seconds()
	return deltaTInSeconds

### plot
# N = getTimePassed(refStopTime, refStartTime) # number of discretized time
from matplotlib.ticker import MaxNLocator
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24 

# ### regular plot 
# fig = plt.figure()
# for j in range(numWorkers):
# 	plt.plot([0, getTimePassed(refStopTime, refStartTime)], [j,j], linestyle=':', linewidth=2, label='worker %d' % j, marker='')
# 	num_of_works = len(workerSchedule[j]) # for worker ${j}
# 	for i_work in range(num_of_works):
# 		i_iteration = workerSchedule[j][i_work]
# 		startTime = getTimePassed(startTimeObjs[i_iteration], refStartTime)
# 		stopTime  = getTimePassed(stopTimeObjs[i_iteration] , refStartTime)
# 		plt.plot([startTime, stopTime] , [j,j], marker='s', color='k', markersize=10, linestyle='-', linewidth=3)
# 		plt.text(0.5 * (startTime + stopTime), j, 'r%d' % i_iteration, {'color': 'C2', 'fontsize': 18}, verticalalignment="top", horizontalalignment="right", weight='bold')

### hbar plot
# https://stackoverflow.com/questions/16653815/horizontal-stacked-bar-chart-in-matplotlib
# https://stackoverflow.com/questions/21397549/stack-bar-plot-in-matplotlib-and-add-label-to-each-section-and-suggestions
patch_handles = []
# fig = plt.figure(figsize=(10,8))
fig = plt.figure()
ax = fig.add_subplot(111)

label_dicts = {"b": "UCB", "violet": "EI", "orange": "PI", "tomato": "MSE", "g": "idle"} # disable label after the first usage

for j in range(numWorkers):
	num_of_works = len(workerSchedule[j]) # for worker ${j}
	print('len(workerSchedule[%d] = %d' % (j, num_of_works))
	if num_of_works == 0:
		patch_handles.append(ax.barh(j, getTimePassed(refStopTime, refStartTime), left=getTimePassed(refStartTime, refStartTime), color='g', alpha=0.5, label=label_dicts["g"]))
		label_dicts["g"] = "_nolegend_"
		print('Warning: worker %d is completely idle!' % (j))
	for i_work in range(num_of_works):
		i_iteration = workerSchedule[j][i_work]
		# running time (busy)
		startTime = startTimeObjs[i_iteration]
		stopTime  = stopTimeObjs[i_iteration]
		#
		# highlight depending on acquisition function
		# patch_handles.append(ax.barh(j, getTimePassed(stopTime, startTime), left=getTimePassed(startTime, refStartTime), color='r', alpha=0.5, label=label_dicts["r"]))
		# label_dicts["r"] = "_nolegend_"
		if acquisition[i_iteration] == 'UCB':
			patch_handles.append(ax.barh(j, getTimePassed(stopTime, startTime), left=getTimePassed(startTime, refStartTime), color='b', alpha=0.5, label=label_dicts["b"]))
			label_dicts["b"] = "_nolegend_"
		elif acquisition[i_iteration] == 'EI':
			patch_handles.append(ax.barh(j, getTimePassed(stopTime, startTime), left=getTimePassed(startTime, refStartTime), color='violet', alpha=0.5, label=label_dicts["violet"]))
			label_dicts["violet"] = "_nolegend_"
		elif acquisition[i_iteration] == 'PI':
			patch_handles.append(ax.barh(j, getTimePassed(stopTime, startTime), left=getTimePassed(startTime, refStartTime), color='orange', alpha=0.5, label=label_dicts["orange"]))
			label_dicts["orange"] = "_nolegend_"
		elif acquisition[i_iteration] == '-1':
			patch_handles.append(ax.barh(j, getTimePassed(stopTime, startTime), left=getTimePassed(startTime, refStartTime), color='tomato', alpha=0.5, label=label_dicts["tomato"]))
			label_dicts["tomato"] = "_nolegend_"
		#
		patch = patch_handles[-1][0] 
		bl = patch.get_xy()
		x = 0.5*patch.get_width() + bl[0]
		y = 0.5*patch.get_height() + bl[1]
		# ax.text(x, y, 'r%d' % i_iteration, {'color': 'C2', 'fontsize': 18}, verticalalignment="center", horizontalalignment="center", weight='bold')
		# flag idle when waiting for the first job of the local worker
		if i_work == 0:
			patch_handles.append(ax.barh(j, getTimePassed(startTime, refStartTime), left=getTimePassed(refStartTime, refStartTime), color='g', alpha=0.5, label=label_dicts["g"]))
			label_dicts["g"] = "_nolegend_"
		#
		# flag idle when waiting for the last job of the last worker
		if i_work == num_of_works - 1:
			patch_handles.append(ax.barh(j, getTimePassed(refStopTime, stopTime), left=getTimePassed(stopTime, refStartTime), color='g', alpha=0.5, label=label_dicts["g"]))
			label_dicts["g"] = "_nolegend_"
		# waiting time (idle)
		if i_work < num_of_works - 1:
			nextStartTime = startTimeObjs[workerSchedule[j][i_work + 1]] # get next job info for the same worker
			patch_handles.append(ax.barh(j, getTimePassed(nextStartTime, stopTime), left=getTimePassed(stopTime, refStartTime), color='g', alpha=0.5))
			label_dicts["g"] = "_nolegend_"
		else:
			nextStartTime = refStopTime

# plt.legend(fontsize=20)
plt.gca().invert_yaxis() # flip y-axis
ax.set_ylim([j+0.5, -0.5])
ax = fig.gca()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# y_pos = np.arange(numWorkers)
# ax.set_yticks(y_pos)

plt.xlabel('Time (seconds)', fontsize=24)
plt.ylabel('Worker ID', fontsize=24)
plt.title('Dashboard: worker schedule and acquisition\n\n', fontsize=24)
plt.xlim([0, getTimePassed(refStopTime, refStartTime)])

# reorder legends
handles, labels = ax.get_legend_handles_labels()
reorder_index = []
for j in range(len(labels)):
	if labels[j] == 'UCB':
		reorder_index.append(j)
		break
for j in range(len(labels)):
	if labels[j] == 'EI':
		reorder_index.append(j)
		break
for j in range(len(labels)):
	if labels[j] == 'PI':
		reorder_index.append(j)
		break
for j in range(len(labels)):
	if labels[j] == 'MSE':
		reorder_index.append(j)
		break
for j in range(len(labels)):
	if labels[j] == 'idle':
		reorder_index.append(j)
		break

print(reorder_index)
print(handles)
print(labels)

# reorder_index = np.array(reorder_index)
# handles = np.array(handles)
# labels = np.array(labels)

reorder_handles = [handles[i] for i in reorder_index]
reorder_labels  = [labels[i]  for i in reorder_index]

plt.legend(reorder_handles, reorder_labels, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=5, mode="expand", borderaxespad=0., fontsize=24, frameon=False)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=24)
# plt.legend(loc='best', borderaxespad=0., fontsize=24)

plt.show()

