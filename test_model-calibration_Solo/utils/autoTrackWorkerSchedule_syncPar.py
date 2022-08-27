
import numpy as np
import os, time, glob, sys, natsort, datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

### read input files

startTime = np.loadtxt('postproc.startTime.dat', dtype=str)
stopTime = np.loadtxt('postproc.stopTime.dat', dtype=str)
batchSettings = np.loadtxt('batchSettings.dat')


n = len(startTime)
numWorkers = int(batchSettings.sum()) # get total number of workers
modelName = np.loadtxt('modelName.dat', dtype=str)
folders = glob.glob('%s_Iter*/' % modelName)
for i in range(len(folders) - 1, -1, -1): # going reverse
	# remove all folders with pattern "${modelName}_Iter*_{Exploit,Explore,ExploreClassif}_Batch*"
	if 'Batch' in folders[i]:
		del folders[i]

numInitPoint = len(folders) # whatever is leftover

### convert and process datetime object
startTimeObjs = []
stopTimeObjs = []
elapsedTimes = [] 

for i in range(n):
	startTimeObjs += [datetime.datetime.strptime(startTime[i][0] + ' ' + startTime[i][1], '%Y-%m-%d %H:%M:%S')]
	stopTimeObjs += [datetime.datetime.strptime(stopTime[i][0] + ' ' + stopTime[i][1], '%Y-%m-%d %H:%M:%S')]
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
	workAssignmentTable[i] = i, np.nan

for i in range(numInitPoint, n): # exclude initial points
	j = (i - numInitPoint) % numWorkers
	availableTime[j] = int(stopTimeObjs[i].strftime('%Y%m%d%H%M%S'))
	currentStartTimeWorker[j] = int(startTimeObjs[i].strftime('%Y%m%d%H%M%S'))
	currentStopTimeWorker[j]  = int(stopTimeObjs[i].strftime('%Y%m%d%H%M%S'))
	print('assign iteration %d to worker %d' % (i+1,j) )
	workAssignmentTable[i] = i, j


### traverse workAssignmentTable track individual performance
# return the list of iteration that has been assigned to an INDIVIDUAL worker ${j}
workerSchedule = [] # assess by workerSchedule[j]
for j in range(numWorkers):
	workerSchedule += [list(np.where((workAssignmentTable==j)[:,1])[0])]

# utils
def getTimePassed(anyStopTimeObj, anyStartTimeObj):
	deltaTInSeconds = (anyStopTimeObj - anyStartTimeObj).total_seconds()
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
# fig = plt.figure()
fig = plt.figure(num=None, figsize=(20, 11.3), dpi=300, facecolor='w', edgecolor='k') # screen size

ax = fig.add_subplot(111)
label_dicts = {"r": "busy", "g": "idle"} # disable label after the first usage

for j in range(numWorkers):
	num_of_works = len(workerSchedule[j]) # for worker ${j}
	for i_work in range(num_of_works):
		i_iteration = workerSchedule[j][i_work]
		# running time (busy)
		startTime = startTimeObjs[i_iteration]
		stopTime  = stopTimeObjs[i_iteration]
		patch_handles.append(ax.barh(j, getTimePassed(stopTime, startTime), height=0.95, left=getTimePassed(startTime, refStartTime), color='r', alpha=0.5, label=label_dicts["r"]))
		label_dicts["r"] = "_nolegend_"
		patch = patch_handles[-1][0] 
		bl = patch.get_xy()
		x = 0.5*patch.get_width() + bl[0]
		y = 0.5*patch.get_height() + bl[1]
		# ax.text(x, y, 'r%d' % i_iteration, {'color': 'C2', 'fontsize': 18}, verticalalignment="center", horizontalalignment="center", weight='bold')
		# waiting time (idle)
		if i_work < num_of_works - 1:
			nextStartTime = startTimeObjs[workerSchedule[j][i_work + 1]] # get next job info for the same worker
		else:
			nextStartTime = refStopTime
		patch_handles.append(ax.barh(j, getTimePassed(nextStartTime, stopTime), height=0.95, left=getTimePassed(stopTime, refStartTime), color='g', alpha=0.5, label=label_dicts["g"]))
		label_dicts["g"] = "_nolegend_"

numOfBatches = len(workerSchedule[0]) # or any workerSchedule[j]
for i in range(numOfBatches):
	# when the first worker in the batch starts
	startTime = getTimePassed(startTimeObjs[workerSchedule[0][i]] , refStartTime) 
	# when the "last" worker in the batch finishes -- but need to find the "last" first
	stopTime = 0
	for j in range(numWorkers):
		if getTimePassed( stopTimeObjs[workerSchedule[j][i]], refStartTime) > stopTime:
			stopTime = getTimePassed( stopTimeObjs[workerSchedule[j][i]], refStartTime)
	plt.plot([startTime, startTime], [-1, j+1], linestyle='--', linewidth=2, color='k')
	# plt.plot([stopTime , stopTime ], [-1, j+1], linestyle='--', linewidth=2, color='k')


# plt.legend(fontsize=20)
plt.gca().invert_yaxis() # flip y-axis
ax = fig.gca()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylim([j+0.5, -0.5])
# y_pos = np.arange(numWorkers)
# ax.set_yticks(y_pos)

plt.xlabel('Time (seconds)', fontsize=24)
plt.ylabel('Worker ID', fontsize=24)
plt.title('Dashboard: worker schedule\n\n', fontsize=24)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=2, mode="expand", borderaxespad=0., fontsize=24, frameon=False)
# plt.xlim([0, 1000])

# plt.show()
plt.savefig('trackWorkerSchedule_' + os.getcwd().split('/')[-1] + '.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)



