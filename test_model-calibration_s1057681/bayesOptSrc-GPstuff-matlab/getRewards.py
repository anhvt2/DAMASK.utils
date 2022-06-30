
import numpy as np
import glob, os
from natsort import natsorted, ns

"""
# see Hoffman, Matthew D., Eric Brochu, and Nando de Freitas. "Portfolio Allocation for Bayesian Optimization." UAI. 2011.
# http://mlg.eng.cam.ac.uk/hoffmanm/papers/hoffman:2011.pdf

1. read rewards.dat
2. dump R.dat
	line 1: UCB; line 2: EI; line 3: PI.
3a. dump rewardsList: list of folders that contain rewards.dat
3b.	remove file after usage


"""

### DEPRECATED
"""
if os.path.isfile('batchSettings.dat'):
	batchSettings = np.loadtxt('batchSettings.dat', dtype=int)
else:
	batchSettings = np.loadtxt('../batchSettings.dat', dtype=int)

if os.path.isfile('modelName.dat'):
	modelName = np.loadtxt('modelName.dat', dtype=str)
else:
	modelName = np.loadtxt('../modelName.dat', dtype=str)
"""

### check from local folder
batchSettings = np.loadtxt('../batchSettings.dat', dtype=int)
modelName = np.loadtxt('../modelName.dat', dtype=str)
modelName = str(modelName)

### read inputs/outputs
S = np.loadtxt('../S.dat', delimiter=',')
Y = np.loadtxt('../Y.dat', delimiter=',')
F = np.loadtxt('../F.dat', delimiter=',')
C = np.loadtxt('../C.dat', delimiter=',')
B = np.loadtxt('../B.dat', delimiter=',')

currentPath = os.getcwd() + '/'
parentPath = currentPath + '/../'

### DEPRECATED
"""
os.system('rm -fv rewardsList.dat')
os.chdir(parentPath)
os.system("find . -name 'rewards.dat' | cut -d'/' -f2 > rewardsList.dat")
os.chdir(currentPath)
os.system('mv -v ../rewardsList.dat .')

# folderList = np.loadtxt('rewardsList.dat', dtype=str) 
"""

# get folder name only
folders = natsorted(glob.glob('../' + modelName + '_Iter*'))
for i in range(len(folders)):
	folder = folders[i]
	folder = folder.split('/')[-1]
	folders[i] = folder

### debug
# print('getRewards.py: print folderList')
# print(folderList)
# print('getRewards.py: parentPath = %s' % parentPath)
# print('getRewards.py: currentPath = %s' % currentPath)


### initialize
d = np.loadtxt('../' + modelName + '_Iter1/projectedInput.dat', delimiter=',').shape[0] # read dimensionality of the problem
# xs = np.empty([len(folders), d]) # do not attempt to read input.dat, particularly in the same folder where it has not yet existed
batchs = np.empty([len(folders)]) # batchIDs: s of batch ID
completes = np.empty([len(folders)]) # completes: s of complete: 0 = incomplete; 1 = complete
rewards = np.empty([len(folders)]) # rewards: output.dat
feasibles = np.empty([len(folders)]) # feasibles: 
# acquisitions = np.empty([len(folderList)], dtype=str) # acquisitions
acquisitions = [0] * len(folders) 
ys = np.empty([len(folders)])
gp_means = np.empty([len(folders)])


### read history
# adopt/inspired from getBatch.py
# implement for cases without acquisitionScheme.dat
for i in range(len(folders)):
	# print('getRewards.py: %s' % parentPath + folders[i] + '/' + 'rewards.dat') # debug 
	if folders[i].split('_')[0] == modelName: # implement safeguard
		# read info from those folders
		activeFolder = parentPath + folders[i] + '/'
		if os.path.exists(activeFolder + 'complete.dat'):
			completes[i] = np.loadtxt(activeFolder + 'complete.dat')
		else:
			completes[i] = 0
		# rewards[i] = np.loadtxt(activeFolder + 'rewards.dat') # deprecated to avoid confusion
		if os.path.exists(activeFolder + 'feasible.dat'):
			feasibles[i] = np.loadtxt(activeFolder + 'feasible.dat')
		else:
			feasibles[i] = 0
		#
		if os.path.exists(activeFolder + 'output.dat'):
			ys[i] = np.loadtxt(activeFolder + 'output.dat')
		else:
			ys[i] = np.nan
		#
		if os.path.exists(activeFolder + 'gpPredictions.dat'):
			gp_means[i] = np.loadtxt(activeFolder + 'gpPredictions.dat')[0]
		else:
			gp_means[i] = 0 # assign 0 if undefined
		#
		if os.path.isfile(activeFolder + '/' + 'acquisitionScheme.dat'):
			acquisitions[i] = str(np.loadtxt(activeFolder + 'acquisitionScheme.dat', dtype=str))
			print('getRewards.py: acquisitions[%d] = %s' % (i, acquisitions[i]))
		else:
			acquisitions[i] = 0



### assign rewards
# always assume maximization problem is being solved

# get numInitPoint
mainprogFile = open('../mainprog_benchR.m')
mainprogText = mainprogFile.readlines()
mainprogFile.close()
for line in mainprogText:
	if 'numInitPoint =' in line:
		# print(line)
		numInitPoint = line

numInitPoint = numInitPoint.split('=')[1].split(';')[0]
numInitPoint = int(numInitPoint)

# find what iteration corresponds to the best-so-far samples
bests = [] # list of bests
tmpMax = ys[0]
acquisitionBests = []
for i in range(numInitPoint, len(ys)):
	if tmpMax < ys[i] and ys[i] != 0: # do not pick up infeasible points
		bests += [i] # add iteration i into list of bests
		acquisitionBests += [acquisitions[i]]
		tmpMax = ys[i]

### count rewards

## increase rewards by 1
# rewardsUCB_array = np.where(np.array(acquisitionBests) == 'UCB')
# rewardsEI_array  = np.where(np.array(acquisitionBests) == 'EI')
# rewardsPI_array  = np.where(np.array(acquisitionBests) == 'PI')
# # rewardsMC  = np.where(np.array(acquisitionBests) == 'MC')
# rewardsUCB = len(rewardsUCB_array[0]) # count hits for best-so-far acquisition functions
# rewardsEI  = len(rewardsEI_array[0]) # count hits for best-so-far acquisition functions
# rewardsPI  = len(rewardsPI_array[0]) # count hits for best-so-far acquisition functions
# # rewardsMC  = rewardsMC[0]

## increase rewards by the GP posterior mean
rewardsUCB_array = gp_means[np.where(np.array(acquisitions) == 'UCB')]
rewardsEI_array  = gp_means[np.where(np.array(acquisitions) == 'EI')]
rewardsPI_array  = gp_means[np.where(np.array(acquisitions) == 'PI')]
if gp_means.max() != gp_means.min():
	rewardsUCB = rewardsUCB_array.sum() / (gp_means.max() - gp_means.min())
	rewardsEI  = rewardsEI_array.sum() / (gp_means.max() - gp_means.min())
	rewardsPI  = rewardsPI_array.sum() / (gp_means.max() - gp_means.min())
else:
	rewardsUCB = 0.0
	rewardsEI = 0.0
	rewardsPI = 0.0


### write rewards
R = open('R.dat','w')

R.write('%.2f\n' % rewardsUCB) # line 1: UCB
R.write('%.2f\n' % rewardsEI)  # line 2: EI
R.write('%.2f\n' % rewardsPI)  # line 3: PI
# R.write('%.2f\n' % len(rewardsMC))  # line 4: MC
R. close()

# os.system('rm -v rewardsList.dat') # deprecated

print('\ngetRewards.py: cat R.dat')
os.system('cat R.dat')
print('\n')


