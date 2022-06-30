import numpy as np
import glob, os

# dump batchID.dat

if os.path.isfile('batchSettings.dat'):
	batchSettings = np.loadtxt('batchSettings.dat', dtype=int)
else:
	batchSettings = np.loadtxt('../batchSettings.dat', dtype=int)

if os.path.isfile('modelName.dat'):
	modelName = np.loadtxt('modelName.dat', dtype=str)
else:
	modelName = np.loadtxt('../modelName.dat', dtype=str)

parentPath = os.getcwd() + '/'
folderList = glob.glob(str(modelName) + '_Iter*')

S = np.loadtxt(parentPath + 'S.dat', delimiter=',')
Y = np.loadtxt(parentPath + 'Y.dat', delimiter=',')
F = np.loadtxt(parentPath + 'F.dat', delimiter=',')
C = np.loadtxt(parentPath + 'C.dat', delimiter=',')
B = np.loadtxt(parentPath + 'B.dat', delimiter=',')

batchID_list = np.empty([len(folderList)]) # batchIDList: list of batch ID
complete_list = np.empty([len(folderList)]) # complete_list: list of complete: 0 = incomplete; 1 = complete

# # debug
# print('getBatch.py: folderList = ')
# print(folderList)
# print('getBatch.py: parentPath = %s' % parentPath)

for i in range(len(folderList)):
	# note: there is always a batchID.dat in each folder when it is created
	# print('getBatch.py: i = %d' % i)

	# if there is a batchID.dat then read it; if not then assume (0 = initial sample)
	if os.path.isfile(parentPath + folderList[i] + '/' + 'batchID.dat'):
		# # debug
		# a = np.loadtxt(parentPath + folderList[i] + '/' + 'batchID.dat')
		# print(folderList[i])
		tmpVal = np.loadtxt(parentPath + folderList[i] + '/' + 'batchID.dat') # only load the last element; soft handler to avoid error
		# in case that batchID.dat is an array, then load the last one
		if type(tmpVal) == float:
			batchID_list[i] = tmpVal[-1]
		else:
			batchID_list[i] = tmpVal
	else:
		batchID_list[i] = 0

	# if there is a complete.dat file
	if os.path.isfile(parentPath + folderList[i] + '/' + 'complete.dat'): # only search for incomplete
		# print 'there is a complete.dat in %s' % folderList[i] # debug
		complete_list[i] = np.loadtxt(parentPath + folderList[i] + '/' + 'complete.dat')
	else:
		complete_list[i] = 0 # don't have complete.dat; assume not complete

# unique, counts = np.unique(batchID_list, return_counts=True)
# # print dict(zip(unique, counts)) # debug 

outFile = open('batchID.dat','w')

## only consider incomplete cases, i.e. complete_list == 0
## NOTE: 'if' is implemented IN PRIORITY

if np.count_nonzero(batchID_list[complete_list == 0] == 1) < batchSettings[0]: 
	outFile.write('1\n')
	print('getBatch.py: batchID.dat = 1: next batch: acquisition')
#
elif np.count_nonzero(batchID_list[complete_list == 0] == 2) < batchSettings[1]: 
	outFile.write('2\n')
	print('getBatch.py: batchID.dat = 2: next batch: explore')
#
elif np.count_nonzero(batchID_list[complete_list == 0] == 3) < batchSettings[2]: 
	outFile.write('3\n')
	print('getBatch.py: batchID.dat = 3: next batch: exploreClassif')
#
else:
	outFile.write('4\n')
	print('getBatch.py: batchID.dat = 4: All batches are full.');

outFile.close()
