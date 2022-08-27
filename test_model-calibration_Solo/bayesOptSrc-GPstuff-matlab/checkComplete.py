# check forensic for simulation termination 
# Ubuntu < 16.04: gnome-terminal -x timeout 6h bash -c "cas64 < dumInp.dat; echo 1 > complete.dat; python ../getOutput.py" > log.timeout
# Ubuntu > 16.04: gnome-terminal -- timeout 6h bash -c "cas64 < dumInp.dat; echo 1 > complete.dat; python ../getOutput.py" > log.timeout

import numpy as np
import glob, os
import datetime

# check for complete.dat that has 0 value
# also check for scripts that are terminated by deadline

modelName = np.loadtxt('modelName.dat', dtype=str)

currentPath = os.getcwd()
if currentPath.split('/')[-1] == 'bayesSrc':
	parentPath = os.getcwd() + '/../'
else:
	parentPath = os.getcwd() + '/'

os.chdir(parentPath)
print('checkComplete.py: currently in parentPath = %s' % parentPath)

# import subprocess
# proc = subprocess.Popen(['/bin/bash'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
# folderList = proc.communicate("find . -name 'complete.dat' | cut -d'/' -f2") 

# # cleanse
# folderList = np.array(str(folderList[0]).replace('\\n',',').replace('\n',',').split(','))
# folderList = np.delete(folderList, np.where(folderList == '') )
# completeList = np.empty([len(folderList)])

print('checkComplete.py: currently in %s' % os.getcwd())
os.system("find . -name {} | cut -d{} -f2 > log.folderList".format('complete.dat','/'))
folderList = np.loadtxt('log.folderList', dtype=str)
completeList = np.zeros(folderList.shape)

# forensic check if the binary has been terminated
for i in range(len(folderList)):
	# adopt from getBatch.py
	if os.path.isfile(parentPath + folderList[i] + '/' + 'complete.dat'): # only search for incomplete
		# print 'there is a complete.dat in %s' % folderList[i] # debug
		completeList[i] = np.loadtxt(parentPath + folderList[i] + '/' + 'complete.dat')
		print('checkComplete.py: there is a complete.dat of value %d in %s' % (completeList[i], folderList[i]))
	else:
		print('checkComplete.py: there is no complete.dat in %s' % folderList[i])
		completeList[i] = 0 # don't have complete.dat; assume not complete

# forensic check:  
# look for case that complete.dat = 0, but has been finished and didn't have chance to update complete.dat
# how to determine if the case has been finished
# look for waitTime in mainprog.m or the allocated time in qsub.*.pace 
# then increase a little bit to allow for safety margin
# waitTime = 10 # unit: hours; import this option from query.sh/qsub.*.pace and 
waitTime = np.loadtxt('waitTime.dat') # unit: hours; import this option from query.sh/qsub.*.pace and 

# flip
for i in range(len(folderList)):
	# datetime.datetime.fromtimestamp(os.path.getmtime(os.getcwd() + '/log.timeout'))
	# datetime.datetime.now()
	if completeList[i] == 0 and os.path.isfile(parentPath + folderList[i] + '/' + '/log.timeout'):
		c = datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getmtime(parentPath + folderList[i] + '/' + '/log.timeout'))
		if  c.total_seconds() > waitTime * 3600:
			np.savetxt(parentPath + folderList[i] + '/' + 'complete.dat', [1], fmt='%d')
			np.savetxt(parentPath + folderList[i] + '/' + 'output.dat', [0], fmt='%d')
			np.savetxt(parentPath + folderList[i] + '/' + 'feasible.dat', [0], fmt='%d')
