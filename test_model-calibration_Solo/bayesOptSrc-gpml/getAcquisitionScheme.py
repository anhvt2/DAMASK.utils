import numpy as np 
import os, glob

"""
# see Hoffman, Matthew D., Eric Brochu, and Nando de Freitas. "Portfolio Allocation for Bayesian Optimization." UAI. 2011.
# http://mlg.eng.cam.ac.uk/hoffmanm/papers/hoffman:2011.pdf
# https://arxiv.org/pdf/1009.5419

# dump to acquisitionScheme.dat

REQUIREMENTS:
	* must be numerically stable, i.e. np.exp(1000) / np.exp(1005) = np.exp(-5); in practice it is np.nan
	* np.exp(709) = 8.218407461554972e+307 is the cut-off
"""

# # potential bug: too many rewards.dat file; switch to acquisitionScheme.dat instead
# import subprocess
# proc = subprocess.Popen(['/bin/bash'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
# T = proc.communicate("find . -name 'acquisitionScheme.dat' | wc -l") 
# T = int(T[0].replace('\n',''))

# currentPath = os.getcwd() + '/'
# parentPath = currentPath + '/../'

## DEPRECATED
"""
os.system('rm -fv rewardsList.dat')
os.chdir(parentPath)
os.system("find . -name 'acquisitionScheme.dat' | wc -l > T.dat")
os.chdir(currentPath)
os.system('mv -v ../T.dat .')
"""

# T: total number of acquisitionScheme.dat files or rough estimation of total folders/iterations
# T = np.loadtxt('T.dat')

batchSettings = np.loadtxt('../batchSettings.dat', dtype=int)
modelName = np.loadtxt('../modelName.dat', dtype=str)
modelName = str(modelName)

## read inputs/outputs
S = np.loadtxt('../S.dat', delimiter=',')
Y = np.loadtxt('../Y.dat', delimiter=',')
F = np.loadtxt('../F.dat', delimiter=',')
C = np.loadtxt('../C.dat', delimiter=',')
B = np.loadtxt('../B.dat', delimiter=',')

currentPath = os.getcwd() + '/'
parentPath = currentPath + '/../'

# get folder name only
folders = glob.glob('../' + modelName + '_Iter*')
for i in range(len(folders)):
	folder = folders[i]
	folder = folder.split('/')[-1]
	folders[i] = folder

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

T = len(folders) - (numInitPoint - 1) # avoid T = 0 scenario

## sample acquisition function
k = 3 # number of acquisition functions

eta = np.sqrt(8 * np.log(k) / T) # lemma 1, GP-Hed
R = np.loadtxt('R.dat');

## augment numerical stability in sampling
# R -= np.min(R)
R -= np.max(R)
th = 50 # cut-off at np.exp(th)
for i in range(len(R)):
	if R[i] > th:
		R[i] = th
		print('getAcquisitionScheme.py: warning: numerical stabilizer activated!')

# compute pmf of UCB, EI, and PI
pmf = np.exp(eta * R) / np.sum(np.exp(eta * R))

u = np.random.uniform()

f = open('acquisitionScheme.dat','w')
# NOTE: don't write '\n' char to the acquisitionScheme.dat
if u <= pmf[0]:
	f.write('UCB');
	print('getAcquisitionScheme.py: UCB is selected')
elif u > pmf[0] and u < pmf[0] + pmf[1]:
	f.write('EI')
	print('getAcquisitionScheme.py: EI is selected')
elif u > pmf[0] + pmf[1] and u < pmf[0] + pmf[1] + pmf[2]:
	f.write('PI')
	print('getAcquisitionScheme.py: PI is selected')
# elif u > pmf[0] + pmf[1] + pmf[2] and u < pmf[0] + pmf[1] + pmf[2] + pmf[3]:
# 	f.write('MC')
# 	print('getAcquisitionScheme.py: MC is selected')

f.close()

