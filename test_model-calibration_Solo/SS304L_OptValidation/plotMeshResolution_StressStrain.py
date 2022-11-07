
# mostly adopted from <main>/plotStressStrain.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys, datetime
# import argparse
import pandas as pd
import scipy
from scipy.interpolate import interp1d


def getMetaInfo(StressStrainFile):
	"""
	return 
	(1) number of lines for headers 
	(2) list of outputs for pandas dataframe
	"""
	fileHandler = open(StressStrainFile)
	txtInStressStrainFile = fileHandler.readlines()
	fileHandler.close()
	numLinesHeader = int(txtInStressStrainFile[0].split('\t')[0])
	fieldsList = txtInStressStrainFile[numLinesHeader].split('\t')
	for i in range(len(fieldsList)):
		fieldsList[i] = fieldsList[i].replace('\n', '')
	print('numLinesHeader = ', numLinesHeader)
	print('fieldsList = ', fieldsList)
	return numLinesHeader, fieldsList

def readLoadFile(LoadFile):
	load_data = np.loadtxt(LoadFile, dtype=str)
	n_fields = len(load_data)
	# assume uniaxial:
	for i in range(n_fields):
		if load_data[i] == 'Fdot' or load_data[i] == 'fdot':
			print('Found *Fdot*!')
			Fdot11 = float(load_data[i+1])
		if load_data[i] == 'time':
			print('Found *totalTime*!')
			totalTime = float(load_data[i+1])
		if load_data[i] == 'incs':
			print('Found *totalIncrement*!')
			totalIncrement = float(load_data[i+1])
		if load_data[i] == 'freq':
			print('Found *freq*!')
			freq = float(load_data[i+1])
	return Fdot11, totalTime, totalIncrement


def getTrueStressStrain(StressStrainFile):
	numLinesHeader, fieldsList = getMetaInfo(StressStrainFile)
	d = np.loadtxt(StressStrainFile, skiprows=numLinesHeader+1)
	df = pd.DataFrame(d, columns=fieldsList)
	vareps = [1] + list(df['1_f']) # d[:,1] # strain -- pad original
	sigma  = [0] + list(df['1_p']) # d[:,2] # stress -- pad original
	_, uniq_idx = np.unique(np.array(vareps), return_index=True)
	vareps = np.array(vareps)[uniq_idx]
	sigma  = np.array(sigma)[uniq_idx]
	x = (vareps - 1)
	y = sigma / 1e6
	return x, y

def getInterpStressStrain(StressStrainFile):
	x, y = getTrueStressStrain(StressStrainFile)
	splineInterp = interp1d(x, y, kind='cubic', fill_value='extrapolate')
	interp_vareps = np.linspace(x.min(), x.max(), num=100)
	interp_sigma = splineInterp(interp_vareps)
	return interp_vareps, interp_sigma

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
fig = plt.figure()
ax = fig.add_subplot(111)

LoadFile = 'tension.load'

# StressStrainFile2 = 'sve1_2x2x2/postProc/stress_strain.log'
# interp_vareps2, interp_sigma2 = getInterpStressStrain(StressStrainFile2)

meshIndexList = np.array([2,4,8,10,16,20])
markerList = ['^', 'v', 'x', 'd', 's', 'o']
colorList = ['k', 'm', 'g', 'purple', 'r', 'b']
array_interp_vareps = np.zeros([100, meshIndexList.shape[0]]) # 100 depends on discretization of getInterpStressStrain()
array_interp_sigma  = np.zeros([100, meshIndexList.shape[0]]) # 100 depends on discretization of getInterpStressStrain()

for i in range(len(meshIndexList)): # mesh-resolution index
	StressStrainFile = 'sve1_%dx%dx%d/postProc/stress_strain.log' % (meshIndexList[i], meshIndexList[i], meshIndexList[i])
	interp_vareps, interp_sigma  = getInterpStressStrain(StressStrainFile)
	array_interp_vareps[:,i] = interp_vareps
	array_interp_sigma[:,i] = interp_sigma
	ax.plot(interp_vareps, interp_sigma, linestyle=':', c=colorList[i], marker=markerList[i], label=r'%d$^3$' % (meshIndexList[i]))


# x, y = getTrueStressStrain(StressStrainFile)
# ax.plot(x, y, c='b', marker='o', linestyle=':', markersize=6)


# ax.plot(interp_vareps, interp_sigma, c='r', marker='^', linestyle='-', markersize=6)
# plt.legend(['true', 'cubic'])
# plt.legend(loc='best', markerscale=3, fontsize=36)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, markerscale=3, fontsize=60, frameon=False)

plt.xlabel(r'$\varepsilon$ [-]', fontsize=30)
plt.ylabel(r'$\sigma$ [MPa]', fontsize=30)
plt.title(r'Mesh-sensitivity effect on homogenized $\sigma-\varepsilon$ curves (spectral CPFEM)', fontsize=24)

plt.xlim(left=0)
plt.ylim(bottom=0)
plt.show()

# ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

# parentFolderName = os.getcwd().split('/')[-4:-1]
# plt.title('%s' % parentFolderName, fontsize=24)

# plt.show()

