

# example syntax: python3 computeLossFunction.py  --f='64x64x64' 
# note: without '/'

import os
import numpy as np
import argparse
import os, sys, datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

parser = argparse.ArgumentParser(description='parse folderName as <str> without /')
parser.add_argument("-f", "--folderName", type=str)
args = parser.parse_args()
folderName = args.folderName
folderName = folderName.split('/')[0]

refData = np.loadtxt('../datasets/true_Ta_polycrystal_SS_HCStack_CLC.dat', skiprows=1)
exp_vareps = refData[:,0] + 1.0
exp_sigma  = refData[:,1] * 1e6

# compData = np.loadtxt(os.getcwd() + '/' + folderName + '/postProc/single_phase_equiaxed_' + folderName + '_tension.txt', skiprows=3)

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

StressStrainFile = os.getcwd() + '/' + folderName + '/postProc/stress_strain.log'
numLinesHeader, fieldsList = getMetaInfo(StressStrainFile)
compData = np.loadtxt(StressStrainFile, skiprows=numLinesHeader+1)


# df = pd.DataFrame(compData, columns=['inc','elem','node','ip','grain','1_pos','2_pos','3_pos','1_f','2_f','3_f','4_f','5_f','6_f','7_f','8_f','9_f','1_p','2_p','3_p','4_p','5_p','6_p','7_p','8_p','9_p'])
df = pd.DataFrame(compData, columns=fieldsList)
comp_vareps = [1] + list(df['1_f']) # d[:,1] # strain -- pad original
comp_sigma  = [0] + list(df['1_p']) # d[:,2] # stress -- pad original
_, uniq_idx = np.unique(np.array(comp_vareps), return_index=True)
comp_vareps = np.array(comp_vareps)[uniq_idx]
comp_sigma  = np.array(comp_sigma)[uniq_idx]


### interpolate & compute loss
interp_vareps = np.linspace(1, np.min([np.max(comp_vareps), np.max(exp_vareps)]), 1000)
from scipy.interpolate import interp1d
# get interpolated exp. stress
interpSpline_exp = interp1d(exp_vareps, exp_sigma, kind='cubic', fill_value='extrapolate')
interp_exp_sigma = interpSpline_exp(interp_vareps)
# get interpolated comp. stress
interpSpline_comp = interp1d(comp_vareps, comp_sigma, kind='cubic', fill_value='extrapolate')
interp_comp_sigma = interpSpline_comp(interp_vareps)

import scipy.linalg as sla
import numpy.linalg as nla
loss_nla = nla.norm((interp_exp_sigma - interp_comp_sigma) / 1.e6, ord=2) # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
loss_sla = sla.norm((interp_exp_sigma - interp_comp_sigma) / 1.e6, ord=2) # https://docs.scipy.org/doc/scipy/reference/linalg.html#module-scipy.linalg
print(loss_nla)
print(loss_sla)

plt.figure()
plt.plot(interp_vareps, interp_exp_sigma, 'g^', ms=5, label='interp. exp.')
plt.plot(interp_vareps, interp_comp_sigma, 'mv', ms=5, label='interp. comp.')
plt.plot(exp_vareps, exp_sigma, 'bo', ms=8, label='exp.')
plt.plot(comp_vareps, comp_sigma, 'rx', ms=8, label='comp.')
plt.legend(fontsize=12)
plt.xlabel(r'$\varepsilon$', fontsize=18)
plt.ylabel(r'$\sigma$', fontsize=18)
plt.xlim(left=1)
plt.ylim(bottom=0)
plt.show()

