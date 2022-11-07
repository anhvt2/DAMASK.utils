

# example syntax: python3 computeLossFunction.py  --f='64x64x64' 
# note: without '/'

import os
import numpy as np
import argparse
import os, sys, datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
currentPath = os.getcwd()

parser = argparse.ArgumentParser(description='parse meshFolderName as <str> without /')
parser.add_argument("-f", "--f", type=str)
parser.add_argument("-p", "--plot", type=bool, default=1) # default: no plot
args = parser.parse_args()
meshFolderName = args.f
meshFolderName = meshFolderName.split('/')[0]
plotDebug = args.plot # debug option for plotting comparison exp. vs. comp.

refData = np.loadtxt('../datasets/true_SS304L_EngStress_EngStrain_exp_4A1.dat', skiprows=1)
exp_vareps = refData[:,0] # start at vareps = 0
exp_sigma  = refData[:,1] * 1e6

# compData = np.loadtxt(os.getcwd() + '/' + meshFolderName + '/postProc/single_phase_equiaxed_' + meshFolderName + '_tension.txt', skiprows=3)

print('\n')

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

StressStrainFile = os.getcwd() + '/' + meshFolderName + '/postProc/stress_strain.log'
numLinesHeader, fieldsList = getMetaInfo(StressStrainFile)
compData = np.loadtxt(StressStrainFile, skiprows=numLinesHeader+1)
# print(compData)
# print(np.isnan(compData[:,-1]))
compData = compData[~np.isnan(compData[:,-1]),:]
compData = compData[~np.isinf(compData[:,-1]),:]

# df = pd.DataFrame(compData, columns=['inc','elem','node','ip','grain','1_pos','2_pos','3_pos','1_f','2_f','3_f','4_f','5_f','6_f','7_f','8_f','9_f','1_p','2_p','3_p','4_p','5_p','6_p','7_p','8_p','9_p'])
df = pd.DataFrame(compData, columns=fieldsList)
# comp_vareps = [1] + list(df['1_f']) # d[:,1] # strain -- pad original
# comp_sigma  = [0] + list(df['1_p']) # d[:,2] # stress -- pad original
comp_vareps = list(df['Mises(ln(V))'])  # strain -- pad original
comp_sigma  = list(df['Mises(Cauchy)']) # stress -- pad original
_, uniq_idx = np.unique(np.array(comp_vareps), return_index=True)
# comp_vareps = np.array(comp_vareps)[uniq_idx] - 1 # start at vareps = 0: ['1_f', '1_p']
comp_vareps = np.array(comp_vareps)[uniq_idx] # start at vareps = 0: ['Mises(ln(V))', 'Mises(Cauchy)']
comp_sigma  = np.array(comp_sigma)[uniq_idx]


### interpolate & compute loss
max_interp_vareps = np.min([np.max(comp_vareps), np.max(exp_vareps)]) 
""" Explanation: 
	Even though with the same loading conditions, sometimes CPFEM ends with much shorter strain. 
	To encourage CPFEM to go further for minimizing the loss between comp. and exp., 
	we penalize the loss function based on max_interp_vareps
"""
interp_vareps = np.linspace(0, max_interp_vareps, 1000) # start at vareps = 0
from scipy.interpolate import interp1d
# get interpolated exp. stress
interpSpline_exp = interp1d(exp_vareps, exp_sigma, kind='linear', fill_value='extrapolate')
interp_exp_sigma = interpSpline_exp(interp_vareps)
# get interpolated comp. stress
interpSpline_comp = interp1d(comp_vareps, comp_sigma, kind='cubic', fill_value='extrapolate')
interp_comp_sigma = interpSpline_comp(interp_vareps)

### plot -- debug
if plotDebug:
	# plt.figure()
	# plt.figure(num=None, figsize=(20, 11.3), dpi=300, facecolor='w', edgecolor='k') # screen size
	mpl.use('Agg')
	plt.figure(num=None, figsize=(20, 11.3), dpi=300)
	plt.plot(interp_vareps, interp_exp_sigma, 'g^', ms=5, label='interp. exp.')
	plt.plot(interp_vareps, interp_comp_sigma, 'mv', ms=5, label='interp. comp.')
	plt.plot(exp_vareps, exp_sigma, 'bo', ms=8, label='exp.')
	plt.plot(comp_vareps, comp_sigma, 'rx', ms=8, label='comp.')
	plt.legend(fontsize=12, markerscale=2)
	plt.xlabel(r'$\varepsilon$', fontsize=18)
	plt.ylabel(r'$\sigma$', fontsize=18)
	plt.xlim(left=0)
	plt.ylim(bottom=0)
	currentFolderName = os.getcwd().split('/')[-1]
	plt.savefig('compareExpComp_%s_%s.png' % (currentFolderName, meshFolderName), dpi='figure', format=None, metadata=None,
        bbox_inches=None, pad_inches=0.1, 
        # facecolor='auto', edgecolor='auto',
        backend=None)
	# plt.show()

### compute loss function

import scipy.linalg as sla
import numpy.linalg as nla
# loss_nla = nla.norm((interp_exp_sigma - interp_comp_sigma) / 1.e6, ord=2) # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
# loss_sla = sla.norm((interp_exp_sigma - interp_comp_sigma) / 1.e6, ord=2) # https://docs.scipy.org/doc/scipy/reference/linalg.html#module-scipy.linalg
# print(loss_nla)
# print(loss_sla)
scaled_l2_loss = np.sqrt(np.trapz((interp_exp_sigma - interp_comp_sigma)**2, x=interp_vareps)) / 1e6
scaled_l2_d1_loss = np.sqrt(np.trapz((np.gradient(interp_exp_sigma) - np.gradient(interp_comp_sigma))**2, x=interp_vareps)) / 1e4 

# negative_loss = - (scaled_l2_loss + scaled_l2_d1_loss) / 1e2
# negative_loss = - scaled_l2_loss / 1e2
# negative_loss = - scaled_l2_loss / 1e2 / max_interp_vareps # normalized by max_interp_vareps
negative_loss = - (scaled_l2_loss + scaled_l2_d1_loss) / 1e2 / max_interp_vareps


### write output to output.dat
print('\nWriting output.dat in folder: %s \n' % meshFolderName)
f = open(currentPath + '/' + meshFolderName + '/' + 'output.dat', 'w') # can be 'r', 'w', 'a', 'r+'
# f.write('%.8e\n' % (loss_nla / 1e3)) # example: 20097.859541889356 -- scale by a factor of 1e3
# f.write('%.8e\n' % (loss_nla / 1e3 / max_interp_vareps)) # example: 20097.859541889356 -- scale by a factor of 1e3
# f.write('%.8e\n' % (- np.log(loss_nla / max_interp_vareps))) # example: 20097.859541889356 -- scale by a factor of 1e3
f.write('%.8e\n' % (negative_loss)) # example: 20097.859541889356 -- scale by a factor of 1e3
i = np.loadtxt('input.dat', delimiter=',')
print('%s' % os.getcwd().split('/')[-1])
print('i = ', i)
print('l2 loss = ', scaled_l2_loss)
print('regularized l2 d1 loss = ', scaled_l2_d1_loss)
print('negative total loss = ', negative_loss)
print('Finished writing output.dat in folder: %s' % meshFolderName)
f.close()

### decide whether the run is feasible or not feasible
feasible = 0 # default = 0 unless pass one of these feasible criteria
if comp_vareps.max() > 0.1 and comp_sigma.max() < 1e12:
	feasible = 1


### write feasible to feasible.dat
print('Writing feasible.dat in folder: %s' % meshFolderName)
f = open(currentPath + '/' + meshFolderName + '/' + 'feasible.dat', 'w') # can be 'r', 'w', 'a', 'r+'
f.write('%d\n' % feasible)
print('Finished writing feasible.dat in folder: %s' % meshFolderName)
f.close()
print('\n')


