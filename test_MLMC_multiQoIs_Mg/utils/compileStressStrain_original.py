
import numpy as np
import os, glob
import matplotlib as mpl
import pandas as pd
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator # monotonic: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html


def getTrueStressStrain(StressStrainFile):
	"""
	NOTE: adopted from computeCollocatedStresses.py (the post-processing script)
	"""
	# d = np.loadtxt(StressStrainFile, skiprows=4)
	numLinesHeader, fieldsList = getMetaInfo(StressStrainFile)
	# d = np.loadtxt(StressStrainFile, skiprows=skiprows)
	d = np.loadtxt(StressStrainFile, skiprows=numLinesHeader+1)
	# df = pd.DataFrame(d, columns=['inc','elem','node','ip','grain','1_pos','2_pos','3_pos','1_f','2_f','3_f','4_f','5_f','6_f','7_f','8_f','9_f','1_p','2_p','3_p','4_p','5_p','6_p','7_p','8_p','9_p'])
	df = pd.DataFrame(d, columns=fieldsList)
	# vareps = [1] + list(df['1_f']) # d[:,1]  # strain -- pad original
	# sigma  = [0] + list(df['1_p']) # d[:,2]  # stress -- pad original
	vareps = list(df['Mises(ln(V))'])  # strain -- pad original
	sigma  = list(df['Mises(Cauchy)']) # stress -- pad original
	_, uniq_idx = np.unique(np.array(vareps), return_index=True)
	vareps = np.array(vareps)[uniq_idx]
	sigma  = np.array(sigma)[uniq_idx]
	# x = (vareps - 1)
	x = (vareps)
	y = sigma / 1e6
	return x, y

def getMetaInfo(StressStrainFile):
	"""
	NOTE: adopted from computeCollocatedStresses.py (the post-processing script)
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
	# print('numLinesHeader = ', numLinesHeader)
	# print('fieldsList = ', fieldsList)
	return numLinesHeader, fieldsList


plt.figure()

dimCellList = [2, 4, 8, 16, 32]
colorList = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
alphaList = [0.4, 0.5, 0.6, 0.7, 0.8]
validLegendFileList = ['', '', '', '', ''] # collect last valid file for legend plotting purpose

for i in range(len(dimCellList)):
	dimCell = dimCellList[i]
	num_obs = len(glob.glob('%sx%sx%s*stress_strain.log' % (dimCell, dimCell, dimCell)))
	print(f"Found {num_obs} observations at {dimCell}x{dimCell}x{dimCell}")

	for fileName in glob.glob('%sx%sx%s*stress_strain.log' % (dimCell, dimCell, dimCell)):
		strain, stress = getTrueStressStrain(fileName)

		#
		if np.any(stress > 3e2) or np.any(np.diff(stress) < 0) or np.max(strain) < 0.1:
			print(f"{fileName} is not usable.")
		else:
			interp_strain = np.linspace(0, 0.12, num=201)
			collocated_strain = np.linspace(0, 0.10, num=11)
			# splineInterp = interp1d(strain, stress, kind='cubic', fill_value='extrapolate')
			splineInterp = PchipInterpolator(strain, stress)
			# splineInterp = interp1d(strain, stress, kind='cubic', fill_value=('NaN','NaN'))
			interp_stress = splineInterp(interp_strain)
			collocated_stress = splineInterp(collocated_strain)
			plt.plot(interp_strain, interp_stress, color=colorList[i], alpha=alphaList[i], linewidth=i)
			plt.plot(collocated_strain, collocated_stress, 'rs', linestyle='None')
			plt.plot(strain, stress, 'bo', linestyle='None')
			# plt.plot(strain, stress, color=, alpha=alphaList[i], marker='o', linewidth=i)
			validLegendFileList[i] = fileName

# plot for legend
plt.plot(strain, stress, 'bo', linestyle='None', label='observations')
plt.plot(collocated_strain, collocated_stress, 'rs', linestyle='None', label='collocated observations')
# print(validLegendFileList)
for i in range(len(dimCellList)):
	dimCell = dimCellList[i]
	fileName = validLegendFileList[i]
	strain, stress = getTrueStressStrain(fileName)
	plt.plot(strain, stress, marker='None', color=colorList[i], label='%sx%sx%s' % (dimCell, dimCell, dimCell))


# custom plot
plt.xlim(left=0, right=0.11)
plt.ylim(bottom=0)
plt.legend(loc='best', fontsize=24, frameon=False, markerscale=2)
plt.xlabel(r'$\varepsilon_{vM}$ [-]', fontsize=24)
plt.ylabel(r'$\sigma_{vM}$ [MPa]' , fontsize=24)
plt.title(r'Compilation of equivalent $\sigma-\varepsilon$ curves for Mg', fontsize=24)
plt.show()

