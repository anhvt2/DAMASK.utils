
import numpy as np
import matplotlib.pyplot as plt
import os, glob, sys

from natsort import natsorted, ns

### user-defined functions

def removeNaNStrainStress(strain,stress):
	m, n = stress.shape
	removeIndices_stress = np.argwhere(np.isnan(stress.ravel()))
	removeIndices_strain = np.argwhere(np.isnan(strain.ravel()))
	removeIndices = np.unique(np.vstack((removeIndices_strain, removeIndices_stress)))
	cleanIndices = np.setdiff1d(range(n), removeIndices)
	return strain[:, cleanIndices], stress[:, cleanIndices]

def removeInfStrainStress(strain,stress):
	m, n = stress.shape
	removeIndices  = np.argwhere(np.isinf(stress.ravel()))
	removeIndices2 = np.argwhere(np.isinf(strain.ravel()))
	cleanIndices = np.setdiff1d(range(n), removeIndices)
	cleanIndices = np.setdiff1d(range(n), removeIndices2)
	return strain[:, cleanIndices], stress[:, cleanIndices]

def removeNonsenseStrain(strain, stress):
	m, n = stress.shape
	removeIndices = []
	for i in range(strain.shape[1]):
		# print(strain[0,i])
		# print(strain.shape)
		if strain[0,i] > 10 or strain[0,i] < 0 or stress[0,i] > 1e15 or stress[0,i] < 0:
			removeIndices += [i]
		if i < strain.shape[1] - 1:
			if strain[0,i] > strain[0,i+1]:
				removeIndices += [i]
	cleanIndices = np.setdiff1d(range(n), removeIndices)
	# print(strain[:, cleanIndices], stress[:, cleanIndices])
	return strain[:, cleanIndices], stress[:, cleanIndices]

def checkMonotonicity(y, folder):
	for i in range(len(y) - 1):
		if y[i] > y[i+1]:
			print('stress is not monotonic in %s' % folder)
			break
	return None

### run

folderList = natsorted(glob.glob('sg_input_*'), alg=ns.IGNORECASE)
from scipy.interpolate import interp1d
plt.figure()

for folder in folderList:
	stress_strain_data = np.loadtxt(folder + '/postProc/stress_strain.log', skiprows=4)

	n = len(stress_strain_data)
	n = int(n) + 1

	## get Stress and Strain
	stress = np.atleast_2d(stress_strain_data[:n, 2])
	strain = np.atleast_2d(stress_strain_data[:n, 1])
	## remove non-unique strain
	_, uniq_idx = np.unique(strain, return_index=True)
	strain = strain[:, uniq_idx]
	stress = stress[:, uniq_idx]
	strain -= 1.0 # offset for DAMASK, as strain = 1 when started
	strain, stress = removeNaNStrainStress(strain, stress)
	strain, stress = removeInfStrainStress(strain, stress)
	strain, stress = removeNonsenseStrain(strain, stress)
	# print(strain, stress) # debug
	print(folder, len(strain.ravel()))
	splineInterp = interp1d(strain.ravel(), stress.ravel(), kind='quadratic', fill_value='extrapolate')
	x = np.linspace(np.min(strain.ravel()), np.max(strain.ravel()))	
	# plt.plot(strain.ravel(), stress.ravel())
	# index_ = np.argmax(splineInterp(x))
	# index_ = 12
	# plt.text(strain.ravel()[index_], stress.ravel()[index_], folder)
	# plt.plot(strain.ravel(), splineInterp(strain.ravel())) # c='tab:blue', marker='o', linestyle='-', markersize=6)
	plt.plot(x, splineInterp(x), marker='o', markersize=7)
	checkMonotonicity(splineInterp(x), folder)
	# plt.text(x[index_], splineInterp(x)[index_], folder)


plt.show()


