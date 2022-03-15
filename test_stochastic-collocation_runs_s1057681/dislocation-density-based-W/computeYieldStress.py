

"""
compute yield stress at strain = 0.002
adopted from computeYoungModulus.py
"""



import numpy as np
import glob, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

print('Running computeYieldStress.py at: %s' % os.getcwd())

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

parser = argparse.ArgumentParser(description='')
parser.add_argument("-StressStrainFile", "--StressStrainFile", default='stress_strain.log', type=str)
parser.add_argument("-LoadFile", "--LoadFile", default='tension.load', type=str)
parser.add_argument("-optSaveFig", "--optSaveFig", type=bool, default=False)
args = parser.parse_args()
StressStrainFile = args.StressStrainFile
LoadFile = args.LoadFile

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


stress_strain_data = np.loadtxt(StressStrainFile, skiprows=4)
increment = np.atleast_2d(stress_strain_data[:, 1])

## deprecated
# load_data = np.loadtxt(LoadFile, dtype=str)
# only consider the first segment
# Fdot = float(load_data[0,1])
# totalTime = float(load_data[0,11])
# totalIncrement = float(load_data[0,13])

# Fdot11, totalTime, totalIncrement = readLoadFile(LoadFile)
# Fdot = Fdot11

# n = len(stress_strain_data) * np.array(load_data[:,11], dtype=float)[0] / np.sum(np.array(load_data[:,13], dtype=float)) # only consider the first loading segment # deprecated -- but should
n = len(stress_strain_data)
n = int(n) + 1

## get Stress and Strain
stress = np.atleast_2d(stress_strain_data[:n, 2])
strain = np.atleast_2d(stress_strain_data[:n, 1])
## remove non-unique strain
strain -= 1.0 # offset for DAMASK, as strain = 1 when started
_, uniq_idx = np.unique(strain, return_index=True)
strain = np.hstack(( np.array([[0]]) , strain )) # pad zeros
stress = np.hstack(( np.array([[0]]) , stress )) # pad zeros
strain = strain[:, uniq_idx]
stress = stress[:, uniq_idx]

def removeNaNStrainStress(strain,stress):
	m, n = stress.shape
	removeIndices = np.argwhere(np.isnan(stress.ravel()))
	cleanIndices = np.setdiff1d(range(n), removeIndices)
	return strain[:, cleanIndices], stress[:, cleanIndices]

def removeNonsenseStrain(strain, stress):
	m, n = stress.shape
	removeIndices = []
	for i in range(strain.shape[1] - 1):
		# print(strain[0,i])
		# print(strain.shape)
		if strain[0,i] > 10 or strain[0,i] < 0 or strain[0,i] > strain[0,i+1]:
			removeIndices += [i]
	cleanIndices = np.setdiff1d(range(n), removeIndices)
	return strain[:, cleanIndices], stress[:, cleanIndices]

strain, stress = removeNaNStrainStress(strain, stress)
strain, stress = removeNonsenseStrain(strain, stress)
print('Stress:')
print(stress.ravel())
print('\n\n')

print('Strain:')
print(strain.ravel())
print('\n\n')

n = stress.shape[1] + 1 # update

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html


## extract elastic part
elasticStress = np.atleast_2d(stress[0,1:3]).T
elasticStrain = np.atleast_2d(strain[0,1:3]).T

# print(elasticStrain.shape)
# print(elasticStress.shape)

## perform linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(elasticStrain, elasticStress)
reg.score(elasticStrain, elasticStress)
youngModulusInGPa = reg.coef_[0,0] / 1e9
youngModulus = youngModulusInGPa * 1e9

print('Elastic Young modulus = %.4f GPa' % youngModulusInGPa)
print('Intercept = %.4f' % (reg.intercept_ / 1e9))



# outFile = open('youngModulus.out', 'w')
# outFile.write('%.6e\n' % youngModulusInGPa)
# outFile.close()

# adopt the intersection from 
# https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
# see: https://stackoverflow.com/a/20679579/2486448
def line(p1, p2):
	A = (p1[1] - p2[1])
	B = (p2[0] - p1[0])
	C = (p1[0]*p2[1] - p2[0]*p1[1])
	return A, B, -C, p1, p2

def intersection(L1, L2):
	D  = L1[0] * L2[1] - L1[1] * L2[0]
	Dx = L1[2] * L2[1] - L1[1] * L2[2]
	Dy = L1[0] * L2[2] - L1[2] * L2[0]
	if D != 0:
		x = Dx / D
		y = Dy / D
		# check if x,y is between p1, p2
		X = np.array([x,y])
		p1 = np.array(L2[3])
		p2 = np.array(L2[4])
		# print('X = ', X, '; p1 = ', p1, '; p2 = ', p2)
		if p2[0] < X[0] < p1[0] or p1[0] < X[0] < p2[0]:
			return x,y
	else:
		return False

# L1 = line([0,1], [2,3])
# L2 = line([2,3], [0,4])

# R = intersection(L1, L2)
# if R:
# 	print "Intersection detected:", R
# else:
# 	print "No single intersection point detected"

maxStrain = np.max(strain)
yieldStrain = 0.002
RefL = line([yieldStrain, 0], [maxStrain, youngModulus * (maxStrain - yieldStrain)])

""" 
EXPLANATION:
Step 1: Compute / Estimate Young modulus using linear regression with a certain number of points

Step 2: From the point of (Strain = yieldStrain [defined at 0.2%], Stress = 0), draw a line with the slope of Young modulus that ends at (maxStrain)
		If visualizing, the end point is at (maxStrain, youngModulus * (maxStrain - yieldStrain)).

Step 3: Check if any segment in the stress strain curve intersects with this line. If yes, return the intersection and we shall call it yieldStress.	
"""


try:
	# stressInGPa = stress / 1e9

	for i in range(n-2):
		Ltest = line([strain[0,i], stress[0,i]], [strain[0,i+1], stress[0,i+1]])
		# print('RefL = ', RefL)
		# print('Ltest = ', Ltest)
		R = intersection(RefL, Ltest)
		# print('%d/%d' % (i,n)) # debug
		if R:
			print("Intersection detected: ", R)
			computed_yieldStrain = R[0]
			computed_yieldStress = R[1]
			break
		# print('\n')

	outFile = open('output.dat', 'w')
	outFile.write('%.12e\n' % computed_yieldStrain)
	outFile.write('%.12e\n' % computed_yieldStress)
	outFile.close()

	print("##########")
	print(r"Intersection with Young modulus (obtained from linear regression) with $\sigma-\varepsilon$ occured at:")
	print("Yield Strain = %.4f" % computed_yieldStrain)
	print("Yield Stress = %.4f GPa" % (computed_yieldStress / 1e9))
	print("##########\n")

	### plot check

	# strain_intersect_line = np.linspace(yieldStrain, maxStrain)
	# stress_intersect_line = np.linspace(0, youngModulus * (maxStrain - yieldStrain))
	# plt.figure()
	# from scipy.interpolate import interp1d
	# splineInterp = interp1d(strain.ravel(), stress.ravel() / 1e6, kind='cubic', fill_value='extrapolate')
	# # aPlt, = plt.plot(strain.ravel(), stress.ravel() / 1e6, 'bo-', markersize=5, linewidth=2)
	# aPlt, = plt.plot(strain.ravel(), splineInterp(strain.ravel()), c='tab:blue', marker='o', linestyle='-', markersize=6, label=r'$\sigma_{vM}-\varepsilon_{vM}$ # curve')
	# # x = np.linspace(np.min(strain.ravel()), np.max(strain.ravel()))	
	# # aPlt, = plt.plot(x, splineInterp(x), c='tab:blue', marker='o', linestyle='-', markersize=6, label=r'$\sigma_{vM}-\varepsilon_{vM}$ curve')

	# bPlt, = plt.plot(strain_intersect_line, stress_intersect_line / 1e6, color='tab:red', marker='s', linestyle=':', markersize=5, label=r'0.2% offset')
	# cPlt, = plt.plot(computed_yieldStrain, computed_yieldStress / 1e6, c='black', marker='*', linestyle='None', markersize=20, label=r'yield point')
	# plt.xlabel(r'$\varepsilon_{vM}$ [-]', fontsize=30)
	# plt.ylabel(r'$\sigma_{vM}$ [MPa]', fontsize=30)
	# plt.title(r'$\sigma_{vM}-\varepsilon_{vM}$ phenomenological constitutive model Cu', fontsize=24)
	# plt.legend(handles=[aPlt, bPlt, cPlt], fontsize=24)
	# # plt.xlim([0, np.max(strain)])
	# plt.ylim([np.min(stress), 1.2 * np.max(stress) / 1e6])
	# plt.show()

except:
	outFile = open('../log.feasible', 'w')
	outFile.write('%d\n' % 0)
	outFile.close()	


