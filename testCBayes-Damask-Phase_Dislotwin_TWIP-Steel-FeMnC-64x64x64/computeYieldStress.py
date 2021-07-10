

"""
compute yield stress at strain = 0.002
adopted from computeYoungModulus.py
"""



import numpy as np
import glob, os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24


## declare input params
yieldStrain = 0.002

data = np.loadtxt('log.stress_strain.txt', skiprows=7)
increment = np.atleast_2d(data[:, 0])
stress = np.atleast_2d(data[:, 2])

tensionLoadFile = np.loadtxt('../tension.load', dtype=str)
Fdot = float(tensionLoadFile[1])
totalTime = float(tensionLoadFile[11])
totalIncrement = float(tensionLoadFile[13])
# strain = increment * Fdot * totalTime / totalIncrement
strain = np.atleast_2d(data[:,1])
n = len(data)

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
RefL = line([yieldStrain, 0], [maxStrain, youngModulus * (maxStrain - yieldStrain)])

# stressInGPa = stress / 1e9

for i in range(n-1):
	Ltest = line([strain[0,i], stress[0,i]], [strain[0,i+1], stress[0,i+1]])
	# print('RefL = ', RefL)
	# print('Ltest = ', Ltest)
	R = intersection(RefL, Ltest)
	if R:
		print("Intersection detected: ", R)
		yieldStrain = R[0]
		yieldStress = R[1]

	# print('\n')

outFile = open('yield.out', 'w')
outFile.write('%.6e\n' % yieldStrain)
outFile.write('%.6e\n' % yieldStress)
outFile.close()

