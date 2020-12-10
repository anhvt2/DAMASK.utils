
# addStrainTensors one_phase_equiaxed_tensionX.txt --left --logarithmic
# addCauchy one_phase_equiaxed_tensionX.txt
# addMises one_phase_equiaxed_tensionX.txt --strain 'ln(V)' --stress Cauchy
# filterTable < one_phase_equiaxed_tensionX.txt --white inc,'Mises(ln(v))','Mises(Cauchy) > log.stress_strain.txt


# http://materials.iisc.ernet.in/~praveenk/CrystalPlasticity/PE_N_DAMASK.pdf
# also see E.1.5 in [Multiscale simulation of metal deformation in deep drawing using machine learning by Verwijs, Floor]

import numpy as np
import glob, os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24


data = np.loadtxt('log.stress_strain.txt', skiprows=7)
increment = np.atleast_2d(data[:, 0])
stress = np.atleast_2d(data[:, 2])

tensionLoadFile = np.loadtxt('../tension.load', dtype=str)
Fdot = float(tensionLoadFile[1])
totalTime = float(tensionLoadFile[11])
totalIncrement = float(tensionLoadFile[13])
strain = increment * Fdot * totalTime / totalIncrement



# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html


## extract elastic part
elasticStress = np.atleast_2d(stress[0,1:5]).T
elasticStrain = np.atleast_2d(strain[0,1:5]).T

# print(elasticStrain.shape)
# print(elasticStress.shape)

## perform linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(elasticStrain, elasticStress)
reg.score(elasticStrain, elasticStress)
youngModulusInGPa = reg.coef_[0,0] / 1e9

print('Elastic Young modulus = %.4f GPa' % youngModulusInGPa)
print('Intercept = %.4f' % (reg.intercept_ / 1e9))


outFile = open('youngModulus.out', 'w')
outFile.write('%.6e\n' % youngModulusInGPa)
outFile.close()


