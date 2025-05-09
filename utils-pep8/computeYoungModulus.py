#!/usr/bin/env python3

# addStrainTensors one_phase_equiaxed_tensionX.txt --left --logarithmic
# addCauchy one_phase_equiaxed_tensionX.txt
# addMises one_phase_equiaxed_tensionX.txt --strain 'ln(V)' --stress Cauchy
# filterTable < one_phase_equiaxed_tensionX.txt --white inc,'Mises(ln(v))','Mises(Cauchy) > log.stress_strain.txt


# http://materials.iisc.ernet.in/~praveenk/CrystalPlasticity/PE_N_DAMASK.pdf
# also see E.1.5 in [Multiscale simulation of metal deformation in deep drawing using machine learning by Verwijs, Floor]

from sklearn.linear_model import LinearRegression
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

parser = argparse.ArgumentParser(description='')
parser.add_argument("-StressStrainFile", "--StressStrainFile",
                    default='stress_strain.log', type=str)
parser.add_argument("-LoadFile", "--LoadFile",
                    default='tension.load', type=str)
parser.add_argument("-optSaveFig", "--optSaveFig", type=bool, default=False)
parser.add_argument("-nElasticPoints", "--nElasticPoints", type=int, default=3)
args = parser.parse_args()
StressStrainFile = args.StressStrainFile
LoadFile = args.LoadFile
nElasticPoints = args.nElasticPoints


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


stress_strain_data = np.loadtxt('stress_strain.log', skiprows=7)
increment = np.atleast_2d(stress_strain_data[:, 1])
Fdot11, totalTime, totalIncrement = readLoadFile(LoadFile)
Fdot = Fdot11
n = len(stress_strain_data)
n = int(n) + 1


# increment = np.atleast_2d(stress_strain_data[:, 0])
# stress = np.atleast_2d(stress_strain_data[:, 2])

# tensionLoadFile = np.loadtxt('../tension.load', dtype=str)

# get Stress and Strain
stress = np.atleast_2d(stress_strain_data[:n, 2])
strain = np.atleast_2d(stress_strain_data[:n, 1])
strain -= 1.0  # offset for DAMASK, as strain = 1 when started
print('Stress:')
print(stress.ravel())
print('\n\n')

print('Strain:')
print(strain.ravel())
print('\n\n')


# Fdot = float(tensionLoadFile[1])
# totalTime = float(tensionLoadFile[11])
# totalIncrement = float(tensionLoadFile[13])
# strain = Fdot * increment * totalTime / totalIncrement
# varepsilon (strain) = varepsilonDot (or strainDot) * time = varepsilonDot * increment / totalIncrement * totalTime

print("#############################")
print("Reading *tension.load* file:")
print("Fdot = %.4e" % Fdot)
print("totalTime = %.1f" % totalTime)
print("totalIncrement = %.1f" % totalIncrement)
print("#############################")


# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html


# extract elastic part
elasticStress = np.atleast_2d(stress[0, 1:nElasticPoints]).T
elasticStrain = np.atleast_2d(strain[0, 1:nElasticPoints]).T

# print(elasticStrain.shape)
# print(elasticStress.shape)

# perform linear regression
reg = LinearRegression().fit(elasticStrain, elasticStress)
reg.score(elasticStrain, elasticStress)
youngModulusInGPa = reg.coef_[0, 0] / 1e9

print('Elastic Young modulus = %.4f GPa' % youngModulusInGPa)
print('Intercept = %.4f' % (reg.intercept_ / 1e9))


outFile = open('output.dat', 'w')
outFile.write('%.6e\n' % youngModulusInGPa)
outFile.close()
