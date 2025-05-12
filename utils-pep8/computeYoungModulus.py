#!/usr/bin/env python3

# addStrainTensors one_phase_equiaxed_tensionX.txt --left --logarithmic
# addCauchy one_phase_equiaxed_tensionX.txt
# addMises one_phase_equiaxed_tensionX.txt --strain 'ln(V)' --stress Cauchy
# filterTable < one_phase_equiaxed_tensionX.txt --white inc,'Mises(ln(v))','Mises(Cauchy) > log.stress_strain.txt


# http://materials.iisc.ernet.in/~praveenk/CrystalPlasticity/PE_N_DAMASK.pdf
# also see E.1.5 in [Multiscale simulation of metal deformation in deep drawing using machine learning by Verwijs, Floor]

import argparse

import matplotlib as mpl
import numpy as np
from sklearn.linear_model import LinearRegression

mpl.rc_params['xtick.labelsize'] = 24
mpl.rc_params['ytick.labelsize'] = 24

PARSER = argparse.ArgumentParser(description='')
PARSER.add_argument(
    "-stress_strain_file", "--stress_strain_file", default='stress_strain.log', type=str
)

PARSER.add_argument("-load_file", "--load_file", default='tension.load', type=str)

PARSER.add_argument("-opt_save_fig", "--opt_save_fig", type=bool, default=False)
PARSER.add_argument("-n_elastic_points", "--n_elastic_points", type=int, default=3)
ARGS = PARSER.parse_args()
LOAD_FILE = ARGS.load_file
N_ELASTIC_POINTS = ARGS.n_elastic_points


def _read_load_file(load_file):
    load_data = np.loadtxt(load_file, dtype=str)
    n_fields = len(load_data)
    for i in range(n_fields):
        if load_data[i] == 'Fdot' or load_data[i] == 'fdot':
            print('Found *Fdot*!')
        if load_data[i] == 'time':
            print('Found *totalTime*!')
        if load_data[i] == 'incs':
            print('Found *totalIncrement*!')
        if load_data[i] == 'freq':
            print('Found *freq*!')
    return (FDOT11, TOTAL_TIME, TOTAL_INCREMENT)


STRESS_STRAIN_DATA = np.loadtxt('stress_strain.log', skiprows=7)
np.atleast_2d(STRESS_STRAIN_DATA[:, 1])
FDOT11, TOTAL_TIME, TOTAL_INCREMENT = _read_load_file(LOAD_FILE)
FDOT = FDOT11
N = len(STRESS_STRAIN_DATA)
N = int(N) + 1
# get Stress and Strain
STRESS = np.atleast_2d(STRESS_STRAIN_DATA[:N, 2])
STRAIN = np.atleast_2d(STRESS_STRAIN_DATA[:N, 1])
STRAIN -= 1.0  # offset for DAMASK, as strain = 1 when started
print('Stress:')
print(STRESS.ravel())
print('\n\n')

print('Strain:')
print(STRAIN.ravel())
print('\n\n')
# varepsilon (strain) = varepsilonDot (or strainDot) * time = varepsilonDot * increment / totalIncrement * totalTime

print("#############################")
print("Reading *tension.load* file:")
print("Fdot = %.4e" % FDOT)
print("totalTime = %.1f" % TOTAL_TIME)
print("totalIncrement = %.1f" % TOTAL_INCREMENT)
print("#############################")


# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html


# extract elastic part
ELASTIC_STRESS = np.atleast_2d(STRESS[0, 1:N_ELASTIC_POINTS]).T
ELASTIC_STRAIN = np.atleast_2d(STRAIN[0, 1:N_ELASTIC_POINTS]).T
# perform linear regression
REG = LinearRegression().fit(ELASTIC_STRAIN, ELASTIC_STRESS)
REG.score(ELASTIC_STRAIN, ELASTIC_STRESS)
YOUNG_MODULUS_IN_G_PA = REG.coef_[0, 0] / 1e9

print('Elastic Young modulus = %.4f GPa' % YOUNG_MODULUS_IN_G_PA)
print('Intercept = %.4f' % (REG.intercept_ / 1e9))


with open('output.dat', 'w') as outFile:
    outFile.write('%.6e\n' % YOUNG_MODULUS_IN_G_PA)
