#!/usr/bin/env python3


"""
compute yield stress at strain = 0.002
adopted from computeYoungModulus.py
"""

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

print('Running computeyield_stress.py at: %s' % os.getcwd())

mpl.rc_params['xtick.labelsize'] = 24
mpl.rc_params['ytick.labelsize'] = 24

PARSER = argparse.ArgumentParser(description='')
PARSER.add_argument(
    "-stress_strain_file", "--stress_strain_file", default='stress_strain.log', type=str
)

PARSER.add_argument("-load_file", "--load_file", default='tension.load', type=str)

PARSER.add_argument("-opt_save_fig", "--opt_save_fig", type=bool, default=False)
ARGS = PARSER.parse_args()
STRESS_STRAIN_FILE = ARGS.stress_strain_file


STRESS_STRAIN_DATA = np.loadtxt(STRESS_STRAIN_FILE, skiprows=4)
np.atleast_2d(STRESS_STRAIN_DATA[:, 1])

# deprecated
# only consider the first segment

N = len(STRESS_STRAIN_DATA)
N = int(N) + 1

# get Stress and Strain
STRESS = np.atleast_2d(STRESS_STRAIN_DATA[:N, 2])
STRAIN = np.atleast_2d(STRESS_STRAIN_DATA[:N, 1])
# remove non-unique strain
_, UNIQ_IDX = np.unique(STRAIN, return_index=True)
STRAIN = STRAIN[:, UNIQ_IDX]
STRESS = STRESS[:, UNIQ_IDX]
STRAIN -= 1.0  # offset for DAMASK, as strain = 1 when started
print('Stress:')
print(STRESS.ravel())
print('\n\n')

print('Strain:')
print(STRAIN.ravel())
print('\n\n')


# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html


# extract elastic part
ELASTIC_STRESS = np.atleast_2d(STRESS[0, 1:5]).T
ELASTIC_STRAIN = np.atleast_2d(STRAIN[0, 1:5]).T
# perform linear regression
REG = LinearRegression().fit(ELASTIC_STRAIN, ELASTIC_STRESS)
REG.score(ELASTIC_STRAIN, ELASTIC_STRESS)
YOUNG_MODULUS_IN_G_PA = REG.coef_[0, 0] / 1e9
YOUNG_MODULUS = YOUNG_MODULUS_IN_G_PA * 1e9

print('Elastic Young modulus = %.4f GPa' % YOUNG_MODULUS_IN_G_PA)
print('Intercept = %.4f' % (REG.intercept_ / 1e9))
# adopt the intersection from
# https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
# see: https://stackoverflow.com/a/20679579/2486448
def _line(p1, p2):
    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = p1[0] * p2[1] - p2[0] * p1[1]
    return (a, b, -c, p1, p2)


def _intersection(L1, L2):
    d = L1[0] * L2[1] - L1[1] * L2[0]
    dx = L1[2] * L2[1] - L1[1] * L2[2]
    dy = L1[0] * L2[2] - L1[2] * L2[0]
    if d == 0:
        return False
    x = dx / d
    y = dy / d
    X = np.array([x, y])
    p1 = np.array(L2[3])
    p2 = np.array(L2[4])
    if p2[0] < X[0] < p1[0] or p1[0] < X[0] < p2[0]:
        return (x, y)
# if R:
#   print "Intersection detected:", R
# else:
#   print "No single intersection point detected"


MAX_STRAIN = np.max(STRAIN)
YIELD_STRAIN = 0.002
REF_L = _line([YIELD_STRAIN, 0], [MAX_STRAIN, YOUNG_MODULUS * (MAX_STRAIN - YIELD_STRAIN)])


try:

    for i in range(N - 2):
        Ltest = _line([STRAIN[0, i], STRESS[0, i]], [STRAIN[0, i + 1], STRESS[0, i + 1]])
        R = _intersection(REF_L, Ltest)
        if R:
            print("Intersection detected: ", R)
            computed_yield_strain = R[0]
            computed_yield_stress = R[1]

    outFile = open('output.dat', 'w')
    outFile.write('%.6e\n' % computed_yield_strain)
    outFile.write('%.6e\n' % computed_yield_stress)
    outFile.close()

    print("##########")
    print(
        r"Intersection with Young modulus (obtained from linear regression) with $\sigma-\varepsilon$ occured at:"
    )
    print("Yield Strain = %.4f" % computed_yield_strain)
    print("Yield Stress = %.4f GPa" % (computed_yield_stress / 1e9))
    print("##########\n")

    # plot check

    strain_intersect_line = np.linspace(YIELD_STRAIN, MAX_STRAIN)
    stress_intersect_line = np.linspace(0, YOUNG_MODULUS * (MAX_STRAIN - YIELD_STRAIN))
    plt.figure()
    from scipy.interpolate import interp1d

    splineInterp = interp1d(
        STRAIN.ravel(), STRESS.ravel() / 1e6, kind='cubic', fill_value='extrapolate'
    )
    (aPlt,) = plt.plot(
        STRAIN.ravel(),
        splineInterp(STRAIN.ravel()),
        c='tab:blue',
        marker='o',
        linestyle='-',
        markersize=6,
        label=r'$\sigma_{vM}-\varepsilon_{vM}$ curve',
    )

    (bPlt,) = plt.plot(
        strain_intersect_line,
        stress_intersect_line / 1e6,
        color='tab:red',
        marker='s',
        linestyle=':',
        markersize=5,
        label=r'0.2% offset',
    )
    (cPlt,) = plt.plot(
        computed_yield_strain,
        computed_yield_stress / 1e6,
        c='black',
        marker='*',
        linestyle='None',
        markersize=20,
        label=r'yield point',
    )
    plt.xlabel(r'$\varepsilon_{vM}$ [-]', fontsize=30)
    plt.ylabel(r'$\sigma_{vM}$ [MPa]', fontsize=30)
    plt.title(r'$\sigma_{vM}-\varepsilon_{vM}$ phenomenological constitutive model Cu', fontsize=24)
    plt.legend(handles=[aPlt, bPlt, cPlt], fontsize=24)
    plt.xlim([0, np.max(STRAIN)])
    plt.ylim([np.min(STRESS), 1.2 * np.max(STRESS) / 1e6])
    plt.show()

except:
    outFile = open('../log.feasible', 'w')
    outFile.write('%d\n' % 0)
    outFile.close()
