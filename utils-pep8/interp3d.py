#!/usr/bin/env python3

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html
# https://stackoverflow.com/questions/21836067/interpolate-3d-volume-with-numpy-and-or-scipy

from numpy import array, linspace, zeros
from scipy.interpolate import RegularGridInterpolator

X = linspace(1, 4, 11)
Y = linspace(4, 7, 22)
Z = linspace(7, 9, 33)
V = zeros((11, 22, 33))
for i in range(11):
    for j in range(22):
        for k in range(33):
            V[i, j, k] = 100 * X[i] + 10 * Y[j] + Z[k]
FN = RegularGridInterpolator((X, Y, Z), V)
PTS = array([[2, 6, 8], [3, 5, 7], [2, 6, 8], [2, 6, 8]])
print(FN(PTS))
