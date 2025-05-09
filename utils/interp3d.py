#!/usr/bin/env python3

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html
# https://stackoverflow.com/questions/21836067/interpolate-3d-volume-with-numpy-and-or-scipy

from scipy.interpolate import RegularGridInterpolator
from numpy import linspace, zeros, array
x = linspace(1,4,11)
y = linspace(4,7,22)
z = linspace(7,9,33)
V = zeros((11,22,33))
for i in range(11):
    for j in range(22):
        for k in range(33):
            V[i,j,k] = 100*x[i] + 10*y[j] + z[k]
fn = RegularGridInterpolator((x,y,z), V)
pts = array([[2,6,8],[3,5,7],[2,6,8],[2,6,8]])
print(fn(pts))

