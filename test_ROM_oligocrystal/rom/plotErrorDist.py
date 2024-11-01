import pyvista
import matplotlib.pyplot as plt
import glob, os
import numpy as np
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

d = np.loadtxt('control.log', skiprows=1, delimiter=',')
dotVarEps = d[:,1]
loadingTime = d[:,2] # dependent - not an input
initialT = d[:,3]

TrainIdx   = np.loadtxt('TrainIdx.dat', dtype=int)
TestIdx    = np.loadtxt('TestIdx.dat', dtype=int)
TestIdxOOD = np.loadtxt('TestIdxOOD.dat', dtype=int)
TestIdxID  = np.loadtxt('TestIdxID.dat', dtype=int)
FoI = ['Mises(Cauchy)','Mises(ln(V))'] # from export2npy.py


