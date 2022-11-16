
import scipy.io as io
d = io.loadmat('AllAdditiveDataCompiled.mat')

sorted(d.keys())
d['AllData']

d['AllData']['StrainPerc']
d['AllData']['StressMPa']
d['AllData']['Name']
d['AllData']['UTSMPa']
d['AllData']['LaserPowerW']
d['AllData']['LaserVelocitymms']
d['AllData']['Hatch']
d['AllData']['YieldStrengthMPa']
d['AllData']['UnloadingModulusGPa']
d['AllData']['PlasticElongationPerc']

n = d['AllData']['StrainPerc'].shape[1] # number of samples
# print(n) 

## reformat/reshape fields

StrainPerc = d['AllData']['StrainPerc'].reshape(n)
StressMPa = d['AllData']['StressMPa'].reshape(n)
Name = d['AllData']['Name'].reshape(n)
UTSMPa = d['AllData']['UTSMPa'].reshape(n)
LaserPowerW = d['AllData']['LaserPowerW'].reshape(n)
LaserVelocitymms = d['AllData']['LaserVelocitymms'].reshape(n)
Hatch = d['AllData']['Hatch'].reshape(n)
YieldStrengthMPa = d['AllData']['YieldStrengthMPa'].reshape(n)
UnloadingModulusGPa = d['AllData']['UnloadingModulusGPa'].reshape(n)
PlasticElongationPerc = d['AllData']['PlasticElongationPerc'].reshape(n)

## plot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

for i in range(n):
	plt.plot(d['AllData']['StrainPerc'][0,i], d['AllData']['StressMPa'][0,i], linestyle='None', marker='o', markersize=1)

plt.xlabel(r'$\varepsilon$ [%]', fontsize=24)
plt.ylabel(r'$\sigma$ [MPa]', fontsize=24)

plt.show()
