
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

