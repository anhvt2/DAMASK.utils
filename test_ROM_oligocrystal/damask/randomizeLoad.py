
import numpy as np
import os, sys
from scipy.stats import qmc

# Set bounds for (1) strain-rates and (2) temperature
lnDotVareps_bounds = [-4, 2]
T_bounds = [273, 1073]
# Set terminated strain
maxStrain = 0.2

lower_bounds = [lnDotVareps_bounds[0], T_bounds[0]]
upper_bounds = [lnDotVareps_bounds[1], T_bounds[1]]

numSim = 1000
currentPath = os.getcwd()

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html
sampler = qmc.LatinHypercube(d=2)
sample = sampler.random(n=numSim)
sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)
sample_scaled[:,0] = np.power(10, sample_scaled[:,0])
sample_scaled[:,1] = np.round(sample_scaled[:,1], decimals=0)

# Calculate time based on terminated strain


# Create file based on DAMASK 'template/' folder
for i in range(1,numSim+1):
	os.system('cp -rfv -L template/ %d/' % i)
	os.chdir(currentPath + '/%d/')
	f = open('tension.load')
	vareps = np.random.uniform()
	f.write('fdot    0 0 0    0 * 0    * 0 1.0e-3    stress    * * *    * 0 *    0 * *    time 20    incs 20')
	f.close()
	os.chdir(currentPath)

os.chdir(currentPath) # reset to the original path

