
import numpy as np
import os, sys
from scipy.stats import qmc

# Set bounds for (1) strain-rates and (2) temperature
lnDotVareps_bounds = [-4, 2]
T_bounds = [273, 1073]
# Set terminated strain
maxTargetStrain = 0.2

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
dotVareps = sample_scaled[:,0]
initialT  = sample_scaled[:,1]
loadingTime = maxTargetStrain / sample_scaled[:,0]
logger = open('control.log', 'w')

# Create file based on DAMASK 'template/' folder
for i in range(numSim):
    # Go to local folder
    folderName = str(i+1)
    os.system(f'cp -rfv -L template/ {folderName}/')
    os.chdir(currentPath + f'/{folderName}/')
    # Write tension.load
    f = open('tension.load', 'w')
    f.write(f"fdot    0 0 0    0 * 0    * 0 {dotVareps[i]:<10.8e}    stress    * * *    * 0 *    0 * *    time {loadingTime[i]:<10.8e}    logincs 20\n") # 'fdot    0 0 0    0 * 0    * 0 1.0e-3    stress    * * *    * 0 *    0 * *    time 20    logincs 20'
    f.close()
    # Write initialT.config
    f = open('initialT.config', 'w')
    f.write(f'initialT    {initialT[i]:<5.1f}\n')
    f.close()
    # Log information in logger
    logger.write(f'{i}, {dotVareps[i]:<10.8e}, {loadingTime[i]:<10.8e}, {initialT[i]:<5.1f}\n')
    # Go back to the main directory
    os.chdir(currentPath)
    # Diagnostics
    print(f'Finished folder {folderName}/')

logger.close()
os.chdir(currentPath) # reset to the original path

