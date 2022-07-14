
"""
This script 
	(1) reads a parameterized input from input.dat
	(2) parse to material.config

All of these is to prepare for a correct DAMASK execution. 

* adopted from test_stochastic-collocation_runs_s1057681/phenomenological-slipping-Cu/parseDakota2MaterialConfig.py
"""



import numpy as np
import os, sys, time, glob, datetime

## read input
paramsInput = np.loadtxt('input.dat')
d = len(paramsInput) # dimensionality

matcfgFile = open('material.config') # use 'materials.config' as the template file
txtcfg = matcfgFile.readlines()
matcfgFile.close()

parentPath = os.getcwd()

### NOTE: for new case study, modify getDamaskParams() and parseInput()
def getDamaskParams(sg_input):
  # translate dakota input from [-1,1] to real bounds imposed by user
  # [tau0_slip, tausat_slip, h0, n, a_slip]
  lower_bounds = [0.5e6,  80e6, 200e6,  50, 1]
  upper_bounds = [3.5e6, 140e6, 280e6, 120, 4]
  sg_input     = np.array(sg_input)[2:] # ignore 'id' and 'weights' two columns
  lower_bounds = np.array(lower_bounds)
  upper_bounds = np.array(upper_bounds)
  real_sg_input = lower_bounds + (upper_bounds - lower_bounds) * (sg_input - (-1)) / (+1 - (-1))
  return real_sg_input

def parseInput(real_sg_input, txtcfg):
  # unpack values and scale to units
  tau0_slip   = real_sg_input[0]
  tausat_slip = real_sg_input[1]
  h0_slipslip = real_sg_input[2] 
  n           = real_sg_input[3]
  a_slip      = real_sg_input[4]
  parsed_txtcfg = txtcfg # work on a copied version
  # change lines: always add '\n' at the end of the line
  # parsed_txtcfg[65 - 1] = 'tau0_slip               16.0e6                # per family\n'
  # parsed_txtcfg[66 - 1] = 'tausat_slip             148.0e6               # per family\n'
  # parsed_txtcfg[70 - 1] = 'h0_slipslip             2.4e8 # old value: 180e6\n'
  # parsed_txtcfg[64 - 1] = 'n_slip                  83.3\n'
  # parsed_txtcfg[67 - 1] = 'a_slip                  2.25\n'
  parsed_txtcfg[65 - 1] = 'tau0_slip               %.12e                # per family\n' % tau0_slip
  parsed_txtcfg[66 - 1] = 'tausat_slip             %.12e               # per family\n' % tausat_slip
  parsed_txtcfg[70 - 1] = 'h0_slipslip             %.12e\n' % h0_slipslip
  parsed_txtcfg[64 - 1] = 'n_slip                  %.12e\n' % n
  parsed_txtcfg[67 - 1] = 'a_slip                  %.12e\n' % a_slip
  return parsed_txtcfg


### main function
os.system('rm -rfv sg_input_*') # remove all folders

for i in range(n):
  ii = i + 1 # dakota index starts from 1
  folderName = 'sg_input_%d' % (ii)
  os.system('mkdir -p %s' % (folderName))
  localPath = parentPath + '/' + folderName
  os.chdir(localPath)
  ## link file from parent directory
  os.system('ln -sf ../numProcessors.dat .')
  os.system('ln -sf ../single_phase_equiaxed.geom .')
  os.system('ln -sf ../numerics.config .')
  os.system('ln -sf ../tension.load .')
  os.system('ln -sf ../run_damask.sh .')
  os.system('ln -sf ../sbatch.damask.solo .')
  ## write new 'material.config' locally
  sg_input = paramsInput[i] # get input from dakota
  real_sg_input = getDamaskParams(sg_input)
  np.savetxt('sg_input_%d.dat' % ii     , sg_input, fmt='%.16e', delimiter=',') # debug
  np.savetxt('real_sg_input_%d.dat' % ii, real_sg_input, fmt='%.16e', delimiter=',') # debug
  parsed_txtcfg = parseInput(real_sg_input, txtcfg)
  f = open('material.config', 'w') # can be 'r', 'w', 'a', 'r+'
  for j in range(len(parsed_txtcfg)):
    f.write(parsed_txtcfg[j])
  f.close()
  os.chdir(parentPath)
  print('Created folder %s' % folderName)

