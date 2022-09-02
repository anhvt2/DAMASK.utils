
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
bayesOpt_input = np.loadtxt('input.dat')
d = len(bayesOpt_input) # dimensionality

parentPath = os.getcwd()

### NOTE: for new case study, modify getDamaskParams() and parseInput()
def getDamaskParams(bayesOpt_input):
  lower_bounds = [0.0001,  10, 1, 1e6, 1e6,    1e6]
  upper_bounds = [0.0085, 120, 5, 1e7, 1e7, 1000e6]
  lower_bounds = np.array(lower_bounds)
  upper_bounds = np.array(upper_bounds)
  matcfg_input = lower_bounds + (upper_bounds - lower_bounds) * (bayesOpt_input - (0)) / (+1 - (0))
  return matcfg_input

def parseInput(matcfg_input, txtcfg):
  # unpack values and scale to units
  gdot0_slip    = matcfg_input[0]
  n_slip        = matcfg_input[1]
  a_slip        = matcfg_input[2]
  tau0_slip     = matcfg_input[3] 
  tausat_slip   = matcfg_input[4]
  h0_slipslip   = matcfg_input[5]
  parsed_txtcfg = txtcfg # work on a copied version
  # change lines: always add '\n' at the end of the line
  # parsed_txtcfg[48 - 1] = 'gdot0_slip              0.001\n'
  # parsed_txtcfg[49 - 1] = 'n_slip                  83.3\n'
  # parsed_txtcfg[50 - 1] = 'a_slip                  2.25\n'
  # parsed_txtcfg[51 - 1] = 'tau0_slip               95.e6\n'
  # parsed_txtcfg[52 - 1] = 'tausat_slip             222.e6\n'
  # parsed_txtcfg[53 - 1] = 'h0_slipslip             1.0e6\n'
  parsed_txtcfg[48 - 1] = 'gdot0_slip              %.12e\n' % gdot0_slip
  parsed_txtcfg[49 - 1] = 'n_slip                  %.12e\n' % n_slip
  parsed_txtcfg[50 - 1] = 'a_slip                  %.12e\n' % a_slip
  parsed_txtcfg[51 - 1] = 'tau0_slip               %.12e\n' % tau0_slip
  parsed_txtcfg[52 - 1] = 'tausat_slip             %.12e\n' % tausat_slip
  parsed_txtcfg[53 - 1] = 'h0_slipslip             %.12e\n' % h0_slipslip
  return parsed_txtcfg


### main function
matcfgFile = open('material.config') # use 'materials.config' as the template file
txtcfg = matcfgFile.readlines()
matcfgFile.close()

matcfg_input = getDamaskParams(bayesOpt_input) # translate from [0,1] to [lower_bounds, upper_bounds]
np.savetxt('bayesOpt_input.dat'     , bayesOpt_input, fmt='%.16e', delimiter=',') # debug
np.savetxt('matcfg_input.dat', matcfg_input, fmt='%.16e', delimiter=',') # debug
parsed_txtcfg = parseInput(matcfg_input, txtcfg)

# write to material.config
f = open('material.config', 'w') # can be 'r', 'w', 'a', 'r+'
for j in range(len(parsed_txtcfg)):
  f.write(parsed_txtcfg[j])
f.close()

# write to material.config.preamble
f = open('material.config.preamble', 'w') # can be 'r', 'w', 'a', 'r+'
for j in range(len(parsed_txtcfg)):
  f.write(parsed_txtcfg[j])
f.close()

