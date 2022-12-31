
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
bayesOpt_input = np.loadtxt('input.dat', delimiter=',')
d = len(bayesOpt_input) # dimensionality

parentPath = os.getcwd()

### NOTE: for new case study, modify getDamaskParams() and parseInput()
def getDamaskParams(bayesOpt_input):
  # n_slip, a_slip, tau0_slip, tausat_slip, h0_slipslip
  # opt 1:
  # lower_bounds = [  1.2,  1, 1e6,     1e8,   1e8]
  # upper_bounds = [  150, 25, 150e6, 120e8, 100e8]
  # opt 2:
  # lower_bounds = [  1.2,  10,  1e5,   1e6,   1e6]
  # upper_bounds = [  50., 100, 50e5, 200e6, 500e6]
  # opt 3:
  # lower_bounds = [  50,  0.1,   1e3,  200e6,   5e8]
  # upper_bounds = [ 200,   10, 100e3, 1000e6, 100e8]
  # opt 4:
  lower_bounds = [  50,  0.1,    1e2,    20e6, 0.5e9]
  upper_bounds = [ 400,  100, 1000e2,  500e6,  100e9]


  lower_bounds = np.array(lower_bounds)
  upper_bounds = np.array(upper_bounds)
  matcfg_input = lower_bounds + (upper_bounds - lower_bounds) * (bayesOpt_input - (0)) / (+1 - (0))
  return matcfg_input

def parseInput(matcfg_input, txtcfg):
  # unpack values and scale to units
  gdot0_slip    = 0.001 # reference: 0.001
  n_slip        = matcfg_input[0]
  a_slip        = matcfg_input[1]
  tau0_slip     = matcfg_input[2]
  tausat_slip   = matcfg_input[3]
  h0_slipslip   = matcfg_input[4]
  parsed_txtcfg = txtcfg # work on a copied version
  # change lines: always add '\n' at the end of the line
  # parsed_txtcfg[48 - 1] = 'gdot0_slip              0.001\n'
  # parsed_txtcfg[49 - 1] = 'n_slip                  83.3\n'
  # parsed_txtcfg[50 - 1] = 'a_slip                  2.25\n'
  # parsed_txtcfg[51 - 1] = 'tau0_slip               95.e6\n'
  # parsed_txtcfg[52 - 1] = 'tausat_slip             222.e6\n'
  # parsed_txtcfg[53 - 1] = 'h0_slipslip             1.0e6\n'
  parsed_txtcfg[60 - 1] = 'gdot0_slip              %.4f\n'  % gdot0_slip
  parsed_txtcfg[61 - 1] = 'n_slip                  %.12e\n' % n_slip
  parsed_txtcfg[62 - 1] = 'a_slip                  %.12e\n' % a_slip
  parsed_txtcfg[63 - 1] = 'tau0_slip               %.12e\n' % tau0_slip
  parsed_txtcfg[64 - 1] = 'tausat_slip             %.12e\n' % tausat_slip
  parsed_txtcfg[65 - 1] = 'h0_slipslip             %.12e\n' % h0_slipslip
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

