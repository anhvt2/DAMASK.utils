
import numpy as np
import os, sys, time, glob, datetime

## read input from dummy dakota output (with the same dimensionality)
sg_input_list = np.loadtxt('dakota_sparse_tabular.dat', skiprows=1)
n = len(sg_input_list)

matcfgFile = open('material.config') # use 'materials.config' as the template file
txtcfg = matcfgFile.readlines()
matcfgFile.close()

parentPath = os.getcwd()

### NOTE: for new case study, modify getDamaskParams() and parseInput()
def getDamaskParams(sg_input):
  # translate dakota input from [-1,1] to real bounds imposed by user
  # [tau0_slip, tausat_slip, h0, n, a]
  # lower_bounds = [1e10, 1e2, 1.64, 0.25, 1.2, 1.5e-19,  5] # 7d
  # upper_bounds = [5e12, 1e4, 2.42, 0.70, 1.8, 3.5e-19, 20] # 7d
  lower_bounds = [1e10, 1.64, 0.25, 1.2, 1.5e-19,  5] # 6d
  upper_bounds = [5e12, 2.42, 0.70, 1.8, 3.5e-19, 20] # 6d
  sg_input     = np.array(sg_input)[2:] # ignore 'id' and 'weights' two columns
  lower_bounds = np.array(lower_bounds)
  upper_bounds = np.array(upper_bounds)
  real_sg_input = lower_bounds + (upper_bounds - lower_bounds) * (sg_input - (-1)) / (+1 - (-1))
  return real_sg_input

def parseInput(real_sg_input, txtcfg):
  # unpack values and scale to units
  rhoedge0      = real_sg_input[0]
  # v0            = real_sg_input[1]
  tau_peierls   = real_sg_input[1] 
  p_slip        = real_sg_input[2]
  q_slip        = real_sg_input[3]
  Qsd           = real_sg_input[4]
  CLambdaSlip   = real_sg_input[5]
  parsed_txtcfg = txtcfg # work on a copied version
  # change lines: always add '\n' at the end of the line
  # parsed_txtcfg[55 - 1] = 'rhoedge0            1.0e12          # Initial edge dislocation density [m/m**3]\n'
  # parsed_txtcfg[57 - 1] = 'v0                  1.0e-4          # Initial glide velocity [m/s]: 1.0e-4\n'
  # parsed_txtcfg[61 - 1] = 'tau_peierls         2.03e9          # peierls stress (for bcc)\n'
  # parsed_txtcfg[59 - 1] = 'p_slip              0.78            # p-exponent in glide velocity\n'
  # parsed_txtcfg[60 - 1] = 'q_slip              1.58            # q-exponent in glide velocity\n'
  # parsed_txtcfg[68 - 1] = 'Qsd                 4.5e-19         # Activation energy for climb [J]\n'
  # parsed_txtcfg[65 - 1] = 'CLambdaSlip         10.0            # Adj. parameter controlling dislocation mean free path\n'

  parsed_txtcfg[55 - 1] = 'rhoedge0            %.12e           # Initial edge dislocation density [m/m**3]\n' % rhoedge0
  # parsed_txtcfg[57 - 1] = 'v0                  %.12e           # Initial glide velocity [m/s]: old value "1.0e-4: corrected to be "1.0e4", cf. Sedighiani et al. 2022 Mech. of Matls.\n' % v0
  parsed_txtcfg[61 - 1] = 'tau_peierls         %.12e           # peierls stress (for bcc)\n' % tau_peierls
  parsed_txtcfg[59 - 1] = 'p_slip              %.12e           # p-exponent in glide velocity\n' % p_slip
  parsed_txtcfg[60 - 1] = 'q_slip              %.12e           # q-exponent in glide velocity\n' % q_slip
  parsed_txtcfg[68 - 1] = 'Qsd                 %.12e           # Activation energy for climb [J]\n' % Qsd
  parsed_txtcfg[65 - 1] = 'CLambdaSlip         %.12e           # Adj. parameter controlling dislocation mean free path\n' % CLambdaSlip
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
  os.system('ln -sf ../single_phase_equiaxed.geom .')
  os.system('ln -sf ../tension.load .')
  os.system('ln -sf ../run_damask.sh .')
  os.system('ln -sf ../sbatch.damask.solo .')
  ## write new 'material.config' locally
  sg_input = sg_input_list[i] # get input from dakota
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



