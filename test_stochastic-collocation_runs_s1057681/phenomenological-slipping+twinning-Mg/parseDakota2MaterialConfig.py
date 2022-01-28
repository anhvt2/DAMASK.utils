
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
  # [tau0_basal, tau0_pris, tau0_pyr_a, tau0_pyr_ca, tau0_T1, tau0_C2, tausat_basal, tausat_pris, tausat_pyr_a, tausat_pyr_ca, h0_twintwin, h0_slipslip, h0_twinslip, n_twin, n_slip, a_slip]
  lower_bounds = [ 5, 30, 50,  50, 35,  60, 30, 100, 120, 120, 30, 100, 400, 15e-6, 3e-6, 2e-6] # last 3 indices do not have unit: add '* 1e-6' to counter '* 1e6'
  upper_bounds = [30, 60, 90, 110, 70, 120, 60, 160, 180, 180, 80, 200, 680, 35e-6, 8e-6, 4e-6] # last 3 indices do not have unit: add '* 1e-6' to counter '* 1e6'
  sg_input     = np.array(sg_input)[2:] # ignore 'id' and 'weights' two columns
  lower_bounds = np.array(lower_bounds) * 1e6 # last 3 indices do not have unit: add '* 1e-6' to counter '* 1e6'
  upper_bounds = np.array(upper_bounds) * 1e6 # last 3 indices do not have unit: add '* 1e-6' to counter '* 1e6'
  real_sg_input = lower_bounds + (upper_bounds - lower_bounds) * (sg_input - (-1)) / (+1 - (-1))
  return real_sg_input

def parseInput(real_sg_input, txtcfg):
  # unpack values and scale to units
  tau0_basal    = real_sg_input[0]
  tau0_pris     = real_sg_input[1]
  tau0_pyr_a    = real_sg_input[2]
  tau0_pyr_ca   = real_sg_input[3]
  tau0_T1       = real_sg_input[4]
  tau0_C2       = real_sg_input[5]
  tausat_basal  = real_sg_input[6]
  tausat_pris   = real_sg_input[7]
  tausat_pyr_a  = real_sg_input[8]
  tausat_pyr_ca = real_sg_input[9]
  h0_twintwin   = real_sg_input[10]
  h0_slipslip   = real_sg_input[11]
  h0_twinslip   = real_sg_input[12]
  n_twin        = real_sg_input[13]
  n_slip        = real_sg_input[14]
  a_slip        = real_sg_input[15]
  parsed_txtcfg = txtcfg # work on a copied version
  # change lines: always add '\n' at the end of the line
  # parsed_txtcfg[61 - 1] = 'tau0_slip               10.0e6  55.0e6  0   60.0e6  0.0   60.0e6     #  - " - table 1, pyr(a) set to pyr(c+a)\n'  
  # parsed_txtcfg[62 - 1] = 'tausat_slip             40.0e6 135.0e6  0  150.0e6  0.0  150.0e6     #  - " - table 1, pyr(a) set to pyr(c+a)\n'
  # parsed_txtcfg[64 - 1] = 'tau0_twin              40e6  0.0  0.0  60.0e6                        #  - " - table 1, compressive twin guessed by Steffi, tensile twin modified to match \n'
  # parsed_txtcfg[66 - 1] = 'h0_twintwin           50.0e6                                         #  - " - table 1, same range as theta_0\n'
  # parsed_txtcfg[67 - 1] = 'h0_slipslip          500.0e6                                         #  - " - table 1, same range as theta_0\n'
  # parsed_txtcfg[68 - 1] = 'h0_twinslip          150.0e6                                         # guessing\n'
  # parsed_txtcfg[84 - 1] = 'n_twin                  20\n'
  # parsed_txtcfg[85 - 1] = 'n_slip                  20\n'
  # parsed_txtcfg[95 - 1] = 'a_slip                  2.25\n'

  parsed_txtcfg[61 - 1] = 'tau0_slip               %.12e %.12e  0  %.12e  0.0  %.12e     #  - " - table 1, pyr(a) set to pyr(c+a)\n'  % (tau0_basal, tau0_pris, tau0_pyr_a, tau0_pyr_ca)
  parsed_txtcfg[62 - 1] = 'tausat_slip             %.12e %.12e  0  %.12e  0.0  %.12e     #  - " - table 1, pyr(a) set to pyr(c+a)\n' % (tausat_basal, tausat_pris, tausat_pyr_a, tausat_pyr_ca)
  parsed_txtcfg[64 - 1] = 'tau0_twin               %.12e  0.0  0.0  %.12e                        #  - " - table 1, compressive twin guessed by Steffi, tensile twin modified to match \n' % (tau0_T1, tau0_C2)
  parsed_txtcfg[66 - 1] = 'h0_twintwin          %.12e                                         #  - " - table 1, same range as theta_0\n' % h0_twintwin
  parsed_txtcfg[67 - 1] = 'h0_slipslip          %.12e                                         #  - " - table 1, same range as theta_0\n' % h0_slipslip
  parsed_txtcfg[68 - 1] = 'h0_twinslip          %.12e                                         # guessing\n' % h0_twinslip
  parsed_txtcfg[84 - 1] = 'n_twin                  %.12e\n' % n_twin
  parsed_txtcfg[85 - 1] = 'n_slip                  %.12e\n' % n_slip
  parsed_txtcfg[95 - 1] = 'a_slip                  %.12e\n' % a_slip
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


