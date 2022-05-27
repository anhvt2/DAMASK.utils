
# Uncertainty quantification for constitutive models in CPFEM

## Phenomenological model (fcc Cu)

* Dakota post-process is located in `textbook5d_uq_sc/`
* See `phenomenological/` 
* Use `Phase_Phenopowerlaw_Copper.config` as an example, vary 5 parameters according to `sedighiani2020efficient`, similar to Table 2.

```
@article{sedighiani2020efficient,
  title={An efficient and robust approach to determine material parameters of crystal plasticity constitutive laws from macro-scale stress--strain curves},
  author={Sedighiani, Karo and Diehl, Martin and Traka, Konstantina and Roters, F and Sietsma, Jilt and Raabe, Dierk},
  journal={International Journal of Plasticity},
  volume={134},
  pages={102779},
  year={2020},
  publisher={Elsevier}
}
```

* DREAM.3D file is generated using `Copper`-type of texture

## Phenomenological model (hcp Mg)
* Dakota post-process is located in `textbook16d_uq_sc/`

## Dislocation-density-based model (bcc W)

* Dakota post-process is located in `textbook7d_uq_sc/`
* See `dislocation-density-based/`
* Use `Phase_Phenopowerlaw_Magnesium.config` as an example, vary 16 parameters according to `sedighiani2020efficient`, similar to Table 8.

```
@article{sedighiani2020efficient,
  title={An efficient and robust approach to determine material parameters of crystal plasticity constitutive laws from macro-scale stress--strain curves},
  author={Sedighiani, Karo and Diehl, Martin and Traka, Konstantina and Roters, F and Sietsma, Jilt and Raabe, Dierk},
  journal={International Journal of Plasticity},
  volume={134},
  pages={102779},
  year={2020},
  publisher={Elsevier}
}
```

* DREAM.3D file is adopted from PRISMS-Plasticity `PRISMS/plasticity/Training_Materials/Pre-Processing/HCP`, mainly from `PRISMS_pipeline_hcp.json` (after updating paths and adding a DAMASK export filter)
  * `PRISMS_pipeline_hcp.json`
  * `Magnesium.xdmf`
  * `grainID.txt`
  * `polefigure_initial_Mg.m`
  * `plotpolefromebsd_mg.m`
  * `micro_snap.png`
  * `mgdata.txt`
  * `polefigures_exp`
  * `polefigures_rve`


## Roadmap for using Dakota (dakota.sandia.gov)


### Pre-run

Steps:

1. Create a dummy Dakota input file with the appropriate parameters, domains and distributions for the actual model.
2. Run the dummy input file and save the input samples: 
  * create a `textbook` example with the **same** dimensionality with the problem considered
  * `../../test/dakota -i rosen_uq_sc.in  > dakota.log`
  * see `dakota_sparse_tabular.dat` for list of inputs used
3. Run the CPFEM code using these sample points: 
  * run `parseDakota2MaterialConfig.py`: this will create a list of folders that has `material.config` changing accordingly to Dakota inputs in `dakota_sparse_tabular.dat`
4. Create the actual Dakota input file that just reads in the sample data and does not call anything else.

Example: `dakota-6.15/build/test/examples-users/rosen_uq_sc.in`

```
# Dakota Input File: rosen_uq_sc.in

environment

method
  stoch_collocation
    sparse_grid_level = 3 # subject to change
    # dimension_preference = 2 1 # switch from anisotropic to isotropic SG
    samples_on_emulator = 10000 seed = 12347
    response_levels = .1 1. 50. 100. 500. 1000.
    variance_based_decomp #interaction_order = 1
  # output silent
  output verbose # print input parameters

variables
  uniform_uncertain = 2 # Legendre polynomial
    lower_bounds      = -1.0  +1.0
    upper_bounds      = -1.0  +1.0
    descriptors       = 'x1'  'x2'

interface
  analysis_drivers = 'textbook'
    direct

responses
  response_functions = 1
  no_gradients
  no_hessians
```

```shell
cd /home/anhvt89/Documents/dakota/6.15/build/test/examples-users
../../test/dakota -i rosen_uq_sc.in  > dakota.log
```

For input parameters, see `dakota_sparse_tabular.dat` file generated from revoking the above command.

### Import results -- post-run

To import the results, build a Python interface script. In this example, we use `damask_query.py` to look up results from a structured table.

```
# Dakota Input File: rosen_uq_sc.in

environment

method
  stoch_collocation
    sparse_grid_level = 3 # subject to change
    # dimension_preference = 2 1 # switch from anisotropic to isotropic SG
    samples_on_emulator = 100000 seed = 12347
    response_levels = .1 1. 50. 100. 500. 1000.
    variance_based_decomp #interaction_order = 1
  # output silent
  output verbose # print input parameters

variables
  uniform_uncertain = 7 # Legendre polynomial
    lower_bounds      = -1.0    -1.0    -1.0    -1.0    -1.0    -1.0     -1.0 
    upper_bounds      = +1.0    +1.0    +1.0    +1.0    +1.0    +1.0     +1.0 
    descriptors       = 'x1'    'x2'    'x3'    'x4'    'x5'    'x6'     'x7' 

interface
  fork
  analysis_drivers = 'python3 damask_query.py'
  file_tag file_save
    # direct

responses
  response_functions = 1
  no_gradients
  no_hessians
```

For the sake of completeness, `damask_query.py` is documented right below as well.
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
adopted from dakota/6.15/dakota-6.15.0-public-src-cli/dakota-examples/official/global_sensitivity/Ishigami.py
"""
from __future__ import division, print_function, unicode_literals, absolute_import
from io import open

from math import sin
import sys
import numpy as np

params_file, output_file = sys.argv[1:]

# print('%s' % params_file)

i_ = np.empty(7)
with open(params_file,'rt',encoding='utf8') as FF:
  d = FF.readlines() # Skip first line
  # print(len(d))
  # print(FF.readline())
  i_[0] = float(d[1].split()[0])
  i_[1] = float(d[2].split()[0])
  i_[2] = float(d[3].split()[0])
  i_[3] = float(d[4].split()[0])
  i_[4] = float(d[5].split()[0])
  i_[5] = float(d[6].split()[0])
  i_[6] = float(d[7].split()[0])

# print(i_)

# required
inputData  = np.loadtxt('dakota_sparse_tabular.dat',skiprows=1)[:,2:]
outputData = np.loadtxt('output.dat',delimiter=',')

def searchIndex(i_, inputData):
  n, d = inputData.shape
  tol = 1e-8
  index_ = np.where(np.linalg.norm(inputData - i_, axis=1) < tol)[0][0]
  return index_

index_ = searchIndex(i_, inputData)
o_ = outputData[index_, 0] # change the second index accordingly

print(i_, o_, index_)

outFile = open(output_file, 'w')
outFile.write('%.12e' % (o_))
outFile.close()
```

## `material.config`

A clean way to modify `material.config`

```
### numerical parameters ###

# The material.config file needs to specify five parts:
# homogenization, microstructure, crystallite, phase, and texture.
# You can either put the full text in here or include suited separate files

<homogenization>
{./Homogenization_Isostrain_SX.config}

<microstructure>
{./Microstructure_ElementHomogeneous.config}

<crystallite>
{./Crystallite_aLittleSomething.config}

<phase>
{./Phase_Phenopowerlaw_Aluminum.config}

<texture>
{./Texture_Rolling.config}
```

### Step to run post-processing

1. Change `damask_query.py`
```python
o_ = outputData[index_, 0] # change the second index accordingly: 0 = strainYield, 1 = stressYield
```
2. run `test.sh`: (a) change `sparse_grid_level` in Dakota input script, (b) file name (e.g. `stressYield_level1.dat`), and (c) index in `damask_query.py`
```shell
../dakota -i textbook7d_uq_sc_pyImport.in > dakota.log
grep -inr ' f1' dakota.log  > tmp.txt
sed -i  's/ f1//g' tmp.txt
mv tmp.txt stressYield_level1.dat # or strainYield.dat -- depending on how damask_query.py is configured
```
3. 
```shell
python3 plotQoIpdf.py --file=stressYield.dat
```
4. Copy the variance-based global sensitivity analysis to 
* `{strain,stress}Yield.dakota.log`
* `plotSobolIndices.py`: must reformat to suit the needs
5. Run `plotSobolIndices.py` for parameter sensitivity


## Computational costs

Note: `tension.load` are set at `time 200 inc 200`, `test-default/` with default `material.config` is used to benchmark

* `phenomenological-slipping-Cu`: 1:02:00 computation, ~4:00:00 post-processing
* `phenomenological-slipping+twinning-Mg`: 0:40:00 computation, 3:03:28 post-processing
* `dislocation-density-W:` 0:12:00 computation, 00:23:10 post-processing

## Parallel post-processing

An interesting observation is that
* post-processing does not increase memory usage
* post-processing can be parallelized -- at the cost of CPU (not memory)
* can be performed using `nohup`
* an example of how to do it:
```shell
cd sg_input_?
nohup bash ../post_process.sh 2>&1 > log.post_process &
cd ..
```
* `post_process.sh` Bash/Shell script as follows, will remove `*.spectralOut` by the end of the script
```shell
#!/bin/bash
postResults single_phase_equiaxed_tension.spectralOut --cr f,p

if [ -d "postProc" ]; then
  cd postProc/

  filterTable < single_phase_equiaxed_tension.txt --white inc,1_f,1_p > stress_strain.log
  # addStrainTensors one_phase_equiaxed_tension.txt --left --logarithmic
  # addCauchy one_phase_equiaxed_tension.txt
  # addMises one_phase_equiaxed_tension.txt --strain 'ln(V)' --stress Cauchy
  # filterTable < one_phase_equiaxed_tension.txt --white inc,'Mises(ln(V))','Mises(Cauchy)' > log.stress_strain.txt

  cp ../tension.load . 
  # check $1 argument in running this script, i.e. bash run_damask.sh $1
  if [[ $1 == "YieldStress" ]]; then 
    python3 ../computeYieldStress.py
  elif [[ $1 == "YoungModulus" ]]; then
    python3 ../computeYoungModulus.py
  else
    echo "run_damask_2.0.3.sh: \$1 argument is not detected in run_damask_2.0.3.sh"
  fi

  if [ -f "output.dat" ]; then
    echo 1 > ../log.feasible
    # needed in wrapper_DREAM3D-DAMASK.py
  fi
  cd ..
else
  echo "Simulation does not converge!!!"
  echo 0 > log.feasible
fi
rm -v *.spectralOut
```


## Accuracy with default parameters

* `phenomenological-slipping-Cu`:
```
Elastic Young modulus = 188.5552 GPa
Intercept = -0.0000
Intersection detected:  (0.0023092829648229344, 58316919.44272391)
##########
Intersection with Young modulus (obtained from linear regression) with $\sigma-\varepsilon$ occured at:
Yield Strain = 0.0023
Yield Stress = 0.0583 GPa
```

```
 c12                     122.1e9
 c44                     75.7e9
 
-gdot0_slip              0.001
-n_slip                  83.3
-tau0_slip               16.0e6                # per family
-tausat_slip             148.0e6               # per family
-a_slip                  2.25
-gdot0_twin              3e-3 # old value: 0.001
+gdot0_slip              3.0e-3 # 0.001
+n_slip                  20.0
+tau0_slip               1.5e6                 # per family
+tausat_slip             112.5e6               # per family
+a_slip                  2.0
+gdot0_twin              1.0e-3 # old value: 0.001
 n_twin                  20
 h0_slipslip             2.4e8 # old value: 180e6
 interaction_slipslip    1 1 1.4 1.4 1.4 1.4
 atol_resistance         1
 
-
+# Config-3.0.0.alpha5/
+#  - T Takeuchi,
+#    Transactions of the Japan Institute of Metals 16(10):629-640, 1975,
+#    https://doi.org/10.2320/matertrans1960.16.629,
+#    fitted from Fig. 3b
+#  - U.F. Kocks,
+#    Metallurgical and Materials Transactions B 1:1121–1143, 1970,
+#    https://doi.org/10.1007/BF02900224
 #-------------------#
 <texture>
 #-------------------#
```

* `phenomenological-slipping+twinning-Mg`:
```
Elastic Young modulus = 45.1172 GPa
Intercept = -0.0000
Intersection detected:  (0.004375425049107471, 107172420.78661838)
##########
Intersection with Young modulus (obtained from linear regression) with $\sigma-\varepsilon$ occured at:
Yield Strain = 0.0044
Yield Stress = 0.1072 GPa
```

* `dislocation-density-based-W`:
```
Elastic Young modulus = 411.3528 GPa
Intercept = -0.0000
Intersection detected:  (0.010984068654145371, 3695621665.287616)
##########
Intersection with Young modulus (obtained from linear regression) with $\sigma-\varepsilon$ occured at:
Yield Strain = 0.0110
Yield Stress = 3.6956 GPa
```

