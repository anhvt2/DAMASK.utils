
# Uncertainty quantification for constitutive models in CPFEM

## Phenomenological model

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

## Dislocation-density-based model

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

## Computational costs

Note: `tension.load` are set at `time 200 inc 200`, `test-default/` with default `material.config` is used to benchmark

* `phenomenological-slipping-Cu`: 1:02:00 computation, ~4:00:00 post-processing
* `phenomenological-slipping+twinning`: 0:40:00 computation, 3:03:28 post-processing
* `dislocation-density-W:` 0:12:00 computation, 00:23:10 post-processing


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

